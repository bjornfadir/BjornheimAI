"""Tests for the guardian transcript deep-dive read + summarize path."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from kai.config import Config, UserConfig
from kai.oneshot import OneShotResult, OneShotTimeout
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.domain import ChannelId, PrincipalId, RuntimeProfileId
from kai.workshop.guardian_access import resolve_guardian_target
from kai.workshop.guardian_transcript import (
    _fallback_summary,
    _format_transcript,
    read_target_transcript,
    summarize_transcript,
)
from kai.workshop.inbound import InboundMessage, record_inbound_message
from kai.workshop.store import WorkshopEventStore

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


async def _open_store_with_kid(path: Path) -> tuple[WorkshopEventStore, UserConfig, UserConfig]:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman(
                "Parent",
                "admin",
                "telegram",
                "101",
                "101",
                RuntimeProfileId("rtp_11111111111111111111111111111111"),
            ),
            BootstrapHuman(
                "Kid",
                "member",
                "discord",
                "202",
                "202",
                RuntimeProfileId("rtp_22222222222222222222222222222222"),
            ),
        ),
    )
    guardian = UserConfig(name="Parent", telegram_id=101, guardian_of=["Kid"])
    kid = UserConfig(name="Kid", discord_id=202, monitored=True)
    return store, guardian, kid


async def _identity_for(store: WorkshopEventStore, provider: str, subject: str) -> tuple[PrincipalId, ChannelId]:
    async with store.connection.execute(
        "SELECT e.principal_id, b.channel_id FROM external_identities e "
        "JOIN channel_bindings b ON b.transport = e.provider "
        "AND b.external_channel_id = e.external_subject "
        "WHERE e.provider = ? AND e.external_subject = ?",
        (provider, subject),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return PrincipalId(str(row[0])), ChannelId(str(row[1]))


class TestReadTargetTranscript:
    async def test_reads_the_target_kids_messages_regardless_of_caller_identity(self, tmp_path: Path) -> None:
        store, guardian, kid = await _open_store_with_kid(tmp_path / "kai.db")
        try:
            await record_inbound_message(
                store,
                InboundMessage("discord", "u1", "m1", "202", "202", "hi from kid", _NOW),
            )
            from kai.workshop.domain import WorkshopId

            # Resolve the real workshop_id off the bootstrapped kid identity
            # rather than fabricating one - bootstrap_default_workshop's
            # derived IDs only match a workshop_id actually used to seed it.
            _principal_id, channel_id = await _identity_for(store, "discord", "202")
            async with store.connection.execute("SELECT workshop_id FROM channels WHERE id = ?", (channel_id,)) as cur:
                row = await cur.fetchone()
            assert row is not None
            workshop_id = WorkshopId(str(row[0]))

            target = resolve_guardian_target(workshop_id, (guardian, kid), guardian, "Kid")
            messages = await read_target_transcript(store, target, limit=10)

            assert len(messages) == 1
            assert messages[0].body == "hi from kid"
            assert messages[0].author_kind == "human"
        finally:
            await store.close()

    async def test_no_messages_yields_empty_list(self, tmp_path: Path) -> None:
        store, guardian, kid = await _open_store_with_kid(tmp_path / "kai.db")
        try:
            _principal_id, channel_id = await _identity_for(store, "discord", "202")
            async with store.connection.execute("SELECT workshop_id FROM channels WHERE id = ?", (channel_id,)) as cur:
                row = await cur.fetchone()
            from kai.workshop.domain import WorkshopId

            assert row is not None
            workshop_id = WorkshopId(str(row[0]))
            target = resolve_guardian_target(workshop_id, (guardian, kid), guardian, "Kid")

            messages = await read_target_transcript(store, target, limit=10)

            assert messages == []
        finally:
            await store.close()


class TestFormatAndFallback:
    def test_format_transcript_labels_speakers_by_kind(self) -> None:
        from kai.workshop.timeline import TimelineMessage

        msg = TimelineMessage(
            message_id="msg_1",  # type: ignore[arg-type]
            channel_id="chn_1",  # type: ignore[arg-type]
            author_principal_id="prn_1",  # type: ignore[arg-type]
            author_kind="human",
            author_display_name="Kid",
            reply_to_message_id=None,
            body="hello",
            event_position=1,
            created_at=_NOW,
        )
        text = _format_transcript("Kid", [msg])
        assert "Kid: hello" in text

    def test_fallback_summary_on_empty_history(self) -> None:
        assert "no recorded conversation" in _fallback_summary("Kid", [])


class TestSummarizeTranscript:
    async def test_falls_back_when_the_reasoner_times_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import kai.workshop.guardian_transcript as gt_module
        from kai.workshop.timeline import TimelineMessage

        async def _raise_timeout(self, **kwargs):
            raise OneShotTimeout("guardian_transcript_summary", 45.0)

        monkeypatch.setattr(gt_module.ClaudeOneShotReasoner, "run", _raise_timeout)

        msg = TimelineMessage(
            message_id="msg_1",  # type: ignore[arg-type]
            channel_id="chn_1",  # type: ignore[arg-type]
            author_principal_id="prn_1",  # type: ignore[arg-type]
            author_kind="human",
            author_display_name="Kid",
            reply_to_message_id=None,
            body="hello",
            event_position=1,
            created_at=_NOW,
        )
        guardian = UserConfig(name="Parent", telegram_id=101)
        config = Config(
            telegram_bot_token="unused",
            allowed_user_ids=set(),
            session_db_path=Path(":memory:"),
            agent_idle_timeout=0,
            default_backend="codex",
            default_model="gpt-5.6-sol",
        )

        summary = await summarize_transcript(
            [msg], target_name="Kid", guardian_config=guardian, config=config
        )

        assert "unsummarized" in summary
        assert "hello" in summary

    async def test_returns_model_text_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import kai.workshop.guardian_transcript as gt_module
        from kai.workshop.timeline import TimelineMessage

        async def _fake_run(self, **kwargs):
            return OneShotResult(text="All quiet, nothing concerning.", backend="claude", model="haiku")

        monkeypatch.setattr(gt_module.ClaudeOneShotReasoner, "run", _fake_run)

        msg = TimelineMessage(
            message_id="msg_1",  # type: ignore[arg-type]
            channel_id="chn_1",  # type: ignore[arg-type]
            author_principal_id="prn_1",  # type: ignore[arg-type]
            author_kind="human",
            author_display_name="Kid",
            reply_to_message_id=None,
            body="hello",
            event_position=1,
            created_at=_NOW,
        )
        guardian = UserConfig(name="Parent", telegram_id=101)
        config = Config(
            telegram_bot_token="unused",
            allowed_user_ids=set(),
            session_db_path=Path(":memory:"),
            agent_idle_timeout=0,
            default_backend="codex",
            default_model="gpt-5.6-sol",
        )

        summary = await summarize_transcript(
            [msg], target_name="Kid", guardian_config=guardian, config=config
        )

        assert summary == "All quiet, nothing concerning."

    async def test_empty_history_skips_the_model_call_entirely(self) -> None:
        guardian = UserConfig(name="Parent", telegram_id=101)
        config = Config(
            telegram_bot_token="unused",
            allowed_user_ids=set(),
            session_db_path=Path(":memory:"),
            agent_idle_timeout=0,
            default_backend="codex",
            default_model="gpt-5.6-sol",
        )

        summary = await summarize_transcript([], target_name="Kid", guardian_config=guardian, config=config)

        assert "no recorded conversation" in summary
