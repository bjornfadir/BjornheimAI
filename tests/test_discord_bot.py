"""Tests for discord_bot.py's guardian /monitor command logic.

Exercises `_guardian_transcript_reply` directly rather than going through
discord.py's Interaction/CommandTree machinery - see that function's
docstring for why it was split out of the `monitor` slash-command closure.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from kai.config import Config, UserConfig
from kai.discord_bot import _guardian_transcript_reply
from kai.oneshot import OneShotResult
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.domain import ChannelId, WorkshopId
from kai.workshop.inbound import InboundMessage, record_inbound_message
from kai.workshop.store import WorkshopEventStore

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


async def _identity_for(store: WorkshopEventStore, provider: str, subject: str) -> ChannelId:
    async with store.connection.execute(
        "SELECT b.channel_id FROM external_identities e "
        "JOIN channel_bindings b ON b.transport = e.provider "
        "AND b.external_channel_id = e.external_subject "
        "WHERE e.provider = ? AND e.external_subject = ?",
        (provider, subject),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return ChannelId(str(row[0]))


async def _open_store_with_family(path: Path) -> tuple[WorkshopEventStore, WorkshopId]:
    from kai.workshop.domain import RuntimeProfileId

    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman(
                "Parent", "admin", "telegram", "101", "101", RuntimeProfileId("rtp_11111111111111111111111111111111")
            ),
            BootstrapHuman(
                "Kid", "member", "discord", "202", "202", RuntimeProfileId("rtp_22222222222222222222222222222222")
            ),
            BootstrapHuman(
                "Stranger",
                "member",
                "discord",
                "303",
                "303",
                RuntimeProfileId("rtp_33333333333333333333333333333333"),
            ),
        ),
    )
    channel_id = await _identity_for(store, "discord", "202")
    async with store.connection.execute("SELECT workshop_id FROM channels WHERE id = ?", (channel_id,)) as cur:
        row = await cur.fetchone()
    assert row is not None
    return store, WorkshopId(str(row[0]))


def _config(*users: UserConfig) -> Config:
    return Config(
        telegram_bot_token="unused",
        allowed_user_ids=set(),
        session_db_path=Path(":memory:"),
        agent_idle_timeout=0,
        default_backend="codex",
        default_model="gpt-5.6-sol",
        user_configs={u.config_id: u for u in users},
    )


@pytest.fixture(autouse=True)
def _stub_summarizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test here cares about authorization/read plumbing, not the
    model call itself - stub it to a fixed, fast, deterministic value."""
    import kai.workshop.guardian_transcript as gt_module

    async def _fake_run(self, **kwargs):
        return OneShotResult(text="stub summary", backend="claude", model="haiku")

    monkeypatch.setattr(gt_module.ClaudeOneShotReasoner, "run", _fake_run)


class TestGuardianTranscriptReply:
    async def test_authorized_guardian_gets_a_summary(self, tmp_path: Path) -> None:
        store, workshop_id = await _open_store_with_family(tmp_path / "kai.db")
        try:
            await record_inbound_message(
                store, InboundMessage("discord", "u1", "m1", "202", "202", "hi", _NOW)
            )
            guardian = UserConfig(name="Parent", telegram_id=101, guardian_of=["Kid"])
            kid = UserConfig(name="Kid", discord_id=202, monitored=True)
            config = _config(guardian, kid)
            core_services = SimpleNamespace(workshop_id=workshop_id, client_store=store)

            reply = await _guardian_transcript_reply(
                core_services=core_services,  # type: ignore[arg-type]
                config=config,
                guardian_config=guardian,
                target_name="Kid",
                count=10,
            )

            assert reply == "stub summary"
        finally:
            await store.close()

    async def test_caller_not_listed_as_guardian_is_refused(self, tmp_path: Path) -> None:
        store, workshop_id = await _open_store_with_family(tmp_path / "kai.db")
        try:
            not_a_guardian = UserConfig(name="Stranger", discord_id=303)
            kid = UserConfig(name="Kid", discord_id=202, monitored=True)
            config = _config(not_a_guardian, kid)
            core_services = SimpleNamespace(workshop_id=workshop_id, client_store=store)

            reply = await _guardian_transcript_reply(
                core_services=core_services,  # type: ignore[arg-type]
                config=config,
                guardian_config=not_a_guardian,
                target_name="Kid",
                count=10,
            )

            assert reply == "Not found, or you are not a configured guardian of that name."
        finally:
            await store.close()

    async def test_nonexistent_target_name_gets_the_identical_refusal(self, tmp_path: Path) -> None:
        """Same wording as the not-a-guardian case - GuardianAccessError does
        not distinguish "not your ward" from "doesn't exist" (fail-closed,
        see guardian_access.GuardianAccessError's docstring)."""
        store, workshop_id = await _open_store_with_family(tmp_path / "kai.db")
        try:
            guardian = UserConfig(name="Parent", telegram_id=101, guardian_of=["Kid"])
            config = _config(guardian)
            core_services = SimpleNamespace(workshop_id=workshop_id, client_store=store)

            reply = await _guardian_transcript_reply(
                core_services=core_services,  # type: ignore[arg-type]
                config=config,
                guardian_config=guardian,
                target_name="NoSuchPerson",
                count=10,
            )

            assert reply == "Not found, or you are not a configured guardian of that name."
        finally:
            await store.close()

    async def test_target_with_no_history_does_not_crash(self, tmp_path: Path) -> None:
        store, workshop_id = await _open_store_with_family(tmp_path / "kai.db")
        try:
            guardian = UserConfig(name="Parent", telegram_id=101, guardian_of=["Kid"])
            kid = UserConfig(name="Kid", discord_id=202, monitored=True)
            config = _config(guardian, kid)
            core_services = SimpleNamespace(workshop_id=workshop_id, client_store=store)

            reply = await _guardian_transcript_reply(
                core_services=core_services,  # type: ignore[arg-type]
                config=config,
                guardian_config=guardian,
                target_name="Kid",
                count=10,
            )

            # No messages recorded for Kid in this test - summarize_transcript
            # short-circuits to its own fallback wording before ever calling
            # the (stubbed) reasoner.
            assert "no recorded conversation" in reply
        finally:
            await store.close()
