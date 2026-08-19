"""Contracts for guardian alert recording and delivery enqueue."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from kai.config import UserConfig
from kai.workshop.bootstrap import (
    BootstrapHuman,
    bootstrap_default_workshop,
    bootstrap_human_channel_id,
    bootstrap_human_principal_id,
)
from kai.workshop.delivery_outbox import NOTIFICATION_PURPOSE
from kai.workshop.domain import WorkshopId
from kai.workshop.guardian_access import GuardianIdentity
from kai.workshop.guardian_alerts import (
    GuardianAlert,
    GuardianAlertChannelNotFoundError,
    WorkshopGuardianAlertRecorder,
    send_guardian_alerts,
)
from kai.workshop.safety_classifier import SafetyCategory, SafetyClassification
from kai.workshop.store import IdempotencyConflictError, WorkshopEventStore

_NOW = datetime(2026, 8, 19, 12, 0, tzinfo=UTC)

_FLAGGED = SafetyClassification(
    flagged=True,
    category=SafetyCategory.SELF_HARM,
    summary="expressed distress",
)


async def _open_store(path: Path, *humans: BootstrapHuman) -> tuple[WorkshopEventStore, WorkshopId]:
    store = await WorkshopEventStore.open(path)
    result = await bootstrap_default_workshop(store, humans)
    return store, result.workshop_id


def _guardian_human(transport: str = "telegram", subject: str = "1") -> BootstrapHuman:
    return BootstrapHuman(
        display_name="parent",
        role="member",
        transport=transport,
        external_subject=subject,
        external_channel_id=subject,
    )


def _kid_human(transport: str = "discord", subject: str = "2") -> BootstrapHuman:
    return BootstrapHuman(
        display_name="kid",
        role="member",
        transport=transport,
        external_subject=subject,
        external_channel_id=subject,
    )


def _guardian_identity(workshop_id, *, transport: str = "telegram", subject: str = "1") -> GuardianIdentity:
    return GuardianIdentity(
        transport=transport,
        principal_id=bootstrap_human_principal_id(workshop_id, transport, subject),
        channel_id=bootstrap_human_channel_id(workshop_id, transport, subject),
    )


def _alert(alert_id: str = "run:test:0") -> GuardianAlert:
    return GuardianAlert(
        alert_id=alert_id,
        target_name="kid",
        category=SafetyCategory.SELF_HARM.value,
        summary="expressed distress",
        occurred_at=_NOW,
    )


class TestGuardianAlertInput:
    @pytest.mark.parametrize(
        ("changes", "match"),
        [
            ({"alert_id": "bad alert"}, "alert_id"),
            ({"target_name": ""}, "target_name"),
            ({"category": ""}, "category"),
            ({"summary": ""}, "summary"),
            ({"occurred_at": datetime(2026, 8, 19)}, "occurred_at"),
        ],
    )
    def test_invalid_input_fails_before_storage(self, changes, match):
        values = {
            "alert_id": "run:test:0",
            "target_name": "kid",
            "category": "self_harm",
            "summary": "expressed distress",
            "occurred_at": _NOW,
        }
        values.update(changes)
        with pytest.raises(ValueError, match=match):
            GuardianAlert(**values)


class TestWorkshopGuardianAlertRecorder:
    async def test_atomically_records_message_and_durable_delivery_work(self, tmp_path: Path) -> None:
        store, workshop_id = await _open_store(tmp_path / "kai.db", _guardian_human())
        try:
            identity = _guardian_identity(workshop_id)
            result = await WorkshopGuardianAlertRecorder(store).record(_alert(), identity)

            assert result.delivery.inserted is True
            assert result.delivery.delivery.purpose == NOTIFICATION_PURPOSE
            assert result.delivery.delivery.status == "pending"
            async with store.connection.execute(
                "SELECT c.kind, m.body FROM messages m "
                "JOIN channels c ON c.id = m.channel_id WHERE m.id = ?",
                (result.message_id,),
            ) as cursor:
                row = await cursor.fetchone()
            assert row[0] == "direct"
            assert "kid" in row[1]
            assert "expressed distress" in row[1]
        finally:
            await store.close()

    async def test_same_alert_id_is_idempotent(self, tmp_path: Path) -> None:
        store, workshop_id = await _open_store(tmp_path / "kai.db", _guardian_human())
        try:
            identity = _guardian_identity(workshop_id)
            recorder = WorkshopGuardianAlertRecorder(store)
            first = await recorder.record(_alert(), identity)
            second = await recorder.record(_alert(), identity)

            assert second.message_id == first.message_id
            assert second.delivery.inserted is False
            async with store.connection.execute(
                "SELECT COUNT(*) FROM delivery_outbox WHERE purpose = 'notification'"
            ) as cursor:
                assert (await cursor.fetchone())[0] == 1
        finally:
            await store.close()

    async def test_reused_alert_id_with_changed_content_fails_closed(self, tmp_path: Path) -> None:
        store, workshop_id = await _open_store(tmp_path / "kai.db", _guardian_human())
        try:
            identity = _guardian_identity(workshop_id)
            recorder = WorkshopGuardianAlertRecorder(store)
            await recorder.record(_alert(), identity)
            with pytest.raises(IdempotencyConflictError):
                await recorder.record(
                    GuardianAlert(
                        alert_id="run:test:0",
                        target_name="kid",
                        category="self_harm",
                        summary="a different summary",
                        occurred_at=_NOW,
                    ),
                    identity,
                )
        finally:
            await store.close()

    async def test_unbound_channel_raises_not_silently_drops(self, tmp_path: Path) -> None:
        store, workshop_id = await _open_store(tmp_path / "kai.db", _guardian_human())
        try:
            # No human was bootstrapped for this transport/subject - unlike
            # GitHub notifications (which can legitimately have no configured
            # destination and return None), a guardian alert always has a
            # specific, already-authorized target: reaching this path means
            # a bug upstream, so fail loudly rather than silently drop.
            unbound = _guardian_identity(workshop_id, transport="discord", subject="999")
            with pytest.raises(GuardianAlertChannelNotFoundError):
                await WorkshopGuardianAlertRecorder(store).record(_alert(), unbound)
        finally:
            await store.close()


class TestSendGuardianAlerts:
    async def test_monitored_user_with_one_guardian_delivers_one_alert(self, tmp_path: Path) -> None:
        store, workshop_id = await _open_store(
            tmp_path / "kai.db",
            _guardian_human(),
            _kid_human(),
        )
        try:
            guardian_config = UserConfig(name="parent", telegram_id=1, guardian_of=["kid"])
            kid_config = UserConfig(name="kid", discord_id=2, monitored=True)
            delivered = await send_guardian_alerts(
                store,
                workshop_id,
                [guardian_config, kid_config],
                kid_config,
                _FLAGGED,
                alert_id="run:test",
                occurred_at=_NOW,
            )
            assert delivered == 1
            async with store.connection.execute(
                "SELECT COUNT(*) FROM delivery_outbox WHERE purpose = 'notification'"
            ) as cursor:
                assert (await cursor.fetchone())[0] == 1
        finally:
            await store.close()

    async def test_monitored_user_with_two_guardians_delivers_two_alerts(self, tmp_path: Path) -> None:
        store, workshop_id = await _open_store(
            tmp_path / "kai.db",
            _guardian_human(subject="1"),
            BootstrapHuman(
                display_name="parent_b",
                role="member",
                transport="telegram",
                external_subject="3",
                external_channel_id="3",
            ),
            _kid_human(),
        )
        try:
            parent_a = UserConfig(name="parent_a", telegram_id=1, guardian_of=["kid"])
            parent_b = UserConfig(name="parent_b", telegram_id=3, guardian_of=["kid"])
            kid_config = UserConfig(name="kid", discord_id=2, monitored=True)
            delivered = await send_guardian_alerts(
                store,
                workshop_id,
                [parent_a, parent_b, kid_config],
                kid_config,
                _FLAGGED,
                alert_id="run:test",
                occurred_at=_NOW,
            )
            assert delivered == 2
        finally:
            await store.close()

    async def test_monitored_user_with_no_guardian_returns_zero_without_crashing(self, tmp_path: Path) -> None:
        store, workshop_id = await _open_store(tmp_path / "kai.db", _kid_human())
        try:
            kid_config = UserConfig(name="kid", discord_id=2, monitored=True)
            delivered = await send_guardian_alerts(
                store,
                workshop_id,
                [kid_config],
                kid_config,
                _FLAGGED,
                alert_id="run:test",
                occurred_at=_NOW,
            )
            assert delivered == 0
        finally:
            await store.close()

    async def test_unflagged_classification_is_rejected(self, tmp_path: Path) -> None:
        store, workshop_id = await _open_store(tmp_path / "kai.db", _kid_human())
        try:
            kid_config = UserConfig(name="kid", discord_id=2, monitored=True)
            with pytest.raises(ValueError, match="flagged"):
                await send_guardian_alerts(
                    store,
                    workshop_id,
                    [kid_config],
                    kid_config,
                    SafetyClassification(flagged=False),
                    alert_id="run:test",
                    occurred_at=_NOW,
                )
        finally:
            await store.close()
