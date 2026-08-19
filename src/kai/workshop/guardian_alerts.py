"""Guardian alert delivery for Bjornheim AI Workshop.

A `GuardianAlert` is deliberately not a `GitHubNotification`
(github_notifications.py): that type's `__post_init__` hard-rejects any
non-negative (individual DM) chat ID by design - it exists for
agent-owned group notification channels, not a guardian's own direct
channel. This module targets the guardian's own already-provisioned
direct channel instead (the same channel `guardian_access.py` resolves
identities against), and reuses the outbox
(`delivery_outbox.WorkshopDeliveryOutbox`) directly rather than through
the GitHub-shaped recorder.

Delivery itself needs no new code: `WorkshopTelegramDeliveryWorker` and
`WorkshopDiscordDeliveryWorker` already claim any outbox row matching
`purpose=NOTIFICATION_PURPOSE, mode="text"` for their transport,
regardless of which recorder enqueued it (see delivery_outbox.py's
`claim_next` - it filters on purpose/mode/transport columns, not on
DTO type). Telegram's notification worker already runs in production;
Discord's needs to be started too (see discord_adapter.py) for an
alert enqueued against a Discord-only guardian to actually go out.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from kai.config import UserConfig
from kai.workshop.delivery_outbox import (
    NOTIFICATION_PURPOSE,
    DeliveryRequest,
    DeliveryRequestResult,
    WorkshopDeliveryOutbox,
)
from kai.workshop.domain import (
    ChannelBindingId,
    EventEnvelope,
    EventId,
    MessageId,
    PrincipalId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.guardian_access import GuardianIdentity, GuardianTarget, resolve_guardians_of
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.safety_classifier import SafetyClassification
from kai.workshop.store import IdempotencyConflictError, WorkshopEventStore

_ALERT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class GuardianAlertError(RuntimeError):
    """A guardian alert could not be recorded safely."""


class GuardianAlertChannelNotFoundError(GuardianAlertError):
    """The guardian's own direct channel has no agent participant to author from."""


@dataclass(frozen=True, slots=True)
class GuardianAlert:
    """One safety-flagging alert for one guardian, before channel resolution."""

    alert_id: str
    target_name: str
    category: str
    summary: str
    occurred_at: datetime

    def __post_init__(self) -> None:
        if not _ALERT_ID_PATTERN.fullmatch(self.alert_id):
            raise ValueError("alert_id must be a bounded identifier")
        if not self.target_name:
            raise ValueError("target_name must be non-empty")
        if not self.category:
            raise ValueError("category must be non-empty")
        if not self.summary or len(self.summary) > 2000:
            raise ValueError("summary must contain between 1 and 2000 characters")
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")

    def body(self) -> str:
        timestamp = self.occurred_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
        return (
            f"Guardian alert: {self.target_name}'s conversation was flagged "
            f"({self.category}) at {timestamp}.\n\n{self.summary}\n\n"
            f"Use /monitor {self.target_name} transcript to review the full conversation."
        )


@dataclass(frozen=True, slots=True)
class GuardianAlertRecord:
    message_id: MessageId
    delivery: DeliveryRequestResult


class WorkshopGuardianAlertRecorder:
    """Record one guardian alert into their own direct channel, under one serialized writer.

    Modeled on WorkshopGitHubNotificationRecorder's idempotency/enqueue
    shape (github_notifications.py), but resolving the target channel by
    (channel_id, transport) - already known from a GuardianIdentity - rather
    than by an external_channel_id string lookup restricted to group
    notification channels.
    """

    def __init__(self, store: WorkshopEventStore) -> None:
        self._store = store
        self._outbox = WorkshopDeliveryOutbox(store)
        self._lock = asyncio.Lock()

    async def record(
        self,
        alert: GuardianAlert,
        guardian_identity: GuardianIdentity,
    ) -> GuardianAlertRecord:
        if not isinstance(alert, GuardianAlert):
            raise ValueError("alert must be a GuardianAlert")
        async with self._lock:
            return await self._record_locked(alert, guardian_identity)

    async def _record_locked(
        self,
        alert: GuardianAlert,
        guardian_identity: GuardianIdentity,
    ) -> GuardianAlertRecord:
        async with self._store.connection.execute(
            "SELECT c.workshop_id, cb.id, a.principal_id "
            "FROM channels c "
            "JOIN channel_bindings cb ON cb.channel_id = c.id AND cb.transport = ? "
            "JOIN channel_agents ca ON ca.channel_id = c.id "
            "JOIN agents a ON a.id = ca.agent_id AND a.workshop_id = c.workshop_id "
            "JOIN principals p ON p.id = a.principal_id AND p.kind = 'agent' "
            "WHERE c.id = ?",
            (guardian_identity.transport, guardian_identity.channel_id),
        ) as cursor:
            rows = list(await cursor.fetchall())
        if not rows:
            raise GuardianAlertChannelNotFoundError(
                f"No agent participant bound to guardian channel "
                f"{guardian_identity.channel_id!r} on transport {guardian_identity.transport!r}"
            )
        workshop_id = WorkshopId(str(rows[0][0]))
        binding_id = ChannelBindingId(str(rows[0][1]))
        agent_principal_id = PrincipalId(str(rows[0][2]))
        channel_id = guardian_identity.channel_id

        stable_name = f"guardian-alert:v1:{binding_id}:{alert.alert_id}"
        message_id = MessageId.derived(workshop_id, stable_name)
        idempotency_key = f"workshop-guardian-alert:v1:{binding_id}:{alert.alert_id}"
        payload = {
            "channel_id": channel_id,
            "author_principal_id": agent_principal_id,
            "body": alert.body(),
        }
        metadata = {
            "source": "guardian_alert",
            "guardian_alert_id": alert.alert_id,
            "target_name": alert.target_name,
            "category": alert.category,
        }
        occurred_at = alert.occurred_at.astimezone(UTC)
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            existing = await self._store.event_by_idempotency_key(idempotency_key)
            message_inserted = existing is None
            if existing is None:
                await self._store.append_in_transaction(
                    EventEnvelope.create(
                        event_id=EventId.derived(workshop_id, f"{stable_name}:event"),
                        event_type=WorkshopEventType.MESSAGE_CREATED,
                        event_version=1,
                        workshop_id=workshop_id,
                        aggregate_type="message",
                        aggregate_id=message_id,
                        actor_principal_id=agent_principal_id,
                        occurred_at=occurred_at,
                        idempotency_key=idempotency_key,
                        payload=payload,
                        metadata=metadata,
                    )
                )
            elif (
                existing.envelope.event_type != WorkshopEventType.MESSAGE_CREATED
                or existing.envelope.aggregate_id != message_id
                or existing.envelope.actor_principal_id != agent_principal_id
                or existing.envelope.payload != payload
                or existing.envelope.metadata != metadata
            ):
                raise IdempotencyConflictError(f"Event identity {idempotency_key!r} was reused with different content")

            projection = CanonicalConversationProjection()
            await self._store.project_pending_in_transaction(projection)
            delivery = await self._outbox.request_delivery_in_transaction(
                DeliveryRequest(
                    message_id=message_id,
                    channel_binding_id=binding_id,
                    mode="text",
                    purpose=NOTIFICATION_PURPOSE,
                    occurred_at=occurred_at,
                    max_attempts=5,
                )
            )
            if message_inserted != delivery.inserted:
                raise GuardianAlertError("Guardian alert message and delivery do not share one prior state")
            await self._store.project_pending_in_transaction(projection)
            await connection.commit()
            return GuardianAlertRecord(message_id=message_id, delivery=delivery)
        except Exception:
            await connection.rollback()
            raise


def _pick_delivery_identity(target: GuardianTarget) -> GuardianIdentity | None:
    """Pick one identity to deliver to for one guardian - never one alert per transport.

    Preference order matches GuardianIdentity's own construction order in
    guardian_access.py (telegram, then discord). A guardian configured on
    both transports gets exactly one alert, not a duplicate per transport;
    which transport wins is an implementation choice, not a guarantee a
    caller should depend on.
    """
    return target.guardian_identities[0] if target.guardian_identities else None


async def send_guardian_alerts(
    store: WorkshopEventStore,
    workshop_id: WorkshopId,
    user_configs: Iterable[UserConfig],
    target_config: UserConfig,
    classification: SafetyClassification,
    *,
    alert_id: str,
    occurred_at: datetime,
) -> int:
    """Fan out one flagged classification to every configured guardian.

    Returns the number of guardians an alert was recorded for. A monitored
    user with no configured guardian is a normal state (guardian_access.py's
    resolve_guardians_of already documents this): this function returns 0,
    does not raise, and does not log a warning for that specific case since
    it is expected, not an error.
    """
    if not classification.flagged or classification.category is None or classification.summary is None:
        raise ValueError("send_guardian_alerts requires a flagged classification")
    guardians = resolve_guardians_of(workshop_id, user_configs, target_config)
    if not guardians:
        return 0
    recorder = WorkshopGuardianAlertRecorder(store)
    delivered = 0
    for index, guardian in enumerate(guardians):
        identity = _pick_delivery_identity(guardian)
        if identity is None:
            continue
        alert = GuardianAlert(
            alert_id=f"{alert_id}:{index}",
            target_name=target_config.name,
            category=classification.category.value,
            summary=classification.summary,
            occurred_at=occurred_at,
        )
        await recorder.record(alert, identity)
        delivered += 1
    return delivered
