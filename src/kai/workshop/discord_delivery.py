"""Discord adapters and workers for durable Workshop delivery claims.

Mirrors telegram_delivery.py's shape (adapter/worker split, error
classification, fragment chunking against a platform text limit) with
Discord-specific substitutions: a flat DiscordTextBot Protocol wraps
discord.py's object-model API (User/DMChannel/Message) so the rest of this
module can stay structurally identical to its Telegram counterpart, Discord's
2000-character message limit replaces Telegram's 4096, and there is no
Markdown-parse-mode retry dance (Discord doesn't reject messages on Markdown
syntax the way Telegram's ParseMode.MARKDOWN can).
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from typing import Protocol

from kai.telegram_utils import chunk_text
from kai.workshop.delivery_fragments import (
    EDIT_OPERATION,
    SEND_OPERATION,
    DeliveryFragment,
    WorkshopDeliveryFragments,
)
from kai.workshop.delivery_outbox import (
    CONVERSATION_REPLY_PURPOSE,
    SEND_FRAGMENTS_CONTRACT,
    STREAMING_FINALIZATION_CONTRACT,
    DeliveryClaim,
    DeliveryPurpose,
    DeliveryRecoveryResult,
    DeliveryState,
    WorkshopDeliveryOutbox,
)
from kai.workshop.domain import DeliveryAuthorityEpochId, DeliveryId

# Discord DM-only for Phase 1: no group/guild channel routing, no username
# targets. external_channel_id is always the recipient's Discord user
# snowflake ID as a decimal string. Snowflakes are 64-bit unsigned, well
# within Python int range; the regex just rejects non-numeric/negative junk
# before it reaches discord.py.
_DISCORD_SNOWFLAKE_PATTERN = re.compile(r"^[1-9][0-9]{0,19}$")
_DISCORD_TEXT_LIMIT = 2000
_MAX_DISCORD_FRAGMENTS = 1000
_MAX_DISCORD_TEXT_SIZE = _DISCORD_TEXT_LIMIT * _MAX_DISCORD_FRAGMENTS
WORKSHOP_CLIENT_TEXT_MODE = "workshop_client_text"


class DiscordSentMessage(Protocol):
    """The successful send/edit response evidence retained by the outbox."""

    id: int


class DiscordTextBot(Protocol):
    """The flat Bot API surface needed by the durable text adapters.

    Deliberately shaped like TelegramTextBot (chat_id/message_id keyword
    args returning an object with an integer message id) rather than
    discord.py's native User/DMChannel/Message object model, so the
    adapters/workers below can mirror telegram_delivery.py structurally.
    A DiscordClientTextBot in discord_adapter.py implements this by wrapping
    a real discord.Client.
    """

    async def send_message(self, *, chat_id: int, text: str) -> DiscordSentMessage: ...

    async def edit_message_text(self, *, chat_id: int, message_id: int, text: str) -> DiscordSentMessage: ...


class DiscordDeliveryFailure(RuntimeError):
    """A sanitized, policy-bearing Discord failure safe for outbox storage."""

    def __init__(
        self,
        *,
        retryable: bool,
        error_code: str,
        minimum_retry_delay: timedelta | None = None,
        ambiguous: bool = False,
    ) -> None:
        super().__init__(error_code)
        self.retryable = retryable
        self.error_code = error_code
        self.minimum_retry_delay = minimum_retry_delay
        self.ambiguous = ambiguous


class DiscordDeliveryContractError(DiscordDeliveryFailure):
    """The durable claim cannot be represented by this Discord adapter."""

    def __init__(self, error_code: str) -> None:
        super().__init__(retryable=False, error_code=error_code)


class DiscordWorkOutcome(StrEnum):
    IDLE = "idle"
    SUCCEEDED = "succeeded"
    RETRY_SCHEDULED = "retry_scheduled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class DiscordWorkResult:
    outcome: DiscordWorkOutcome
    delivery_id: DeliveryId | None = None
    attempt_number: int | None = None
    error_code: str | None = None


def _discord_target(external_channel_id: str) -> int:
    if _DISCORD_SNOWFLAKE_PATTERN.fullmatch(external_channel_id):
        value = int(external_channel_id)
        if value <= 2**63 - 1:
            return value
    raise DiscordDeliveryContractError("discord_target_invalid")


def _classify_discord_error(error: Exception) -> DiscordDeliveryFailure:
    """Classify a discord.py exception without importing discord at module scope.

    Imported lazily so this module (and its unit tests) don't require the
    discord.py package to be installed just to exercise error-classification
    logic in isolation; discord_adapter.py's real client wiring imports
    discord.py directly.
    """
    import discord

    if isinstance(error, discord.Forbidden):
        return DiscordDeliveryFailure(retryable=False, error_code="discord_forbidden")
    if isinstance(error, discord.NotFound):
        return DiscordDeliveryFailure(retryable=False, error_code="discord_not_found")
    if isinstance(error, discord.RateLimited):
        minimum_retry_delay = max(timedelta(seconds=error.retry_after), timedelta(seconds=1))
        if minimum_retry_delay > timedelta(days=1):
            return DiscordDeliveryFailure(retryable=False, error_code="discord_rate_limit_too_long")
        return DiscordDeliveryFailure(
            retryable=True,
            error_code="discord_rate_limited",
            minimum_retry_delay=minimum_retry_delay,
        )
    if isinstance(error, discord.DiscordServerError):
        return DiscordDeliveryFailure(retryable=False, error_code="discord_server_error_uncertain", ambiguous=True)
    if isinstance(error, discord.HTTPException):
        return DiscordDeliveryFailure(retryable=False, error_code="discord_bad_request")
    return DiscordDeliveryFailure(retryable=False, error_code="discord_error_uncertain", ambiguous=True)


def _discord_fragments(body: str) -> tuple[str, ...]:
    if not body or len(body) > _MAX_DISCORD_TEXT_SIZE:
        raise DiscordDeliveryContractError("discord_text_size_unsupported")
    fragments = tuple(chunk_text(body, max_len=_DISCORD_TEXT_LIMIT))
    if not fragments or len(fragments) > _MAX_DISCORD_FRAGMENTS:
        raise DiscordDeliveryContractError("discord_text_size_unsupported")
    return fragments


def _discord_delivery_body(claim: DeliveryClaim) -> str:
    if claim.mode == "text":
        return claim.body
    if claim.mode == WORKSHOP_CLIENT_TEXT_MODE:
        display_name = claim.author_display_name.strip()
        if not display_name or len(display_name) > 200:
            raise DiscordDeliveryContractError("discord_author_invalid")
        return f"{display_name} via Workshop:\n{claim.body}"
    raise DiscordDeliveryContractError("discord_mode_unsupported")


class WorkshopDiscordDeliveryAdapter:
    """Deliver one durably selected Discord text fragment."""

    def __init__(self, bot: DiscordTextBot) -> None:
        self._bot = bot

    async def deliver_fragment(self, claim: DeliveryClaim, fragment: DeliveryFragment) -> int:
        if not isinstance(claim, DeliveryClaim):
            raise ValueError("claim must be a DeliveryClaim")
        if not isinstance(fragment, DeliveryFragment) or fragment.delivery_id != claim.delivery_id:
            raise ValueError("fragment must belong to the claimed delivery")
        if claim.transport != "discord":
            raise DiscordDeliveryContractError("discord_transport_mismatch")
        if claim.mode not in {"text", WORKSHOP_CLIENT_TEXT_MODE}:
            raise DiscordDeliveryContractError("discord_mode_unsupported")
        if fragment.operation != "send" or fragment.target_external_message_id is not None:
            raise DiscordDeliveryContractError("discord_operation_unsupported")
        if not fragment.body or len(fragment.body) > _DISCORD_TEXT_LIMIT:
            raise DiscordDeliveryContractError("discord_text_size_unsupported")

        target = _discord_target(claim.external_channel_id)
        try:
            sent = await self._bot.send_message(chat_id=target, text=fragment.body)
        except Exception as error:
            raise _classify_discord_error(error) from error
        message_id = getattr(sent, "id", None)
        if not isinstance(message_id, int) or isinstance(message_id, bool) or message_id <= 0:
            raise DiscordDeliveryFailure(retryable=False, error_code="discord_response_invalid", ambiguous=True)
        return message_id


class WorkshopDiscordStreamingFinalizationAdapter:
    """Execute one immutable edit-or-send finalization operation."""

    def __init__(self, bot: DiscordTextBot) -> None:
        self._bot = bot
        self._send_adapter = WorkshopDiscordDeliveryAdapter(bot)

    async def deliver_fragment(self, claim: DeliveryClaim, fragment: DeliveryFragment) -> int:
        if not isinstance(claim, DeliveryClaim):
            raise ValueError("claim must be a DeliveryClaim")
        if not isinstance(fragment, DeliveryFragment) or fragment.delivery_id != claim.delivery_id:
            raise ValueError("fragment must belong to the claimed delivery")
        if claim.execution_contract != STREAMING_FINALIZATION_CONTRACT:
            raise DiscordDeliveryContractError("discord_execution_contract_mismatch")
        if claim.transport != "discord":
            raise DiscordDeliveryContractError("discord_transport_mismatch")
        if claim.mode != "text":
            raise DiscordDeliveryContractError("discord_mode_unsupported")
        if not fragment.body or len(fragment.body) > _DISCORD_TEXT_LIMIT:
            raise DiscordDeliveryContractError("discord_text_size_unsupported")

        if fragment.operation == SEND_OPERATION and fragment.target_external_message_id is None:
            return await self._send_adapter.deliver_fragment(claim, fragment)
        if fragment.operation != EDIT_OPERATION or fragment.target_external_message_id is None:
            raise DiscordDeliveryContractError("discord_operation_unsupported")

        target_message_id = fragment.target_external_message_id
        if not isinstance(target_message_id, int) or isinstance(target_message_id, bool) or target_message_id <= 0:
            raise DiscordDeliveryContractError("discord_edit_target_invalid")
        target = _discord_target(claim.external_channel_id)
        try:
            edited = await self._bot.edit_message_text(
                chat_id=target,
                message_id=target_message_id,
                text=fragment.body,
            )
        except Exception as error:
            raise _classify_discord_error(error) from error

        message_id = getattr(edited, "id", None)
        if not isinstance(message_id, int) or isinstance(message_id, bool) or message_id != target_message_id:
            raise DiscordDeliveryFailure(retryable=False, error_code="discord_edit_response_invalid", ambiguous=True)
        return message_id


class WorkshopDiscordDeliveryWorker:
    """Claim and settle one explicitly assigned lane of Discord text work."""

    def __init__(
        self,
        outbox: WorkshopDeliveryOutbox,
        fragments: WorkshopDeliveryFragments,
        adapter: WorkshopDiscordDeliveryAdapter,
        *,
        worker_id: str,
        purpose: DeliveryPurpose,
        modes: tuple[str, ...] = ("text",),
        lease_duration: timedelta = timedelta(seconds=30),
        poll_interval: float = 1.0,
    ) -> None:
        if poll_interval <= 0 or poll_interval > 60:
            raise ValueError("poll_interval must be positive and at most 60 seconds")
        self._outbox = outbox
        self._fragments = fragments
        self._adapter = adapter
        self._worker_id = worker_id
        self._purpose: DeliveryPurpose = purpose
        if not modes or len(set(modes)) != len(modes):
            raise ValueError("modes must contain unique values")
        self._modes = modes
        self._lease_duration = lease_duration
        self._poll_interval = poll_interval

    async def run_once(self) -> DiscordWorkResult:
        claim = await self._outbox.claim_next(
            self._worker_id,
            purposes=(self._purpose,),
            execution_contracts=(SEND_FRAGMENTS_CONTRACT,),
            lease_duration=self._lease_duration,
            transport="discord",
            modes=self._modes,
        )
        return await self._run_claim(claim)

    async def run_delivery(self, delivery_id: DeliveryId) -> DiscordWorkResult:
        """Run only one explicitly selected delivery without draining the outbox."""
        if not isinstance(delivery_id, DeliveryId):
            raise ValueError("delivery_id must be a DeliveryId")
        claim = await self._outbox.claim_next(
            self._worker_id,
            purposes=(self._purpose,),
            execution_contracts=(SEND_FRAGMENTS_CONTRACT,),
            lease_duration=self._lease_duration,
            transport="discord",
            modes=self._modes,
            delivery_id=delivery_id,
        )
        return await self._run_claim(claim)

    async def recover_expired_leases(self) -> DeliveryRecoveryResult:
        """Recover only work in this worker's explicitly assigned purpose lanes."""
        return await self._outbox.recover_expired_leases(
            purposes=(self._purpose,),
            execution_contracts=(SEND_FRAGMENTS_CONTRACT,),
        )

    async def _run_claim(self, claim: DeliveryClaim | None) -> DiscordWorkResult:
        if claim is None:
            return DiscordWorkResult(outcome=DiscordWorkOutcome.IDLE)

        try:
            bodies = _discord_fragments(_discord_delivery_body(claim))
            await self._fragments.prepare(claim, bodies)
        except DiscordDeliveryFailure as failure:
            return await self._settle_failure(claim, None, failure)

        while True:
            fragment = await self._fragments.begin_next(claim)
            if fragment is None:
                state = await self._outbox.mark_succeeded(claim)
                return self._success_result(claim, state)
            try:
                external_message_id = await self._adapter.deliver_fragment(claim, fragment)
            except DiscordDeliveryFailure as failure:
                return await self._settle_failure(claim, fragment, failure)
            await self._fragments.mark_sent(claim, fragment, external_message_id=external_message_id)

    async def run(self, stop_event: asyncio.Event) -> None:
        if not isinstance(stop_event, asyncio.Event):
            raise ValueError("stop_event must be an asyncio.Event")
        while not stop_event.is_set():
            result = await self.run_once()
            if result.outcome != DiscordWorkOutcome.IDLE:
                continue
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self._poll_interval)
            except TimeoutError:
                pass

    @staticmethod
    def _success_result(claim: DeliveryClaim, state: DeliveryState) -> DiscordWorkResult:
        if state.status != "succeeded":
            raise RuntimeError("Successful Discord delivery did not reach succeeded outbox state")
        return DiscordWorkResult(
            outcome=DiscordWorkOutcome.SUCCEEDED,
            delivery_id=claim.delivery_id,
            attempt_number=claim.attempt_number,
        )

    async def _settle_failure(
        self,
        claim: DeliveryClaim,
        fragment: DeliveryFragment | None,
        failure: DiscordDeliveryFailure,
    ) -> DiscordWorkResult:
        if fragment is not None:
            if failure.ambiguous:
                await self._fragments.mark_uncertain(claim, fragment)
            else:
                await self._fragments.release_after_definitive_failure(claim, fragment)
        state = await self._outbox.mark_failed(
            claim,
            retryable=failure.retryable and not failure.ambiguous,
            error_code=failure.error_code,
            minimum_retry_delay=failure.minimum_retry_delay,
        )
        return DiscordWorkResult(
            outcome=(DiscordWorkOutcome.RETRY_SCHEDULED if state.status == "retry_wait" else DiscordWorkOutcome.FAILED),
            delivery_id=claim.delivery_id,
            attempt_number=claim.attempt_number,
            error_code=failure.error_code,
        )


class WorkshopDiscordStreamingFinalizationWorker:
    """Execute only durable conversation streaming-finalization plans."""

    def __init__(
        self,
        outbox: WorkshopDeliveryOutbox,
        fragments: WorkshopDeliveryFragments,
        adapter: WorkshopDiscordStreamingFinalizationAdapter,
        *,
        worker_id: str,
        authority_epoch_id: DeliveryAuthorityEpochId,
        lease_duration: timedelta = timedelta(seconds=30),
        poll_interval: float = 1.0,
    ) -> None:
        if poll_interval <= 0 or poll_interval > 60:
            raise ValueError("poll_interval must be positive and at most 60 seconds")
        if not isinstance(authority_epoch_id, DeliveryAuthorityEpochId):
            raise ValueError("authority_epoch_id must be a DeliveryAuthorityEpochId")
        self._outbox = outbox
        self._fragments = fragments
        self._adapter = adapter
        self._worker_id = worker_id
        self._authority_epoch_id = authority_epoch_id
        self._lease_duration = lease_duration
        self._poll_interval = poll_interval

    async def run_once(self) -> DiscordWorkResult:
        claim = await self._outbox.claim_next(
            self._worker_id,
            purposes=(CONVERSATION_REPLY_PURPOSE,),
            execution_contracts=(STREAMING_FINALIZATION_CONTRACT,),
            lease_duration=self._lease_duration,
            transport="discord",
            modes=("text",),
            authority_epoch_id=self._authority_epoch_id,
        )
        return await self._run_claim(claim)

    async def run_delivery(self, delivery_id: DeliveryId) -> DiscordWorkResult:
        """Run one exact finalization delivery without draining other work."""
        if not isinstance(delivery_id, DeliveryId):
            raise ValueError("delivery_id must be a DeliveryId")
        claim = await self._outbox.claim_next(
            self._worker_id,
            purposes=(CONVERSATION_REPLY_PURPOSE,),
            execution_contracts=(STREAMING_FINALIZATION_CONTRACT,),
            lease_duration=self._lease_duration,
            transport="discord",
            modes=("text",),
            delivery_id=delivery_id,
            authority_epoch_id=self._authority_epoch_id,
        )
        return await self._run_claim(claim)

    async def recover_expired_leases(self) -> DeliveryRecoveryResult:
        return await self._outbox.recover_expired_leases(
            purposes=(CONVERSATION_REPLY_PURPOSE,),
            execution_contracts=(STREAMING_FINALIZATION_CONTRACT,),
            authority_epoch_id=self._authority_epoch_id,
        )

    async def _run_claim(self, claim: DeliveryClaim | None) -> DiscordWorkResult:
        if claim is None:
            return DiscordWorkResult(outcome=DiscordWorkOutcome.IDLE)
        if claim.execution_contract != STREAMING_FINALIZATION_CONTRACT:
            raise RuntimeError("Finalization worker claimed another execution contract")

        while True:
            fragment = await self._fragments.begin_next(claim)
            if fragment is None:
                state = await self._outbox.mark_succeeded(claim)
                return self._success_result(claim, state)
            try:
                external_message_id = await self._adapter.deliver_fragment(claim, fragment)
            except DiscordDeliveryFailure as failure:
                return await self._settle_failure(claim, fragment, failure)
            await self._fragments.mark_sent(claim, fragment, external_message_id=external_message_id)

    async def run(self, stop_event: asyncio.Event) -> None:
        if not isinstance(stop_event, asyncio.Event):
            raise ValueError("stop_event must be an asyncio.Event")
        while not stop_event.is_set():
            result = await self.run_once()
            if result.outcome != DiscordWorkOutcome.IDLE:
                continue
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self._poll_interval)
            except TimeoutError:
                pass

    @staticmethod
    def _success_result(claim: DeliveryClaim, state: DeliveryState) -> DiscordWorkResult:
        if state.status != "succeeded":
            raise RuntimeError("Successful Discord finalization did not reach succeeded outbox state")
        return DiscordWorkResult(
            outcome=DiscordWorkOutcome.SUCCEEDED,
            delivery_id=claim.delivery_id,
            attempt_number=claim.attempt_number,
        )

    async def _settle_failure(
        self,
        claim: DeliveryClaim,
        fragment: DeliveryFragment,
        failure: DiscordDeliveryFailure,
    ) -> DiscordWorkResult:
        if failure.ambiguous:
            await self._fragments.mark_uncertain(claim, fragment)
        else:
            await self._fragments.release_after_definitive_failure(claim, fragment)
        state = await self._outbox.mark_failed(
            claim,
            retryable=failure.retryable and not failure.ambiguous,
            error_code=failure.error_code,
            minimum_retry_delay=failure.minimum_retry_delay,
        )
        return DiscordWorkResult(
            outcome=(DiscordWorkOutcome.RETRY_SCHEDULED if state.status == "retry_wait" else DiscordWorkOutcome.FAILED),
            delivery_id=claim.delivery_id,
            attempt_number=claim.attempt_number,
            error_code=failure.error_code,
        )
