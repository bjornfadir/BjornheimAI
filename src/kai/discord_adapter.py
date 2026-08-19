"""Explicit lifecycle boundary for Bjornheim AI's Discord adapter.

Mirrors telegram_adapter.py's shape (state machine, readiness reporting,
owned delivery service). Discord has no webhook/polling distinction the way
Telegram does - the client holds one persistent gateway connection - so
there is no separate activate_ingress() step; start() does everything.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import StrEnum

import discord

from kai.application_host import KaiCoreServices
from kai.config import Config
from kai.discord_bot import create_discord_client
from kai.workshop.discord_delivery_runtime import (
    WorkshopDiscordConversationDeliveryService,
    WorkshopDiscordNotificationService,
)

log = logging.getLogger(__name__)


class DiscordAdapterState(StrEnum):
    """Observable lifecycle state for the Discord adapter."""

    NEW = "new"
    STARTING = "starting"
    READY = "ready"
    DRAINING = "draining"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class DiscordAdapterReadiness:
    """Non-sensitive readiness for Discord-owned components."""

    state: DiscordAdapterState
    client_ready: bool
    conversation_delivery: bool
    notification_delivery: bool

    @property
    def ready(self) -> bool:
        return (
            self.state == DiscordAdapterState.READY
            and self.client_ready
            and self.conversation_delivery
            and self.notification_delivery
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.state.value,
            "ready": self.ready,
            "components": {
                "client_ready": self.client_ready,
                "conversation_delivery": self.conversation_delivery,
                "notification_delivery": self.notification_delivery,
            },
        }


class DiscordClientTextBot:
    """Adapt discord.py's object model to the flat DiscordTextBot Protocol.

    discord_delivery.py's adapters/workers are written against a flat
    send_message(chat_id=...)/edit_message_text(chat_id=..., message_id=...)
    shape (mirroring python-telegram-bot's Bot API) rather than discord.py's
    User/DMChannel/Message object model. This class is the one place that
    translation happens, isolating every discord.py-specific object-model
    call behind a narrow boundary.
    """

    def __init__(self, client: discord.Client) -> None:
        self._client = client

    async def _resolve_user(self, chat_id: int) -> discord.User:
        user = self._client.get_user(chat_id)
        if user is None:
            user = await self._client.fetch_user(chat_id)
        return user

    async def send_message(self, *, chat_id: int, text: str) -> discord.Message:
        user = await self._resolve_user(chat_id)
        return await user.send(content=text)

    async def edit_message_text(self, *, chat_id: int, message_id: int, text: str) -> discord.Message:
        user = await self._resolve_user(chat_id)
        channel = user.dm_channel or await user.create_dm()
        message = await channel.fetch_message(message_id)
        return await message.edit(content=text)


class DiscordAdapter:
    """Own the Discord client connection and its conversation delivery worker."""

    def __init__(self, config: Config, core_services: KaiCoreServices) -> None:
        self._config = config
        self._core_services = core_services
        self._state = DiscordAdapterState.NEW
        self._client: discord.Client | None = None
        self._client_task: asyncio.Task[None] | None = None
        self._conversation_delivery: WorkshopDiscordConversationDeliveryService | None = None
        # Started purely for its worker draining purpose=notification,
        # transport=discord outbox rows - never call .record() on this,
        # that method is hard-typed to GitHubNotification. Guardian alerts
        # (guardian_alerts.py) enqueue directly against the outbox with a
        # generic GuardianAlert, and this worker delivers whatever it
        # finds there regardless of which recorder wrote it (see
        # delivery_outbox.py's claim_next: it filters on purpose/mode/
        # transport columns, not on DTO type).
        self._notification_delivery: WorkshopDiscordNotificationService | None = None

    @property
    def client(self) -> discord.Client:
        client = self._client
        if client is None:
            raise RuntimeError("Discord client is not started")
        return client

    @property
    def readiness(self) -> DiscordAdapterReadiness:
        return DiscordAdapterReadiness(
            state=self._state,
            client_ready=self._client is not None and self._client.is_ready(),
            conversation_delivery=(self._conversation_delivery is not None and self._conversation_delivery.ready),
            notification_delivery=(self._notification_delivery is not None and self._notification_delivery.ready),
        )

    async def start(self) -> None:
        if self._state != DiscordAdapterState.NEW:
            raise RuntimeError(f"Discord adapter cannot start from {self._state.value}")
        self._state = DiscordAdapterState.STARTING
        try:
            token = self._config.discord_bot_token
            if not token:
                raise RuntimeError("Discord adapter started without DISCORD_BOT_TOKEN configured")

            client = create_discord_client(self._config, self._core_services)
            self._client = client

            client_task = asyncio.create_task(
                client.start(token, reconnect=True),
                name="kai-discord-client",
            )
            self._client_task = client_task
            await self._await_ready_or_fail(client, client_task)

            self._conversation_delivery = await WorkshopDiscordConversationDeliveryService.open_and_start(
                self._config.session_db_path,
                DiscordClientTextBot(client),
                authority_epoch_id=self._core_services.delivery_authority_epoch.epoch_id,
            )
            log.info("Workshop Discord conversation delivery is ready")
            self._notification_delivery = await WorkshopDiscordNotificationService.open_and_start(
                self._config.session_db_path,
                DiscordClientTextBot(client),
            )
            log.info("Workshop Discord notification delivery is ready")
            self._state = DiscordAdapterState.READY
        except BaseException:
            self._state = DiscordAdapterState.FAILED
            await self._stop_components()
            raise

    _READY_TIMEOUT_SECONDS = 60

    async def _await_ready_or_fail(self, client: discord.Client, client_task: asyncio.Task[None]) -> None:
        """Wait for the gateway handshake without hanging forever.

        client.wait_until_ready() alone can block indefinitely: a login
        failure (bad token, disallowed privileged intents) or an
        unreachable gateway raises *inside* client_task, not through
        wait_until_ready() - nothing was watching that task, so a failed
        connection looked identical to a slow one, forever. Race both: if
        client_task finishes first, it failed before ever becoming ready,
        so surface its real exception instead of hanging.
        """
        ready_task = asyncio.create_task(client.wait_until_ready(), name="kai-discord-ready")
        try:
            done, _pending = await asyncio.wait(
                {ready_task, client_task},
                timeout=self._READY_TIMEOUT_SECONDS,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            if not ready_task.done():
                ready_task.cancel()

        if ready_task in done:
            return
        if client_task in done:
            exc = client_task.exception()
            if exc is not None:
                raise RuntimeError("Discord client failed before becoming ready") from exc
            raise RuntimeError("Discord client task exited before the gateway became ready")
        raise TimeoutError(
            f"Discord gateway did not become ready within {self._READY_TIMEOUT_SECONDS}s - "
            "check DISCORD_BOT_TOKEN, the Message Content privileged intent, and network "
            "connectivity to Discord"
        )

    async def wait(self) -> None:
        if self._state != DiscordAdapterState.READY:
            raise RuntimeError(f"Discord adapter cannot be supervised while {self._state.value}")
        conversation_delivery = self._conversation_delivery
        notification_delivery = self._notification_delivery
        client_task = self._client_task
        if conversation_delivery is None or notification_delivery is None or client_task is None:
            raise RuntimeError("Discord delivery services are unavailable")
        await asyncio.gather(
            asyncio.shield(client_task),
            conversation_delivery.wait(),
            notification_delivery.wait(),
        )

    async def stop(self) -> None:
        if self._state in {DiscordAdapterState.NEW, DiscordAdapterState.STOPPED}:
            self._state = DiscordAdapterState.STOPPED
            return
        self._state = DiscordAdapterState.DRAINING
        errors = await self._stop_components()
        self._state = DiscordAdapterState.STOPPED
        if errors:
            raise ExceptionGroup("Discord adapter shutdown failed", errors)

    async def _stop_components(self) -> list[Exception]:
        errors: list[Exception] = []
        if self._notification_delivery is not None:
            try:
                await self._notification_delivery.stop()
            except Exception as exc:
                errors.append(exc)
        self._notification_delivery = None
        if self._conversation_delivery is not None:
            try:
                await self._conversation_delivery.stop()
            except Exception as exc:
                errors.append(exc)
        self._conversation_delivery = None

        client = self._client
        if client is not None:
            try:
                await client.close()
            except Exception as exc:
                errors.append(exc)
        client_task = self._client_task
        if client_task is not None:
            try:
                await asyncio.wait_for(client_task, timeout=10)
            except (TimeoutError, asyncio.CancelledError):
                client_task.cancel()
            except Exception as exc:
                errors.append(exc)
        self._client_task = None
        self._client = None
        return errors
