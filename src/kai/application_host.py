"""Transport-neutral construction and lifecycle for Bjornheim AI core services."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from kai import sessions
from kai.config import Config, UserConfig
from kai.pool import SubprocessPool
from kai.workshop.bootstrap import bootstrap_human_principal_id
from kai.workshop.client_commands import WorkshopClientCommandExecutor
from kai.workshop.compatibility_state import WorkshopCompatibilityStateWriter
from kai.workshop.conversation_runs import WorkshopConversationRunService
from kai.workshop.delivery_authority import (
    DeliveryAuthorityEpoch,
    WorkshopConversationDeliveryAuthority,
)
from kai.workshop.domain import PrincipalId, RunId, WorkshopId
from kai.workshop.execution_coordinator import CanonicalExecutionDisposition, CanonicalExecutionResult
from kai.workshop.guardian_alerts import send_guardian_alerts
from kai.workshop.private_text_execution import WorkshopPrivateTextExecutionService
from kai.workshop.runtime_pool import WorkshopRuntimePool
from kai.workshop.runtime_profiles import WorkshopRuntimeProfileRegistry
from kai.workshop.safety_classifier import classify as classify_safety
from kai.workshop.storage_namespaces import WorkshopPrincipalStorageRegistry
from kai.workshop.store import WorkshopEventStore


class KaiApplicationState(StrEnum):
    """Observable lifecycle state for the transport-neutral application host."""

    NEW = "new"
    STARTING = "starting"
    READY = "ready"
    DRAINING = "draining"
    STOPPED = "stopped"
    FAILED = "failed"


class KaiAdapterReadiness(Protocol):
    """JSON-safe readiness contract for one external adapter."""

    def as_dict(self) -> dict[str, object]: ...


class KaiApplicationAdapter(Protocol):
    """Lifecycle contract for one configured external adapter."""

    @property
    def readiness(self) -> KaiAdapterReadiness: ...

    async def start(self) -> None: ...

    async def wait(self) -> None: ...

    async def stop(self) -> None: ...


@dataclass(frozen=True, slots=True)
class KaiCoreReadiness:
    """Non-sensitive component readiness exposed to operators and adapters."""

    state: KaiApplicationState
    runtime: bool
    executor: bool
    client_api: bool
    store: bool

    @property
    def ready(self) -> bool:
        return self.state == KaiApplicationState.READY and all(
            (self.runtime, self.executor, self.client_api, self.store)
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.state.value,
            "ready": self.ready,
            "components": {
                "runtime": self.runtime,
                "executor": self.executor,
                "client_api": self.client_api,
                "store": self.store,
            },
        }


@dataclass(frozen=True, slots=True)
class KaiCoreServices:
    """Typed dependencies shared by client and transport adapters."""

    subprocess_pool: SubprocessPool
    runtime_profiles: WorkshopRuntimeProfileRegistry
    runtime_pool: WorkshopRuntimePool
    conversation_runs: WorkshopConversationRunService
    private_text_execution: WorkshopPrivateTextExecutionService
    client_commands: WorkshopClientCommandExecutor
    client_store: WorkshopEventStore
    principal_storage: WorkshopPrincipalStorageRegistry
    delivery_authority_epoch: DeliveryAuthorityEpoch
    workshop_id: WorkshopId


class KaiApplicationHost:
    """Own core service construction, supervision, readiness, and shutdown.

    This module deliberately imports no Telegram package and accepts no
    Telegram object. Adapters receive :attr:`services` after startup and cannot
    become the owner of runtime or Workshop execution lifecycle.
    """

    def __init__(
        self,
        *,
        config: Config,
        runtime_profiles: WorkshopRuntimeProfileRegistry,
        principal_storage: WorkshopPrincipalStorageRegistry,
        services_info: list[dict],
        registered_backend_ids: frozenset[str],
        workshop_id: WorkshopId,
    ) -> None:
        self._config = config
        self._runtime_profiles = runtime_profiles
        self._principal_storage = principal_storage
        self._services_info = services_info
        self._registered_backend_ids = registered_backend_ids
        self._workshop_id = workshop_id
        self._state = KaiApplicationState.NEW
        self._services: KaiCoreServices | None = None
        self._adapters: dict[str, KaiApplicationAdapter] = {}

    @property
    def services(self) -> KaiCoreServices:
        services = self._services
        if services is None:
            raise RuntimeError("Bjornheim AI core services are not started")
        return services

    @property
    def readiness(self) -> KaiCoreReadiness:
        services = self._services
        return KaiCoreReadiness(
            state=self._state,
            runtime=services is not None and self._state == KaiApplicationState.READY,
            executor=services is not None and services.private_text_execution.ready,
            client_api=services is not None and services.client_commands.ready,
            store=services is not None,
        )

    @property
    def adapter_readiness(self) -> dict[str, object]:
        """Return non-sensitive readiness reported by attached adapters."""
        return {name: adapter.readiness.as_dict() for name, adapter in self._adapters.items()}

    async def start(self) -> KaiCoreServices:
        if self._state != KaiApplicationState.NEW:
            raise RuntimeError(f"Bjornheim AI application host cannot start from {self._state.value}")
        self._state = KaiApplicationState.STARTING

        subprocess_pool: SubprocessPool | None = None
        private_execution: WorkshopPrivateTextExecutionService | None = None
        client_store: WorkshopEventStore | None = None
        client_commands: WorkshopClientCommandExecutor | None = None
        try:
            subprocess_pool = SubprocessPool(
                config=self._config,
                services_info=self._services_info,
                runtime_profiles=self._runtime_profiles,
            )
            runtime_pool = WorkshopRuntimePool(subprocess_pool, self._runtime_profiles)
            conversation_runs = WorkshopConversationRunService(
                runtime_pool,
                sessions.resolve_workshop_conversation_run,
            )

            subprocess_pool.start()
            # Terminal run recovery atomically creates delivery work stamped
            # with the active authority epoch. Resume or create that durable,
            # transport-neutral authority before either recovery owner starts.
            client_store = await WorkshopEventStore.open(Path(self._config.session_db_path))
            delivery_authority_epoch = (await WorkshopConversationDeliveryAuthority(client_store).activate()).epoch
            private_execution = await WorkshopPrivateTextExecutionService.open_and_start(
                Path(self._config.session_db_path),
                runtime_pool,
                registered_backend_ids=self._registered_backend_ids,
                on_completed=self._make_on_execution_completed(client_store),
            )
            client_commands = WorkshopClientCommandExecutor(
                private_execution,
                WorkshopCompatibilityStateWriter(self._config, runtime_pool),
            )
            await client_commands.start()

            self._services = KaiCoreServices(
                subprocess_pool=subprocess_pool,
                runtime_profiles=self._runtime_profiles,
                runtime_pool=runtime_pool,
                conversation_runs=conversation_runs,
                private_text_execution=private_execution,
                client_commands=client_commands,
                client_store=client_store,
                principal_storage=self._principal_storage,
                delivery_authority_epoch=delivery_authority_epoch,
                workshop_id=self._workshop_id,
            )
            self._state = KaiApplicationState.READY
            return self._services
        except BaseException:
            self._state = KaiApplicationState.FAILED
            if client_commands is not None:
                await client_commands.stop()
            if client_store is not None:
                await client_store.close()
            if private_execution is not None:
                await private_execution.stop()
            if subprocess_pool is not None:
                await subprocess_pool.shutdown()
            raise

    def _make_on_execution_completed(
        self,
        client_store: WorkshopEventStore,
    ) -> Callable[[RunId, CanonicalExecutionResult], Awaitable[None]]:
        """Build the safety-flagging hook for monitored users' completed turns.

        Bound to client_store (already open for the lifetime of this host)
        rather than opening a dedicated store: the hook's read (one message
        body) and write (an occasional guardian alert enqueue) are light and
        infrequent relative to client_store's existing traffic, and reusing
        it avoids one more store-lifecycle to manage. Runs fire-and-forget
        from WorkshopPrivateTextExecutionService.execute() - see that
        module's _fire_on_completed for the "never delay or fail the turn"
        contract this closure relies on.
        """

        async def _on_completed(run_id: RunId, result: CanonicalExecutionResult) -> None:
            if result.disposition != CanonicalExecutionDisposition.COMPLETED or result.terminal is None:
                return
            run = result.run
            target_config = self._resolve_monitored_user(run.workshop_id, run.requested_by_principal_id)
            if target_config is None or not target_config.monitored:
                return
            async with client_store.connection.execute(
                "SELECT body FROM messages WHERE id = ? AND channel_id = ? AND author_principal_id = ?",
                (run.inbound_message_id, run.channel_id, run.requested_by_principal_id),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                return
            user_text = str(row[0])
            assistant_text = result.terminal.body
            classification = await classify_safety(
                user_text,
                assistant_text,
                self._config,
                user_config=target_config,
            )
            if not classification.flagged:
                return
            await send_guardian_alerts(
                client_store,
                run.workshop_id,
                self._config.user_configs.values(),
                target_config,
                classification,
                alert_id=f"run:{run_id}",
                occurred_at=run.terminal_at or run.accepted_at,
            )

        return _on_completed

    def _resolve_monitored_user(self, workshop_id: WorkshopId, principal_id: PrincipalId) -> UserConfig | None:
        """Reverse-resolve which configured user a run's principal belongs to.

        Principal IDs are derived deterministically from (workshop_id,
        transport, external_subject) - see bootstrap.py's
        bootstrap_human_principal_id - not stored as a lookup table keyed
        the other direction, so this checks each configured user's own
        transport identities the same way guardian_access.py's
        _resolve_identities does, rather than adding a new reverse index
        for what is, at family scale, a handful of comparisons.
        """
        for user_config in self._config.user_configs.values():
            for transport, external_subject in (
                ("telegram", str(user_config.telegram_id) if user_config.telegram_id is not None else None),
                ("discord", str(user_config.discord_id) if user_config.discord_id is not None else None),
            ):
                if external_subject is None:
                    continue
                if bootstrap_human_principal_id(workshop_id, transport, external_subject) == principal_id:
                    return user_config
        return None

    async def attach_adapter(self, name: str, adapter: KaiApplicationAdapter) -> None:
        """Start and supervise an adapter after the core is ready."""
        if self._state != KaiApplicationState.READY:
            raise RuntimeError(f"Bjornheim AI adapter cannot start while core is {self._state.value}")
        if not name:
            raise RuntimeError("Bjornheim AI adapter name cannot be empty")
        if name in self._adapters:
            raise RuntimeError(f"Bjornheim AI adapter {name!r} is already attached")
        try:
            await adapter.start()
        except BaseException:
            self._state = KaiApplicationState.FAILED
            try:
                await adapter.stop()
            except Exception:
                pass
            raise
        self._adapters[name] = adapter

    async def wait(self) -> None:
        """Expose failure of a required supervised core worker."""
        await asyncio.gather(
            self.services.private_text_execution.wait(),
            *(adapter.wait() for adapter in self._adapters.values()),
        )

    async def stop(self) -> None:
        if self._state in {KaiApplicationState.NEW, KaiApplicationState.STOPPED}:
            self._state = KaiApplicationState.STOPPED
            return
        services = self._services
        self._state = KaiApplicationState.DRAINING
        if services is None:
            self._state = KaiApplicationState.STOPPED
            return

        errors: list[Exception] = []
        for adapter in reversed(tuple(self._adapters.values())):
            try:
                await adapter.stop()
            except Exception as exc:
                errors.append(exc)
        self._adapters.clear()
        for operation in (
            services.client_commands.stop,
            services.client_store.close,
            services.private_text_execution.stop,
            services.subprocess_pool.shutdown,
        ):
            try:
                await operation()
            except Exception as exc:
                errors.append(exc)
        self._services = None
        self._state = KaiApplicationState.STOPPED
        if errors:
            raise ExceptionGroup("Bjornheim AI core shutdown failed", errors)
