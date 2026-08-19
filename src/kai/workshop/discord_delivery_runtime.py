"""Lifecycle ownership for Workshop Discord delivery.

Mirrors telegram_delivery_runtime.py's shape exactly. The runtime state
machine (DiscordDeliveryRuntime) has no Discord-specific logic in it at all -
it's duplicated here rather than shared with WorkshopTelegramDeliveryRuntime
only because this Phase 1 change is scoped to adding the Discord adapter, not
refactoring the existing Telegram runtime; a follow-up could extract a shared
transport-neutral WorkshopDeliveryRuntime used by both.
"""

from __future__ import annotations

import asyncio
from enum import StrEnum
from pathlib import Path
from typing import Protocol, cast

from kai.workshop.delivery_authority import WorkshopConversationDeliveryAuthority
from kai.workshop.delivery_fragments import WorkshopDeliveryFragments
from kai.workshop.delivery_outbox import (
    CONVERSATION_REPLY_PURPOSE,
    NOTIFICATION_PURPOSE,
    DeliveryRecoveryResult,
    WorkshopDeliveryOutbox,
)
from kai.workshop.discord_delivery import (
    WORKSHOP_CLIENT_TEXT_MODE,
    DiscordTextBot,
    WorkshopDiscordDeliveryAdapter,
    WorkshopDiscordDeliveryWorker,
    WorkshopDiscordStreamingFinalizationAdapter,
    WorkshopDiscordStreamingFinalizationWorker,
)
from kai.workshop.domain import DeliveryAuthorityEpochId
from kai.workshop.github_notifications import (
    GitHubNotification,
    GitHubNotificationRecord,
    WorkshopGitHubNotificationRecorder,
)
from kai.workshop.store import WorkshopEventStore


class _DeliveryLeaseRecovery(Protocol):
    async def recover_expired_leases(self) -> DeliveryRecoveryResult: ...


class _DiscordDeliveryWorkerLoop(Protocol):
    async def run(self, stop_event: asyncio.Event) -> None: ...


class DiscordDeliveryRuntimeState(StrEnum):
    """Observable lifecycle state for the delivery runtime owner."""

    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    FAILED = "failed"


class DiscordDeliveryRuntimeStateError(RuntimeError):
    """The requested runtime transition is not valid in the current state."""


class DiscordDeliveryWorkerExitedError(RuntimeError):
    """The owned worker task exited without a shutdown request."""


class DiscordDeliveryRuntime:
    """Own exactly one recoverable Discord worker task when explicitly started."""

    def __init__(
        self,
        recovery: _DeliveryLeaseRecovery,
        worker: _DiscordDeliveryWorkerLoop,
    ) -> None:
        self._recovery = recovery
        self._worker = worker
        self._lock = asyncio.Lock()
        self._state = DiscordDeliveryRuntimeState.STOPPED
        self._stop_event: asyncio.Event | None = None
        self._task: asyncio.Task[None] | None = None
        self._failure: BaseException | None = None
        self._recovery_result: DeliveryRecoveryResult | None = None

    @property
    def state(self) -> DiscordDeliveryRuntimeState:
        return self._state

    @property
    def ready(self) -> bool:
        return self._state == DiscordDeliveryRuntimeState.READY

    @property
    def has_worker_task(self) -> bool:
        return self._task is not None

    @property
    def failure(self) -> BaseException | None:
        return self._failure

    @property
    def recovery_result(self) -> DeliveryRecoveryResult | None:
        return self._recovery_result

    async def start(self) -> DeliveryRecoveryResult:
        """Recover abandoned work, then create the single owned worker task."""
        async with self._lock:
            if self._state != DiscordDeliveryRuntimeState.STOPPED:
                raise DiscordDeliveryRuntimeStateError(
                    f"Discord delivery runtime cannot start while {self._state.value}"
                )

            self._state = DiscordDeliveryRuntimeState.STARTING
            self._failure = None
            self._recovery_result = None
            try:
                recovered = await self._recovery.recover_expired_leases()
            except asyncio.CancelledError:
                self._state = DiscordDeliveryRuntimeState.STOPPED
                raise
            except Exception as error:
                self._failure = error
                self._state = DiscordDeliveryRuntimeState.FAILED
                raise

            stop_event = asyncio.Event()
            task = asyncio.create_task(
                self._worker.run(stop_event),
                name="kai-workshop-discord-delivery",
            )
            self._stop_event = stop_event
            self._task = task
            self._recovery_result = recovered
            self._state = DiscordDeliveryRuntimeState.READY
            task.add_done_callback(self._worker_done)
            return recovered

    async def wait(self) -> None:
        """Wait for the worker and propagate any unexpected termination."""
        task = self._task
        if task is None:
            if self._failure is not None:
                raise self._failure
            raise DiscordDeliveryRuntimeStateError("Discord delivery runtime has no worker task")

        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.cancelled():
                raise
        except Exception:
            pass
        self._record_completion(task)
        if self._failure is not None:
            raise self._failure

    async def stop(self) -> None:
        """Cooperatively stop polling and await the current serialized iteration."""
        async with self._lock:
            if self._state == DiscordDeliveryRuntimeState.STOPPED:
                return

            task = self._task
            stop_event = self._stop_event
            if task is not None and task.done() and self._failure is None:
                self._record_completion(task, stop_was_requested=False)
            self._state = DiscordDeliveryRuntimeState.STOPPING
            if stop_event is not None:
                stop_event.set()

        if task is not None:
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if not task.cancelled():
                    raise
            except Exception:
                pass
            self._record_completion(task)

        async with self._lock:
            failure = self._failure
            if task is self._task:
                self._task = None
                self._stop_event = None
            self._state = DiscordDeliveryRuntimeState.STOPPED

        if failure is not None:
            raise failure

    def _worker_done(self, task: asyncio.Task[None]) -> None:
        self._record_completion(task)

    def _record_completion(
        self,
        task: asyncio.Task[None],
        *,
        stop_was_requested: bool | None = None,
    ) -> None:
        if task is not self._task or not task.done():
            return

        if stop_was_requested is None:
            stop_was_requested = self._stop_event is not None and self._stop_event.is_set()
        if task.cancelled():
            failure: BaseException | None = DiscordDeliveryWorkerExitedError(
                "Discord delivery worker was cancelled outside its runtime owner"
            )
        else:
            failure = task.exception()
            if failure is None and not stop_was_requested:
                failure = DiscordDeliveryWorkerExitedError("Discord delivery worker exited without a shutdown request")

        if failure is not None:
            self._failure = failure
            self._state = DiscordDeliveryRuntimeState.FAILED


class WorkshopDiscordConversationDeliveryService:
    """Own dedicated Discord stores and workers for one core-owned epoch."""

    def __init__(
        self,
        store: WorkshopEventStore,
        client_store: WorkshopEventStore,
        runtime: DiscordDeliveryRuntime,
        client_runtime: DiscordDeliveryRuntime,
    ) -> None:
        self._store = store
        self._client_store = client_store
        self._runtime = runtime
        self._client_runtime = client_runtime
        self._closed = False

    @classmethod
    async def open_and_start(
        cls,
        database_path: Path,
        bot: DiscordTextBot,
        *,
        authority_epoch_id: DeliveryAuthorityEpochId,
        worker_id: str = "kai-discord-conversation-finalization",
    ) -> WorkshopDiscordConversationDeliveryService:
        """Validate core authority and recover Discord work before ready."""
        store = await WorkshopEventStore.open(database_path)
        client_store: WorkshopEventStore | None = None
        client_runtime: DiscordDeliveryRuntime | None = None
        try:
            client_store = await WorkshopEventStore.open(database_path)
            epoch = await WorkshopConversationDeliveryAuthority(store).active_epoch()
            if epoch.epoch_id != authority_epoch_id:
                raise RuntimeError("Discord delivery epoch does not match core authority")
            outbox = WorkshopDeliveryOutbox(store)
            fragments = WorkshopDeliveryFragments(store)
            client_worker = WorkshopDiscordDeliveryWorker(
                WorkshopDeliveryOutbox(client_store),
                WorkshopDeliveryFragments(client_store),
                WorkshopDiscordDeliveryAdapter(cast(DiscordTextBot, bot)),
                worker_id=f"{worker_id}-client-message",
                purpose=CONVERSATION_REPLY_PURPOSE,
                modes=(WORKSHOP_CLIENT_TEXT_MODE,),
            )
            client_runtime = DiscordDeliveryRuntime(client_worker, client_worker)
            await client_runtime.start()
            worker = WorkshopDiscordStreamingFinalizationWorker(
                outbox,
                fragments,
                WorkshopDiscordStreamingFinalizationAdapter(cast(DiscordTextBot, bot)),
                worker_id=worker_id,
                authority_epoch_id=epoch.epoch_id,
            )
            runtime = DiscordDeliveryRuntime(worker, worker)
            await runtime.start()
        except BaseException:
            if client_runtime is not None:
                try:
                    await client_runtime.stop()
                except Exception:
                    pass
            if client_store is not None:
                await client_store.close()
            await store.close()
            raise
        assert client_store is not None
        assert client_runtime is not None
        return cls(store, client_store, runtime, client_runtime)

    @property
    def ready(self) -> bool:
        return not self._closed and self._runtime.ready and self._client_runtime.ready

    @property
    def runtime(self) -> DiscordDeliveryRuntime:
        return self._runtime

    async def wait(self) -> None:
        waits = {
            asyncio.create_task(self._runtime.wait()),
            asyncio.create_task(self._client_runtime.wait()),
        }
        done, pending = await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            await task

    async def stop(self) -> None:
        if self._closed:
            return
        try:
            results = await asyncio.gather(
                self._client_runtime.stop(),
                self._runtime.stop(),
                return_exceptions=True,
            )
        finally:
            self._closed = True
            await self._client_store.close()
            await self._store.close()
        for result in results:
            if isinstance(result, BaseException):
                raise result


class WorkshopDiscordNotificationService:
    """Own canonical GitHub recording and its dedicated Discord worker."""

    def __init__(
        self,
        recorder_store: WorkshopEventStore,
        worker_store: WorkshopEventStore,
        recorder: WorkshopGitHubNotificationRecorder,
        runtime: DiscordDeliveryRuntime,
    ) -> None:
        self._recorder_store = recorder_store
        self._worker_store = worker_store
        self._recorder = recorder
        self._runtime = runtime
        self._closed = False

    @classmethod
    async def open_and_start(
        cls,
        database_path: Path,
        bot: DiscordTextBot,
        *,
        worker_id: str = "kai-discord-notification",
    ) -> WorkshopDiscordNotificationService:
        recorder_store = await WorkshopEventStore.open(database_path)
        worker_store: WorkshopEventStore | None = None
        runtime: DiscordDeliveryRuntime | None = None
        try:
            worker_store = await WorkshopEventStore.open(database_path)
            worker = WorkshopDiscordDeliveryWorker(
                WorkshopDeliveryOutbox(worker_store),
                WorkshopDeliveryFragments(worker_store),
                WorkshopDiscordDeliveryAdapter(cast(DiscordTextBot, bot)),
                worker_id=worker_id,
                purpose=NOTIFICATION_PURPOSE,
                modes=("text",),
            )
            runtime = DiscordDeliveryRuntime(worker, worker)
            await runtime.start()
        except BaseException:
            if runtime is not None:
                try:
                    await runtime.stop()
                except Exception:
                    pass
            if worker_store is not None:
                await worker_store.close()
            await recorder_store.close()
            raise
        assert worker_store is not None
        assert runtime is not None
        return cls(
            recorder_store,
            worker_store,
            WorkshopGitHubNotificationRecorder(recorder_store),
            runtime,
        )

    @property
    def ready(self) -> bool:
        return not self._closed and self._runtime.ready

    async def record(self, notification: GitHubNotification) -> GitHubNotificationRecord | None:
        if self._closed:
            raise RuntimeError("Workshop Discord notification service is closed")
        return await self._recorder.record(notification)

    async def wait(self) -> None:
        await self._runtime.wait()

    async def stop(self) -> None:
        if self._closed:
            return
        try:
            await self._runtime.stop()
        finally:
            self._closed = True
            await self._worker_store.close()
            await self._recorder_store.close()
