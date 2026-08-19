"""Lifecycle contracts for the explicit Discord adapter boundary."""

import asyncio
from types import SimpleNamespace

import pytest

import kai.discord_adapter as adapter_module
from kai.discord_adapter import DiscordAdapter, DiscordAdapterState


class _FakeDiscordClient:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self._ready = asyncio.Event()
        self._closed = asyncio.Event()

    async def start(self, token: str, *, reconnect: bool = True) -> None:
        assert token == "tok"
        self.events.append("client:start")
        self._ready.set()
        await self._closed.wait()
        self.events.append("client:start-returned")

    async def wait_until_ready(self) -> None:
        await self._ready.wait()

    def is_ready(self) -> bool:
        return self._ready.is_set() and not self._closed.is_set()

    async def close(self) -> None:
        self.events.append("client:close")
        self._closed.set()


class _FakeDelivery:
    def __init__(self, events: list[str], *, label: str = "conversation") -> None:
        self.events = events
        self.ready = True
        self._label = label

    async def wait(self) -> None:
        self.events.append(f"{self._label}:wait")
        await asyncio.Event().wait()

    async def stop(self) -> None:
        self.events.append(f"{self._label}:stop")
        self.ready = False


@pytest.fixture
def adapter_dependencies(monkeypatch):
    events: list[str] = []
    client: _FakeDiscordClient | None = None

    def fake_create_discord_client(_config, _core_services):
        nonlocal client
        events.append("client:create")
        client = _FakeDiscordClient(events)
        return client

    class FakeConversationDelivery:
        @classmethod
        async def open_and_start(cls, _path, _bot, *, authority_epoch_id):
            assert authority_epoch_id == "dae_test"
            events.append("conversation:start")
            return _FakeDelivery(events, label="conversation")

    class FakeNotificationDelivery:
        @classmethod
        async def open_and_start(cls, _path, _bot):
            events.append("notification:start")
            return _FakeDelivery(events, label="notification")

    monkeypatch.setattr(adapter_module, "create_discord_client", fake_create_discord_client)
    monkeypatch.setattr(
        adapter_module,
        "WorkshopDiscordConversationDeliveryService",
        FakeConversationDelivery,
    )
    monkeypatch.setattr(
        adapter_module,
        "WorkshopDiscordNotificationService",
        FakeNotificationDelivery,
    )
    return events


def _adapter() -> DiscordAdapter:
    config = SimpleNamespace(discord_bot_token="tok", session_db_path="/tmp/kai.db")
    core_services = SimpleNamespace(delivery_authority_epoch=SimpleNamespace(epoch_id="dae_test"))
    return DiscordAdapter(config, core_services)  # type: ignore[arg-type]


async def test_adapter_owns_client_and_delivery_lifecycle(adapter_dependencies) -> None:
    adapter = _adapter()

    await adapter.start()
    assert adapter.readiness.as_dict() == {
        "status": "ready",
        "ready": True,
        "components": {"client_ready": True, "conversation_delivery": True, "notification_delivery": True},
    }

    await adapter.stop()
    assert adapter.readiness.state == DiscordAdapterState.STOPPED
    assert adapter_dependencies[:4] == [
        "client:create",
        "client:start",
        "conversation:start",
        "notification:start",
    ]
    assert "conversation:stop" in adapter_dependencies
    assert "notification:stop" in adapter_dependencies
    assert "client:close" in adapter_dependencies
    assert adapter_dependencies.index("notification:stop") < adapter_dependencies.index("conversation:stop")
    assert adapter_dependencies.index("conversation:stop") < adapter_dependencies.index("client:close")


async def test_start_without_token_fails_closed() -> None:
    config = SimpleNamespace(discord_bot_token=None, session_db_path="/tmp/kai.db")
    core_services = SimpleNamespace(delivery_authority_epoch=SimpleNamespace(epoch_id="dae_test"))
    adapter = DiscordAdapter(config, core_services)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="DISCORD_BOT_TOKEN"):
        await adapter.start()
    assert adapter.readiness.state == DiscordAdapterState.FAILED


async def test_start_surfaces_client_task_failure_instead_of_hanging(monkeypatch) -> None:
    """A login failure (bad token, disallowed intents) raises inside the
    background client task, not through wait_until_ready(). Before this was
    fixed, nothing watched that task, so start() hung forever instead of
    surfacing the real error - identical in symptom to a network hang."""

    class _FailingClient:
        async def start(self, token: str, *, reconnect: bool = True) -> None:
            raise RuntimeError("simulated login failure")

        async def wait_until_ready(self) -> None:
            await asyncio.Event().wait()

        def is_ready(self) -> bool:
            return False

        async def close(self) -> None:
            pass

    monkeypatch.setattr(
        adapter_module,
        "create_discord_client",
        lambda _config, _core_services: _FailingClient(),
    )
    adapter = _adapter()

    with pytest.raises(RuntimeError, match="failed before becoming ready"):
        await adapter.start()
    assert adapter.readiness.state == DiscordAdapterState.FAILED


async def test_start_times_out_instead_of_hanging_forever(monkeypatch) -> None:
    """A reachable-but-unresponsive gateway (or any hang neither succeeding
    nor failing) must not block startup forever."""

    class _StuckClient:
        async def start(self, token: str, *, reconnect: bool = True) -> None:
            await asyncio.Event().wait()

        async def wait_until_ready(self) -> None:
            await asyncio.Event().wait()

        def is_ready(self) -> bool:
            return False

        async def close(self) -> None:
            pass

    monkeypatch.setattr(adapter_module, "create_discord_client", lambda _config, _core_services: _StuckClient())
    monkeypatch.setattr(DiscordAdapter, "_READY_TIMEOUT_SECONDS", 0.05)
    adapter = _adapter()

    with pytest.raises(TimeoutError, match="did not become ready"):
        await adapter.start()
    assert adapter.readiness.state == DiscordAdapterState.FAILED


async def test_adapter_rejects_double_start(adapter_dependencies) -> None:
    adapter = _adapter()
    await adapter.start()
    with pytest.raises(RuntimeError, match="cannot start from ready"):
        await adapter.start()
    await adapter.stop()


async def test_adapter_rejects_supervision_before_start() -> None:
    adapter = _adapter()
    with pytest.raises(RuntimeError, match="cannot be supervised while new"):
        await adapter.wait()
