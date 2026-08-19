"""End-to-end smoke test for the Discord adapter + guardian pipeline.

Every other Discord/guardian test file exercises its own module in
isolation with mocks at that module's boundary. This file is the one
place that boots the real `KaiApplicationHost`, attaches a real
`DiscordAdapter`, and pushes fake DMs all the way through: ingest via
`discord_bot._handle_dm_text` -> `private_text_execution` -> (for a
monitored user) the real `on_completed` hook -> `safety_classifier` ->
`guardian_alerts` -> the real `WorkshopDeliveryOutbox` -> the real,
running `WorkshopDiscordNotificationService` worker -> a "sent" call on
a fake Discord user.

Mocks are placed only at the three true external edges:
  - the Discord gateway itself (`create_discord_client`, patched to a
    fake client so no real network connection is attempted)
  - the backend agent subprocess (`SubprocessPool.prepare_execution`,
    patched to return a canned reply instead of spawning a real
    `codex`/`claude` CLI process)
  - the safety-classifier / summarizer LLM calls (`ClaudeOneShotReasoner`
    in `safety_classifier.py` and `guardian_transcript.py`, patched to
    avoid a real subprocess call requiring live authentication)
Every other component (store, bootstrap, private_text_execution, the
on_completed hook, guardian_access, guardian_alerts, the outbox, and
the Discord delivery worker) is the real production object.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import kai.discord_adapter as discord_adapter_module
import kai.workshop.guardian_transcript as guardian_transcript_module
import kai.workshop.safety_classifier as safety_classifier_module
from kai import sessions
from kai.application_host import KaiApplicationHost
from kai.backend import AgentResponse, StreamEvent
from kai.config import Config, UserConfig
from kai.discord_adapter import DiscordAdapter
from kai.discord_bot import _guardian_transcript_reply, _handle_dm_text
from kai.oneshot import OneShotResult
from kai.pool import PreparedBackendSelection, SubprocessPool
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop, bootstrap_human_principal_id
from kai.workshop.domain import PrincipalId
from kai.workshop.storage_namespaces import WorkshopPrincipalStorageNamespace, WorkshopPrincipalStorageRegistry
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id, profile_registry

_PLAIN_ID = 1001
_KID_ID = 2002
_GUARDIAN_ID = 3003
_RANDO_ID = 4004

_FAKE_REPLY = "Sure, here's some help with that."


# ── Fake Discord gateway (the only real network boundary) ──────────────


class _NullAsyncCM:
    async def __aenter__(self) -> _NullAsyncCM:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False


class _FakeDiscordMessage:
    def __init__(self, channel: _FakeDiscordChannel, msg_id: int, content: str) -> None:
        self.channel = channel
        self.id = msg_id
        self.content = content

    async def edit(self, content: str = "", **_kw: object) -> _FakeDiscordMessage:
        self.content = content
        self.channel.sent.append(content)
        return self


class _FakeDiscordChannel:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self._messages: dict[int, _FakeDiscordMessage] = {}
        self._next_id = 1

    async def send(self, content: str = "", **_kw: object) -> _FakeDiscordMessage:
        msg = _FakeDiscordMessage(self, self._next_id, content)
        self._next_id += 1
        self.sent.append(content)
        self._messages[msg.id] = msg
        return msg

    def typing(self) -> _NullAsyncCM:
        return _NullAsyncCM()

    async def fetch_message(self, message_id: int) -> _FakeDiscordMessage:
        return self._messages[message_id]


class _FakeDiscordUser:
    def __init__(self, user_id: int) -> None:
        self.id = user_id
        self.bot = False
        self.dm_channel = _FakeDiscordChannel()

    async def create_dm(self) -> _FakeDiscordChannel:
        return self.dm_channel

    async def send(self, content: str = "", **_kw: object) -> _FakeDiscordMessage:
        return await self.dm_channel.send(content=content)


class _FakeGatewayClient:
    def __init__(self, users: dict[int, _FakeDiscordUser]) -> None:
        self._users = users
        self._ready = asyncio.Event()
        self._closed = asyncio.Event()

    async def start(self, token: str, *, reconnect: bool = True) -> None:
        self._ready.set()
        await self._closed.wait()

    async def wait_until_ready(self) -> None:
        await self._ready.wait()

    def is_ready(self) -> bool:
        return self._ready.is_set() and not self._closed.is_set()

    async def close(self) -> None:
        self._closed.set()

    def get_user(self, user_id: int) -> _FakeDiscordUser | None:
        return self._users.get(user_id)

    async def fetch_user(self, user_id: int) -> _FakeDiscordUser:
        return self._users[user_id]


class _FakeAuthor:
    def __init__(self, user: _FakeDiscordUser) -> None:
        self.id = user.id
        self.bot = False


class _FakeInboundMessage:
    """Stands in for discord.Message: only the attributes _handle_dm_text touches."""

    def __init__(self, user: _FakeDiscordUser, content: str, msg_id: int) -> None:
        self.author = _FakeAuthor(user)
        self.guild = None
        self.content = content
        self.id = msg_id
        self.created_at = datetime.now(UTC)
        self.channel = user.dm_channel


class _FakeDiscordClientHandle:
    """Stands in for KaiDiscordClient: only kai_config/kai_core_services are read."""

    def __init__(self, config: Config, core_services: object) -> None:
        self.kai_config = config
        self.kai_core_services = core_services


# ── Fake backend subprocess (the second real edge) ──────────────────────


class _FakeRuntime:
    def __init__(self) -> None:
        self.selection = PreparedBackendSelection(backend="codex", provider="openai", model="gpt-5.6-sol")
        self.workspace = Path("/tmp")

    def validate_current(self) -> None:
        return None

    async def cancel(self) -> None:
        return None

    async def stream(self, _prompt: object):
        yield StreamEvent(
            text_so_far=_FAKE_REPLY,
            done=True,
            response=AgentResponse(success=True, text=_FAKE_REPLY, session_id="fake-session", duration_ms=1),
        )


async def _fake_prepare_execution(_self: SubprocessPool, _chat_id: int) -> _FakeRuntime:
    return _FakeRuntime()


# ── Fake LLM calls (the third real edge) ─────────────────────────────────


class _FakeFlaggingReasoner:
    """safety_classifier.py's ClaudeOneShotReasoner substitute: always flags."""

    def __init__(self, *_a: object, **_kw: object) -> None:
        pass

    async def run(self, **_kw: object) -> OneShotResult:
        payload = {"flagged": True, "category": "self_harm", "summary": "expressed distress"}
        return OneShotResult(text=json.dumps(payload), backend="claude", model="haiku")


class _FakeSummaryReasoner:
    """guardian_transcript.py's ClaudeOneShotReasoner substitute: canned summary."""

    def __init__(self, *_a: object, **_kw: object) -> None:
        pass

    async def run(self, **_kw: object) -> OneShotResult:
        return OneShotResult(text="Summary: recent conversation looked ordinary.", backend="claude", model="haiku")


# ── Stack assembly ────────────────────────────────────────────────────


class _Stack:
    def __init__(self, host: KaiApplicationHost, adapter: DiscordAdapter, config: Config, users: dict[int, _FakeDiscordUser]):
        self.host = host
        self.adapter = adapter
        self.config = config
        self.users = users

    def client_handle(self) -> _FakeDiscordClientHandle:
        return _FakeDiscordClientHandle(self.config, self.host.services)

    async def stop(self) -> None:
        await self.host.stop()
        await sessions.close_db()


async def _build_stack(tmp_path: Path, monkeypatch) -> _Stack:
    monkeypatch.setattr(SubprocessPool, "prepare_execution", _fake_prepare_execution)
    monkeypatch.setattr(safety_classifier_module, "ClaudeOneShotReasoner", _FakeFlaggingReasoner)
    monkeypatch.setattr(guardian_transcript_module, "ClaudeOneShotReasoner", _FakeSummaryReasoner)

    users = {uid: _FakeDiscordUser(uid) for uid in (_PLAIN_ID, _KID_ID, _GUARDIAN_ID, _RANDO_ID)}
    monkeypatch.setattr(
        discord_adapter_module,
        "create_discord_client",
        lambda _config, _core_services: _FakeGatewayClient(users),
    )

    database = tmp_path / "kai.db"
    # Mirrors main.py's real startup sequence: the legacy per-user
    # sessions/settings DB (last-used model, /stats, etc.) is a separate
    # module-level connection from the Workshop event store and must be
    # initialized before any code path that calls sessions.save_session /
    # get_stats - both discord_bot.py and bot.py call these unconditionally
    # after a completed turn.
    await sessions.init_db(database)
    profiles = profile_registry(_PLAIN_ID, _KID_ID, _GUARDIAN_ID, _RANDO_ID)

    humans = tuple(
        BootstrapHuman(
            display_name=name,
            role="member",
            transport="discord",
            external_subject=str(uid),
            external_channel_id=str(uid),
            runtime_profile_id=profile_id(uid),
        )
        for uid, name in (
            (_PLAIN_ID, "PlainUser"),
            (_KID_ID, "Kid"),
            (_GUARDIAN_ID, "Guardian"),
            (_RANDO_ID, "Rando"),
        )
    )
    store = await WorkshopEventStore.open(database)
    try:
        bootstrap = await bootstrap_default_workshop(store, humans)
    finally:
        await store.close()

    principal_ids = {
        uid: bootstrap_human_principal_id(bootstrap.workshop_id, "discord", str(uid))
        for uid in (_PLAIN_ID, _KID_ID, _GUARDIAN_ID, _RANDO_ID)
    }
    principal_storage = WorkshopPrincipalStorageRegistry(
        tuple(
            WorkshopPrincipalStorageNamespace(PrincipalId(str(principal_ids[uid])), profile_id(uid), uid)
            for uid in (_PLAIN_ID, _KID_ID, _GUARDIAN_ID, _RANDO_ID)
        )
    )

    user_configs = {
        _PLAIN_ID: UserConfig(name="PlainUser", discord_id=_PLAIN_ID),
        _KID_ID: UserConfig(name="Kid", discord_id=_KID_ID, monitored=True),
        _GUARDIAN_ID: UserConfig(name="Guardian", discord_id=_GUARDIAN_ID, guardian_of=["Kid"]),
        _RANDO_ID: UserConfig(name="Rando", discord_id=_RANDO_ID),
    }
    config = Config(
        telegram_bot_token=None,
        allowed_user_ids=set(),
        session_db_path=database,
        agent_idle_timeout=0,
        default_backend="codex",
        default_model="gpt-5.6-sol",
        discord_bot_token="tok",
        allowed_discord_user_ids=set(user_configs),
        user_configs_by_discord_id=dict(user_configs),
        enabled_adapters=frozenset({"discord"}),
        user_configs=dict(user_configs),
    )

    host = KaiApplicationHost(
        config=config,
        runtime_profiles=profiles,
        principal_storage=principal_storage,
        services_info=[],
        registered_backend_ids=frozenset({"codex"}),
        workshop_id=bootstrap.workshop_id,
    )
    core_services = await host.start()
    adapter = DiscordAdapter(config, core_services)
    await host.attach_adapter("discord", adapter)

    return _Stack(host, adapter, config, users)


async def _wait_until(predicate, *, timeout: float = 8.0, interval: float = 0.05) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return predicate()


# ── Tests ─────────────────────────────────────────────────────────────


async def test_plain_user_dm_gets_a_reply_and_triggers_no_classifier(tmp_path, monkeypatch) -> None:
    classify_calls: list[str] = []
    real_classify = safety_classifier_module.classify

    async def _spying_classify(user_text: str, *a: object, **kw: object):
        classify_calls.append(user_text)
        return await real_classify(user_text, *a, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(safety_classifier_module, "classify", _spying_classify)
    # application_host.py imported classify_safety by reference at module
    # load time (`from ... import classify as classify_safety`); patch the
    # binding it actually calls, not just the source module's attribute.
    import kai.application_host as host_module

    monkeypatch.setattr(host_module, "classify_safety", _spying_classify)

    stack = await _build_stack(tmp_path, monkeypatch)
    try:
        message = _FakeInboundMessage(stack.users[_PLAIN_ID], "What's a good pasta recipe?", msg_id=1)
        await _handle_dm_text(stack.client_handle(), message)  # type: ignore[arg-type]

        assert stack.users[_PLAIN_ID].dm_channel.sent, "plain user should receive a reply"
        assert _FAKE_REPLY in stack.users[_PLAIN_ID].dm_channel.sent[-1]

        # Give the fire-and-forget on_completed hook a moment to run (it
        # would only do anything for a monitored user, so this also
        # confirms it ran and correctly no-op'd for a plain one).
        await asyncio.sleep(0.3)
        assert classify_calls == [], "non-monitored user must never reach the classifier"
        assert stack.users[_GUARDIAN_ID].dm_channel.sent == []
    finally:
        await stack.stop()


async def test_monitored_user_flagged_message_alerts_guardian_end_to_end(tmp_path, monkeypatch) -> None:
    stack = await _build_stack(tmp_path, monkeypatch)
    try:
        message = _FakeInboundMessage(stack.users[_KID_ID], "I want to kill myself", msg_id=2)
        await _handle_dm_text(stack.client_handle(), message)  # type: ignore[arg-type]

        assert stack.users[_KID_ID].dm_channel.sent, "kid should still get a normal reply"

        delivered = await _wait_until(lambda: bool(stack.users[_GUARDIAN_ID].dm_channel.sent))
        assert delivered, "guardian alert never reached the fake Discord client"
        alert_text = stack.users[_GUARDIAN_ID].dm_channel.sent[-1]
        assert "Kid" in alert_text
        assert "self_harm" in alert_text
        assert "/monitor Kid" in alert_text
    finally:
        await stack.stop()


async def test_guardian_monitor_command_summarizes_target(tmp_path, monkeypatch) -> None:
    stack = await _build_stack(tmp_path, monkeypatch)
    try:
        message = _FakeInboundMessage(stack.users[_KID_ID], "How do volcanoes work?", msg_id=3)
        await _handle_dm_text(stack.client_handle(), message)  # type: ignore[arg-type]

        reply = await _guardian_transcript_reply(
            core_services=stack.host.services,
            config=stack.config,
            guardian_config=stack.config.user_configs[_GUARDIAN_ID],
            target_name="Kid",
            count=10,
        )
        assert "Summary: recent conversation looked ordinary." in reply
    finally:
        await stack.stop()


async def test_guardian_monitor_command_refuses_unauthorized_caller(tmp_path, monkeypatch) -> None:
    stack = await _build_stack(tmp_path, monkeypatch)
    try:
        reply = await _guardian_transcript_reply(
            core_services=stack.host.services,
            config=stack.config,
            guardian_config=stack.config.user_configs[_RANDO_ID],
            target_name="Kid",
            count=10,
        )
        assert "not" in reply.lower()

        # Same refusal wording for "not your ward" and "doesn't exist" -
        # guardian_access.py's fail-closed design (see its module docstring).
        missing_target_reply = await _guardian_transcript_reply(
            core_services=stack.host.services,
            config=stack.config,
            guardian_config=stack.config.user_configs[_GUARDIAN_ID],
            target_name="NoSuchPerson",
            count=10,
        )
        assert missing_target_reply == reply
    finally:
        await stack.stop()


async def test_plain_users_guardian_of_kid_has_no_effect_without_monitored_flag(tmp_path, monkeypatch) -> None:
    """A non-monitored user's turns never reach the classifier, even if
    someone happens to be configured as their guardian - `monitored` gates
    the pipeline, `guardian_of` alone does not (see UserConfig's docstring)."""
    stack = await _build_stack(tmp_path, monkeypatch)
    try:
        stack.config.user_configs[_PLAIN_ID].guardian_of.clear()  # sanity: PlainUser has no guardian at all
        message = _FakeInboundMessage(stack.users[_PLAIN_ID], "I want to kill myself", msg_id=4)
        await _handle_dm_text(stack.client_handle(), message)  # type: ignore[arg-type]
        await asyncio.sleep(0.3)
        assert stack.users[_GUARDIAN_ID].dm_channel.sent == [], "PlainUser is not monitored; no alert should fire"
    finally:
        await stack.stop()
