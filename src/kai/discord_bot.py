"""Discord DM handling for Bjornheim AI - Phase 1 (message flow + a starter command set).

Mirrors bot.py's private-text flow (`handle_message` -> `_handle_workshop_private_text`)
for the one path every new adapter needs: DM text in, canonical Workshop
execution, streamed reply out. Unlike bot.py, this module has no legacy
non-Workshop fallback path to support (Discord is a new transport with no
pre-Workshop history), so the flow here is deliberately smaller than
bot.py's handle_message.

Scope, deliberately narrower than bot.py for this phase:
    - DMs only. Messages from guild/server text channels are ignored.
    - No TOTP gate, no voice mode, no legacy log_message()/memory-extraction
      provenance write (log_message's `chat_id` is documented as a Telegram
      chat id and resolves through a Telegram-specific compatibility
      registry in kai.history - reusing it for a Discord id would either
      fail to resolve or silently create an incorrectly-namespaced entry;
      see the Phase 1 report for the full reasoning). Discord users' turns
      are durably stored by the canonical Workshop event store regardless
      (via private_text_execution.accept/execute), so nothing is lost for
      the conversation itself - only the legacy JSONL shadow-copy and
      Haiku memory-fact-extraction pipeline are skipped for now.
    - Slash commands: /stats and /new only, wrapping the same sessions.py
      functions bot.py's equivalent handlers call. /model, /memory,
      /workspace, /job, /settings, /voice, /webhooks, /github, /review are
      deferred to a follow-up phase.
"""

from __future__ import annotations

import asyncio
import logging
import time

import discord
from discord import app_commands
from discord.ext import commands

from kai import sessions
from kai.application_host import KaiCoreServices
from kai.bot import _stream_publishable_prefix
from kai.config import Config, UserConfig
from kai.workshop.domain import MessageId, RunId
from kai.workshop.execution_coordinator import CanonicalExecutionDisposition
from kai.workshop.guardian_access import GuardianAccessError, resolve_guardian_target
from kai.workshop.guardian_transcript import read_target_transcript, summarize_transcript
from kai.workshop.inbound import InboundMessage

log = logging.getLogger(__name__)

_DISCORD_TEXT_LIMIT = 2000
_EDIT_INTERVAL = 2.0


def _truncate_for_discord(text: str, max_len: int = _DISCORD_TEXT_LIMIT) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


class KaiDiscordClient(commands.Bot):
    """Discord client carrying Bjornheim AI's config and core services.

    Subclassing (rather than free-standing @client.event functions) is the
    simplest way to thread config/core_services into on_message and the
    slash-command callbacks below without module-level globals.
    """

    def __init__(self, config: Config, core_services: KaiCoreServices) -> None:
        intents = discord.Intents.default()
        intents.message_content = True  # required to read DM text bodies
        intents.dm_messages = True
        super().__init__(command_prefix=commands.when_mentioned, intents=intents)
        self.kai_config = config
        self.kai_core_services = core_services

    async def setup_hook(self) -> None:
        _register_commands(self.tree)
        await self.tree.sync()

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return
        if message.guild is not None:
            # DM-only for Phase 1; ignore anything from a guild/server channel.
            return
        if not message.content:
            return
        await _handle_dm_text(self, message)


def create_discord_client(config: Config, core_services: KaiCoreServices) -> discord.Client:
    return KaiDiscordClient(config, core_services)


def _authorized_user(config: Config, discord_user_id: int) -> UserConfig | None:
    """Mirror bot.py's _require_auth: silently drop unauthorized senders.

    Returns the sender's UserConfig, or None if unauthorized (caller should
    silently return without responding, to avoid revealing the bot's
    existence - same policy as bot.py's Telegram auth).
    """
    if discord_user_id not in config.allowed_discord_user_ids:
        return None
    return config.get_user_config_by_discord(discord_user_id)


async def _handle_dm_text(client: KaiDiscordClient, message: discord.Message) -> None:
    config = client.kai_config
    user_config = _authorized_user(config, message.author.id)
    if user_config is None:
        return

    chat_id = message.author.id
    prompt = message.content

    execution = client.kai_core_services.private_text_execution
    try:
        acceptance = await execution.accept(
            InboundMessage(
                transport="discord",
                update_id=str(message.id),
                message_id=str(message.id),
                sender_subject=str(chat_id),
                channel_subject=str(chat_id),
                body=prompt,
                occurred_at=message.created_at,
            )
        )
        aggregate_id = acceptance.message.event.envelope.aggregate_id
        if not isinstance(aggregate_id, MessageId):
            raise RuntimeError("Workshop command acceptance returned a non-message aggregate")
        run_id = acceptance.run.run_id
    except Exception:
        log.exception("Workshop private-text acceptance failed (discord message_id=%s)", message.id)
        await message.channel.send("Bjornheim AI could not safely accept this request. Please try again.")
        return

    await _stream_and_deliver(client, message, run_id=run_id)


async def _stream_and_deliver(client: KaiDiscordClient, source_message: discord.Message, *, run_id: RunId) -> None:
    execution = client.kai_core_services.private_text_execution
    live_msg: discord.Message | None = None
    last_edit_time = 0.0
    last_edit_text = ""

    async def observe(event) -> None:
        nonlocal live_msg, last_edit_time, last_edit_text
        if not event.text_so_far:
            if not event.activity:
                return
            publishable = f"_{event.activity}_"
        else:
            publishable = _stream_publishable_prefix(event.text_so_far)
        if publishable is None or publishable == last_edit_text:
            return
        now = time.monotonic()
        if live_msg is None:
            live_msg = await source_message.channel.send(content=_truncate_for_discord(publishable))
            last_edit_time = now
            last_edit_text = publishable
        elif now - last_edit_time >= _EDIT_INTERVAL:
            try:
                await live_msg.edit(content=_truncate_for_discord(publishable))
            except discord.HTTPException:
                pass
            else:
                last_edit_time = now
                last_edit_text = publishable

    typing_task = asyncio.create_task(_keep_typing(source_message.channel))
    try:
        result = await execution.execute(run_id, stream_observer=observe)
    finally:
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass

    if result.disposition == CanonicalExecutionDisposition.PREPARATION_DEFERRED:
        await source_message.channel.send("Bjornheim AI could not safely prepare this request. Please try again.")
        return
    if result.disposition in {
        CanonicalExecutionDisposition.ACTIVE_REPLAY,
        CanonicalExecutionDisposition.CANCELLATION_PENDING_REPLAY,
        CanonicalExecutionDisposition.TERMINAL_REPLAY,
    }:
        return
    if result.terminal is None:
        raise RuntimeError("Canonical execution settled without a terminal transaction")

    final_text = result.terminal.body
    if result.session_id and result.selection is not None:
        await sessions.save_session(source_message.author.id, result.session_id, result.selection.model)

    if live_msg is not None and final_text == last_edit_text:
        return
    if live_msg is not None:
        try:
            await live_msg.edit(content=_truncate_for_discord(final_text))
            return
        except discord.HTTPException:
            pass
    await source_message.channel.send(content=_truncate_for_discord(final_text))


async def _keep_typing(channel: discord.abc.Messageable) -> None:
    while True:
        try:
            async with channel.typing():
                await asyncio.sleep(8)
        except Exception:
            await asyncio.sleep(4)


# ── Slash commands ───────────────────────────────────────────────────


def _register_commands(tree: app_commands.CommandTree) -> None:
    @tree.command(name="stats", description="Show session info and process status")
    async def stats(interaction: discord.Interaction) -> None:
        client = interaction.client
        assert isinstance(client, KaiDiscordClient)
        config = client.kai_config
        user_config = _authorized_user(config, interaction.user.id)
        if user_config is None:
            await interaction.response.send_message("Not authorized.", ephemeral=True)
            return
        chat_id = interaction.user.id
        pool = client.kai_core_services.subprocess_pool
        session_stats = await sessions.get_stats(chat_id)
        alive = pool.is_alive(chat_id)
        if not session_stats:
            await interaction.response.send_message(f"No active session.\nProcess alive: {alive}", ephemeral=True)
            return
        await interaction.response.send_message(
            f"Session: {session_stats['session_id'][:8]}...\n"
            f"Model: {session_stats['model']}\n"
            f"Started: {session_stats['created_at']}\n"
            f"Last used: {session_stats['last_used_at']}\n"
            f"Process alive: {alive}",
            ephemeral=True,
        )

    @tree.command(name="new", description="Start a fresh session")
    async def new(interaction: discord.Interaction) -> None:
        client = interaction.client
        assert isinstance(client, KaiDiscordClient)
        config = client.kai_config
        user_config = _authorized_user(config, interaction.user.id)
        if user_config is None:
            await interaction.response.send_message("Not authorized.", ephemeral=True)
            return
        chat_id = interaction.user.id
        await sessions.clear_session(chat_id)
        await interaction.response.send_message("Started a fresh session.", ephemeral=True)

    @tree.command(name="monitor", description="Guardian-only: summarize a monitored family member's recent activity")
    @app_commands.describe(name="The monitored user's configured name", count="How many recent turns to consider")
    async def monitor(interaction: discord.Interaction, name: str, count: app_commands.Range[int, 1, 100] = 40) -> None:
        client = interaction.client
        assert isinstance(client, KaiDiscordClient)
        config = client.kai_config
        guardian_config = _authorized_user(config, interaction.user.id)
        if guardian_config is None:
            await interaction.response.send_message("Not authorized.", ephemeral=True)
            return

        # Deferred: this path does a canonical-timeline read plus an LLM
        # summarization call, which can exceed Discord's 3-second initial
        # interaction-response window (unlike /stats and /new above, which
        # answer synchronously from already-loaded state).
        await interaction.response.defer(ephemeral=True)
        reply = await _guardian_transcript_reply(
            core_services=client.kai_core_services,
            config=config,
            guardian_config=guardian_config,
            target_name=name,
            count=count,
        )
        await interaction.followup.send(_truncate_for_discord(reply))


async def _guardian_transcript_reply(
    *,
    core_services: KaiCoreServices,
    config: Config,
    guardian_config: UserConfig,
    target_name: str,
    count: int,
) -> str:
    """Resolve + read + summarize one guardian /monitor request.

    Split out from the `monitor` slash-command closure above so the actual
    authorization/read/summarize logic is unit-testable without going
    through discord.py's Interaction/CommandTree machinery - the closure
    itself is left as a thin adapter (defer, call this, send the result),
    matching how thin `/stats` and `/new` already are.
    """
    try:
        target = resolve_guardian_target(
            core_services.workshop_id,
            config.user_configs.values(),
            guardian_config,
            target_name,
        )
    except GuardianAccessError:
        # Deliberately identical wording whether `target_name` isn't
        # configured at all or simply isn't one of this guardian's wards -
        # see GuardianAccessError's docstring on why the two must not be
        # distinguishable to the caller.
        return "Not found, or you are not a configured guardian of that name."

    messages = await read_target_transcript(core_services.client_store, target, limit=count)
    return await summarize_transcript(
        messages,
        target_name=target.name,
        guardian_config=guardian_config,
        config=config,
    )
