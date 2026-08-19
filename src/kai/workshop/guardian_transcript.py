"""On-demand guardian transcript deep-dive: read + summarize a target's turns.

guardian_access.py deliberately stops at identity resolution (principal_id/
channel_id per transport) and does not read any content - see its module
docstring. This module is the caller that actually reads content, for the
one on-demand path that needs it: a guardian pulling a monitored user's
recent conversation on request (discord_bot.py's "/monitor transcript"
command).

Reads go through kai.workshop.timeline.read_channel_timeline - the
canonical, transport-independent Workshop message log - rather than
kai.history's JSONL shadow-copy. history.py's chat_id-keyed reader only
covers Telegram (Discord turns are never written there; see
discord_bot.py's module docstring), and even for Telegram users the
canonical timeline is the more complete, non-legacy source. Every turn
either transport accepts through private_text_execution lands in the
canonical timeline regardless of transport.
"""

from __future__ import annotations

import logging

from kai.config import Config, ModelRole, UserConfig, resolve_user_model
from kai.oneshot import ClaudeOneShotReasoner, OneShotError
from kai.workshop.domain import ChannelId, PrincipalId
from kai.workshop.guardian_access import GuardianTarget
from kai.workshop.store import WorkshopEventStore
from kai.workshop.timeline import ChannelTimelineAuthorizer, TimelineMessage, read_channel_timeline

log = logging.getLogger(__name__)

_DEFAULT_TURN_LIMIT = 40
_MAX_TURN_LIMIT = 100
_SUMMARY_TIMEOUT_S = 45.0

_SUMMARY_SYSTEM_PROMPT = (
    "You are summarizing a family member's recent conversation with a personal AI "
    "assistant, for a parent/guardian who has been granted access to review it. "
    "Write a brief, neutral, factual summary of what topics came up and what the "
    "assistant helped with. If anything in the conversation reads as concerning "
    "(safety, wellbeing, distress, or content clearly inappropriate for a minor), "
    "say so plainly and specifically at the start of the summary. Do not moralize, "
    "do not pad with reassurance, do not invent detail that is not in the "
    "transcript. Plain text only, no markdown headers. Keep it under 1500 characters."
)


class _AlreadyAuthorizedTimelineReader:
    """A trivially-permissive ChannelTimelineAuthorizer.

    read_channel_timeline() enforces its own principal-is-a-member check by
    design (self-scoped reads for every other caller in this codebase). This
    module's caller has already cleared a DIFFERENT, stricter check before
    ever reaching here: guardian_access.resolve_guardian_target() confirmed
    the requesting guardian is explicitly listed as a guardian of this exact
    target in users.yaml. Re-running the self-scoped membership check with
    the target's OWN principal_id here would trivially pass anyway (the
    target is of course a member of their own channel) - it would not add
    real protection, just a second query. Reusing the target's principal_id
    is what makes that check meaningless here, not a mistake; the real gate
    already ran in guardian_access before this class is ever constructed.
    Never expose this authorizer on a path guardian_access has not already
    gated.
    """

    async def can_read_channel(self, principal_id: PrincipalId, channel_id: ChannelId) -> bool:
        return True


async def read_target_transcript(
    store: WorkshopEventStore,
    target: GuardianTarget,
    *,
    limit: int = _DEFAULT_TURN_LIMIT,
) -> list[TimelineMessage]:
    """Read up to `limit` recent canonical messages across all of target's channels.

    A user with both telegram_id and discord_id has one channel per
    transport (see guardian_access.GuardianTarget docstring); this merges
    across all of them so a guardian sees the whole picture regardless of
    which transport the target actually used.
    """
    limit = max(1, min(limit, _MAX_TURN_LIMIT))
    authorizer: ChannelTimelineAuthorizer = _AlreadyAuthorizedTimelineReader()
    collected: list[TimelineMessage] = []
    for identity in target.identities:
        page = await read_channel_timeline(
            store,
            principal_id=identity.principal_id,
            channel_id=identity.channel_id,
            authorizer=authorizer,
            limit=limit,
        )
        collected.extend(page.messages)
    collected.sort(key=lambda message: message.created_at)
    if len(collected) > limit:
        collected = collected[-limit:]
    return collected


def _format_transcript(target_name: str, messages: list[TimelineMessage]) -> str:
    if not messages:
        return f"{target_name} has no recorded conversation yet."
    lines = [f"Recent conversation for {target_name}:"]
    for message in messages:
        speaker = target_name if message.author_kind == "human" else "assistant"
        lines.append(f"[{message.created_at.isoformat()}] {speaker}: {message.body}")
    return "\n".join(lines)


def _fallback_summary(target_name: str, messages: list[TimelineMessage]) -> str:
    """Plain, ungenerated rendering used when the LLM summarizer fails.

    A guardian asking for a transcript deep-dive should never get "nothing"
    back just because a subprocess call was flaky - they get the raw recent
    lines instead, capped and labeled as unsummarized so it's clear this is
    the degraded path.
    """
    if not messages:
        return f"{target_name} has no recorded conversation yet."
    recent = messages[-10:]
    lines = [f"(unsummarized - the summarizer was unavailable) Last {len(recent)} turns for {target_name}:"]
    for message in recent:
        speaker = target_name if message.author_kind == "human" else "assistant"
        body = message.body if len(message.body) <= 200 else message.body[:200].rstrip() + "…"
        lines.append(f"[{message.created_at.isoformat()}] {speaker}: {body}")
    return "\n".join(lines)


async def summarize_transcript(
    messages: list[TimelineMessage],
    *,
    target_name: str,
    guardian_config: UserConfig,
    config: Config,
) -> str:
    """Summarize a target's transcript for their guardian via a one-off model call.

    Uses ModelRole.MEMORY_EXTRACTION's "cheap" tier rather than introducing a
    new ModelRole - this is a low-stakes, low-volume summarization task with
    the same cost profile as memory extraction, and adding a registry row for
    a single call site is not warranted. Falls back to _fallback_summary on
    any OneShotError so a flaky subprocess call never leaves the guardian
    with nothing.
    """
    if not messages:
        return _fallback_summary(target_name, messages)

    transcript_text = _format_transcript(target_name, messages)
    backend = guardian_config.backend or config.default_backend
    provider = guardian_config.provider or config.default_provider
    try:
        model = resolve_user_model(
            ModelRole.MEMORY_EXTRACTION,
            guardian_config,
            config,
            backend=backend,
            provider=provider,
        )
        reasoner = ClaudeOneShotReasoner(os_user=guardian_config.os_user)
        result = await reasoner.run(
            prompt=transcript_text,
            system_prompt=_SUMMARY_SYSTEM_PROMPT,
            model=model,
            timeout=_SUMMARY_TIMEOUT_S,
            purpose="guardian_transcript_summary",
        )
    except OneShotError:
        log.exception("Guardian transcript summarization failed for target=%s", target_name)
        return _fallback_summary(target_name, messages)
    text = result.text.strip()
    if not text:
        return _fallback_summary(target_name, messages)
    return text
