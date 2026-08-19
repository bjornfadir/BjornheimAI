"""Safety-flagging classifier for guardian-monitored users' turns.

Scoped narrowly: this module answers one question - "does this exchange
need a guardian's attention?" - for users whose `UserConfig.monitored` is
True. It does not run for anyone else and never blocks or alters a turn;
callers invoke it after a turn is already durably recorded and delivered
(see private_text_execution.py's on_completed hook), so a slow or failed
classification can never add latency or failure risk to the user's own
response.

Cost model mirrors memory_extraction.py's Track 2 design: a free, local
keyword pre-filter runs on every monitored-user turn, and only an
ambiguous match escalates to one cheap `claude --print` subprocess call
(kai.oneshot.ClaudeOneShotReasoner, ModelRole.SAFETY_CLASSIFICATION - the
same "cheap" tier as memory extraction). A clean turn costs zero
subprocess calls.

DISCLOSED LIMITATION, not just a code comment: this is a keyword filter
plus one cheap LLM call. It will miss things - coded language, context
spread across multiple turns, anything the prompt does not anticipate.
It is a supplementary signal, not a substitute for actual parental
attention, and must never be presented to a guardian as reliable
coverage. See the false-negative risk note in this session's Phase 3
report.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import StrEnum

from kai.config import Config, ModelRole, UserConfig, resolve_user_model
from kai.oneshot import ClaudeOneShotReasoner, OneShotError, OneShotSubprocessError, OneShotTimeout

log = logging.getLogger(__name__)

_CLASSIFIER_TIMEOUT_S = 10.0


class SafetyCategory(StrEnum):
    """Flagged-content categories a guardian alert can carry.

    Deliberately small and coarse - this is a routing signal for a
    guardian alert, not a content-moderation taxonomy. Each value names
    what to tell the guardian, not a precise clinical or legal category.
    """

    SELF_HARM = "self_harm"
    SEXUAL_CONTENT_MINOR = "sexual_content_minor"
    DANGEROUS_ACTIVITY = "dangerous_activity"
    HARASSMENT = "harassment"


@dataclass(frozen=True, slots=True)
class SafetyClassification:
    """Result of classifying one exchange. flagged=False is the default."""

    flagged: bool
    category: SafetyCategory | None = None
    summary: str | None = None

    def __post_init__(self) -> None:
        if self.flagged and self.category is None:
            raise ValueError("a flagged classification must carry a category")
        if not self.flagged and (self.category is not None or self.summary is not None):
            raise ValueError("an unflagged classification must not carry category or summary")


_NOT_FLAGGED = SafetyClassification(flagged=False)

# Cheap, local pre-filter. Deliberately broad/high-recall - false
# positives here only cost one subprocess call (via the LLM escalation
# below), while a false negative here means the LLM never even sees the
# turn. Matches are case-insensitive; each arm targets one category so a
# hit can seed the LLM prompt with which category to focus on, though the
# LLM makes the final call rather than trusting the keyword match.
_PREFILTER_PATTERNS: dict[SafetyCategory, re.Pattern[str]] = {
    SafetyCategory.SELF_HARM: re.compile(
        r"\b(kill myself|suicide|suicidal|self.?harm|cutting myself|"
        r"want to die|end my life|hurt myself|overdose)\b",
        re.IGNORECASE,
    ),
    SafetyCategory.SEXUAL_CONTENT_MINOR: re.compile(
        r"\b(nudes?|sext|sexting|naked pics?|send pics?)\b",
        re.IGNORECASE,
    ),
    SafetyCategory.DANGEROUS_ACTIVITY: re.compile(
        r"\b(how to make a bomb|make a weapon|buy a gun|get drugs|"
        r"how to hack (into|someone)|meet up alone with|runaway|run away from home)\b",
        re.IGNORECASE,
    ),
    SafetyCategory.HARASSMENT: re.compile(
        r"\b(everyone hates me|being bullied|they keep threatening|"
        r"kill you|going to hurt you)\b",
        re.IGNORECASE,
    ),
}

_CLASSIFIER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "flagged": {"type": "boolean"},
        "category": {
            "type": "string",
            "enum": [c.value for c in SafetyCategory],
        },
        "summary": {
            "type": "string",
            "minLength": 1,
            "maxLength": 200,
        },
    },
    "required": ["flagged"],
    "additionalProperties": False,
}

_CLASSIFIER_SYSTEM_PROMPT = """You are a safety-flagging classifier for Bjornheim AI, a personal AI \
agent used by a family. You review one exchange between a monitored (child) user and the \
assistant and decide whether a guardian should be alerted.

Flag ONLY when the exchange shows a real, current safety concern in one of these categories:
- self_harm: the user expresses suicidal ideation, self-harm intent, or an immediate crisis.
- sexual_content_minor: sexual content involving the user, or signs of grooming by a third party.
- dangerous_activity: the user is seeking help with something physically dangerous or illegal \
(weapons, drugs, meeting a stranger alone, running away).
- harassment: the user describes being bullied, threatened, or harassed in a way that suggests \
ongoing harm.

Do NOT flag: normal curiosity, schoolwork, jokes, fiction/creative writing, or venting that does \
not indicate real risk. Err toward NOT flagging casual or ambiguous language - the pre-filter \
that routed this exchange to you is intentionally broad, so most exchanges you see should NOT \
be flagged.

If flagged, summary must be a short, neutral, non-quoting description of the concern (e.g. \
"expressed suicidal thoughts", "asked how to obtain a weapon") - never a direct quote of the \
user's words, and never include names or identifying details beyond what a guardian needs to \
know to check in.

Return JSON matching the schema. If not flagged, omit category and summary."""


def _prefilter_hit(user_text: str, assistant_text: str) -> SafetyCategory | None:
    combined = f"{user_text}\n{assistant_text}"
    for category, pattern in _PREFILTER_PATTERNS.items():
        if pattern.search(combined):
            return category
    return None


def _build_payload(user_text: str, assistant_text: str) -> str:
    return f"USER: {user_text}\n\nASSISTANT: {assistant_text}"


async def classify(
    user_text: str,
    assistant_text: str,
    config: Config,
    *,
    user_config: UserConfig | None,
) -> SafetyClassification:
    """Classify one exchange for a monitored user. Never raises.

    Every failure mode (prefilter miss, LLM timeout, subprocess error,
    invalid JSON, invalid schema shape) collapses to `_NOT_FLAGGED` -
    the caller (the on_completed hook) must never have a monitored
    user's turn fail or delay because this classifier broke. That also
    means classifier failures are silent false negatives; see the
    module docstring's disclosed limitation.
    """
    prefilter_category = _prefilter_hit(user_text, assistant_text)
    if prefilter_category is None:
        return _NOT_FLAGGED

    reasoner = ClaudeOneShotReasoner()
    try:
        result = await reasoner.run(
            prompt=_build_payload(user_text, assistant_text),
            system_prompt=_CLASSIFIER_SYSTEM_PROMPT,
            model=resolve_user_model(
                ModelRole.SAFETY_CLASSIFICATION,
                user_config,
                config,
                backend="claude",
            ),
            timeout=_CLASSIFIER_TIMEOUT_S,
            purpose="safety_classification",
            json_schema=_CLASSIFIER_SCHEMA,
        )
    except OneShotTimeout:
        log.warning("Safety classification timed out after %ss", _CLASSIFIER_TIMEOUT_S)
        return _NOT_FLAGGED
    except OneShotSubprocessError as e:
        log.warning(
            "Safety classification subprocess exited %d: %s",
            e.returncode,
            e.stderr[:500].decode("utf-8", errors="replace"),
        )
        return _NOT_FLAGGED
    except OneShotError:
        log.warning("Safety classification reasoner error", exc_info=True)
        return _NOT_FLAGGED

    try:
        parsed = json.loads(result.text)
    except json.JSONDecodeError:
        log.warning("Safety classification produced invalid JSON: %r", result.text[:500])
        return _NOT_FLAGGED
    if not isinstance(parsed, dict):
        log.warning("Safety classification returned non-object JSON: %r", parsed)
        return _NOT_FLAGGED
    if parsed.get("is_error") is True:
        log.warning("Safety classification CLI envelope reports is_error=true")
        return _NOT_FLAGGED
    # Mirrors memory_extraction.py's nested/root resolution: schema-validated
    # payloads land under `structured_output`, but fall back to the root for
    # resilience against a future CLI shape change or a hand-rolled mock.
    nested = parsed.get("structured_output")
    structured = nested if isinstance(nested, dict) and "flagged" in nested else parsed

    flagged = structured.get("flagged")
    if not isinstance(flagged, bool):
        log.warning("Safety classification missing/invalid 'flagged' field: %r", structured)
        return _NOT_FLAGGED
    if not flagged:
        return _NOT_FLAGGED

    category_raw = structured.get("category")
    try:
        category = SafetyCategory(category_raw)
    except ValueError:
        log.warning("Safety classification flagged with invalid category: %r", category_raw)
        return _NOT_FLAGGED
    summary_raw = structured.get("summary")
    summary = summary_raw.strip() if isinstance(summary_raw, str) and summary_raw.strip() else "flagged, no summary provided"

    return SafetyClassification(flagged=True, category=category, summary=summary)
