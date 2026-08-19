"""Contracts for the safety-flagging classifier's cost model and parsing."""

from __future__ import annotations

import pytest

from kai.config import Config
from kai.oneshot import OneShotResult, OneShotSubprocessError, OneShotTimeout
from kai.workshop import safety_classifier
from kai.workshop.safety_classifier import SafetyCategory, SafetyClassification, classify

_CONFIG = Config(telegram_bot_token="test-token", allowed_user_ids={12345}, default_backend="claude")


class _CountingReasoner:
    """Counts .run() calls so tests can prove the keyword pre-filter,
    not the LLM, gates the common (clean-message) case."""

    calls = 0

    def __init__(self, result: OneShotResult | None = None, error: Exception | None = None):
        self._result = result
        self._error = error

    async def run(self, **kwargs):
        type(self).calls += 1
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


def _envelope(flagged: bool, category: str | None = None, summary: str | None = None) -> str:
    import json

    structured: dict = {"flagged": flagged}
    if category is not None:
        structured["category"] = category
    if summary is not None:
        structured["summary"] = summary
    return json.dumps({"is_error": False, "structured_output": structured})


def _result(text: str) -> OneShotResult:
    return OneShotResult(text=text, backend="claude", model="claude-haiku-4-5-20251001")


@pytest.fixture(autouse=True)
def _reset_call_counter():
    _CountingReasoner.calls = 0
    yield


class TestSafetyClassificationInvariants:
    def test_flagged_without_category_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="category"):
            SafetyClassification(flagged=True)

    def test_unflagged_with_category_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unflagged"):
            SafetyClassification(flagged=False, category=SafetyCategory.SELF_HARM)

    def test_unflagged_default_is_valid(self) -> None:
        result = SafetyClassification(flagged=False)
        assert result.category is None
        assert result.summary is None


class TestCostModel:
    async def test_clean_exchange_never_calls_the_reasoner(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reasoner = _CountingReasoner()
        monkeypatch.setattr(safety_classifier, "ClaudeOneShotReasoner", lambda: reasoner)

        result = await classify(
            "what's the capital of France?",
            "Paris is the capital of France.",
            _CONFIG,
            user_config=None,
        )

        assert result == SafetyClassification(flagged=False)
        assert reasoner.calls == 0

    async def test_prefilter_hit_escalates_to_exactly_one_reasoner_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reasoner = _CountingReasoner(result=_result(_envelope(False)))
        monkeypatch.setattr(safety_classifier, "ClaudeOneShotReasoner", lambda: reasoner)

        await classify(
            "I want to kill myself",
            "I'm sorry you're feeling this way.",
            _CONFIG,
            user_config=None,
        )

        assert reasoner.calls == 1


class TestLLMEscalation:
    async def test_llm_confirms_flag_returns_category_and_summary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reasoner = _CountingReasoner(
            result=_result(_envelope(True, category="self_harm", summary="expressed suicidal thoughts"))
        )
        monkeypatch.setattr(safety_classifier, "ClaudeOneShotReasoner", lambda: reasoner)

        result = await classify("I want to kill myself", "assistant reply", _CONFIG, user_config=None)

        assert result.flagged is True
        assert result.category == SafetyCategory.SELF_HARM
        assert result.summary == "expressed suicidal thoughts"

    async def test_llm_overrides_prefilter_with_not_flagged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Prefilter is intentionally high-recall/low-precision - the LLM
        # makes the real call. A prose mention that trips the keyword
        # regex but isn't a real concern must not produce an alert.
        reasoner = _CountingReasoner(result=_result(_envelope(False)))
        monkeypatch.setattr(safety_classifier, "ClaudeOneShotReasoner", lambda: reasoner)

        result = await classify(
            "we're reading a book where a character talks about wanting to die",
            "That sounds like a heavy topic for a book.",
            _CONFIG,
            user_config=None,
        )

        assert result == SafetyClassification(flagged=False)

    @pytest.mark.parametrize(
        "error",
        [OneShotTimeout(), OneShotSubprocessError(returncode=1, stderr=b"boom")],
    )
    async def test_reasoner_failure_never_raises_and_collapses_to_not_flagged(
        self, monkeypatch: pytest.MonkeyPatch, error: Exception
    ) -> None:
        reasoner = _CountingReasoner(error=error)
        monkeypatch.setattr(safety_classifier, "ClaudeOneShotReasoner", lambda: reasoner)

        result = await classify("I want to kill myself", "assistant reply", _CONFIG, user_config=None)

        assert result == SafetyClassification(flagged=False)

    async def test_invalid_json_collapses_to_not_flagged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reasoner = _CountingReasoner(result=_result("not json"))
        monkeypatch.setattr(safety_classifier, "ClaudeOneShotReasoner", lambda: reasoner)

        result = await classify("I want to kill myself", "assistant reply", _CONFIG, user_config=None)

        assert result == SafetyClassification(flagged=False)

    async def test_invalid_category_collapses_to_not_flagged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reasoner = _CountingReasoner(result=_result(_envelope(True, category="not_a_real_category")))
        monkeypatch.setattr(safety_classifier, "ClaudeOneShotReasoner", lambda: reasoner)

        result = await classify("I want to kill myself", "assistant reply", _CONFIG, user_config=None)

        assert result == SafetyClassification(flagged=False)

    async def test_flagged_without_summary_gets_a_placeholder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reasoner = _CountingReasoner(result=_result(_envelope(True, category="self_harm")))
        monkeypatch.setattr(safety_classifier, "ClaudeOneShotReasoner", lambda: reasoner)

        result = await classify("I want to kill myself", "assistant reply", _CONFIG, user_config=None)

        assert result.flagged is True
        assert result.summary
