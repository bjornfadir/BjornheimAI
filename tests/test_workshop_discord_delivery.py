"""Pure-function contracts for kai.workshop.discord_delivery.

Mirrors the error-classification/target-validation coverage style used for
the Telegram delivery adapter (tests/test_workshop_telegram_delivery.py),
scoped to what's cheaply unit-testable without a live discord.py client:
target parsing and error classification. Worker/adapter integration
coverage (claim -> deliver -> settle against a fake outbox) is deferred to
a follow-up - see the Phase 1 report.
"""

import discord
import pytest

from kai.workshop.discord_delivery import (
    DiscordDeliveryContractError,
    _classify_discord_error,
    _discord_fragments,
    _discord_target,
)


class TestDiscordTarget:
    def test_accepts_positive_snowflake(self):
        assert _discord_target("123456789012345678") == 123456789012345678

    def test_rejects_negative(self):
        with pytest.raises(DiscordDeliveryContractError):
            _discord_target("-123")

    def test_rejects_zero(self):
        with pytest.raises(DiscordDeliveryContractError):
            _discord_target("0")

    def test_rejects_non_numeric(self):
        with pytest.raises(DiscordDeliveryContractError):
            _discord_target("@someuser")

    def test_rejects_leading_zero(self):
        with pytest.raises(DiscordDeliveryContractError):
            _discord_target("0123")


class TestDiscordFragments:
    def test_splits_long_body_under_discord_limit(self):
        body = "a" * 5000
        fragments = _discord_fragments(body)
        assert all(len(f) <= 2000 for f in fragments)
        assert "".join(fragments).replace("", "") != ""  # non-empty, sanity

    def test_rejects_empty_body(self):
        with pytest.raises(DiscordDeliveryContractError):
            _discord_fragments("")

    def test_rejects_oversized_body(self):
        with pytest.raises(DiscordDeliveryContractError):
            _discord_fragments("a" * (2000 * 1000 + 1))


class _FakeResponse:
    status = 403
    reason = "Forbidden"


class TestClassifyDiscordError:
    def test_forbidden_is_non_retryable(self):
        failure = _classify_discord_error(discord.Forbidden(_FakeResponse(), "no permission"))
        assert failure.error_code == "discord_forbidden"
        assert failure.retryable is False

    def test_not_found_is_non_retryable(self):
        failure = _classify_discord_error(discord.NotFound(_FakeResponse(), "unknown message"))
        assert failure.error_code == "discord_not_found"
        assert failure.retryable is False

    def test_rate_limited_is_retryable_with_delay(self):
        failure = _classify_discord_error(discord.RateLimited(2.5))
        assert failure.error_code == "discord_rate_limited"
        assert failure.retryable is True
        assert failure.minimum_retry_delay is not None
        assert failure.minimum_retry_delay.total_seconds() >= 2.5

    def test_rate_limited_over_a_day_is_not_retryable(self):
        failure = _classify_discord_error(discord.RateLimited(60 * 60 * 25))
        assert failure.error_code == "discord_rate_limit_too_long"
        assert failure.retryable is False

    def test_unknown_error_is_ambiguous(self):
        failure = _classify_discord_error(RuntimeError("boom"))
        assert failure.error_code == "discord_error_uncertain"
        assert failure.ambiguous is True
