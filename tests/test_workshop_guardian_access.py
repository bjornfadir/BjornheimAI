"""Tests for the config-driven guardian authorization primitive."""

from __future__ import annotations

import pytest

from kai.config import UserConfig
from kai.workshop.domain import WorkshopId
from kai.workshop.guardian_access import (
    GuardianAccessError,
    resolve_guardian_target,
    resolve_guardians_of,
)

WORKSHOP_ID = WorkshopId.new()


def _users(*configs: UserConfig) -> tuple[UserConfig, ...]:
    return configs


class TestResolveGuardianTarget:
    def test_guardian_can_resolve_a_listed_monitored_user(self) -> None:
        guardian = UserConfig(name="parent", telegram_id=1, guardian_of=["kid"])
        kid = UserConfig(name="kid", discord_id=2, monitored=True)
        users = _users(guardian, kid)

        target = resolve_guardian_target(WORKSHOP_ID, users, guardian, "kid")

        assert target.name == "kid"
        assert target.monitored is True
        assert len(target.identities) == 1
        assert target.identities[0].transport == "discord"
        assert len(target.guardian_identities) == 1
        assert target.guardian_identities[0].transport == "telegram"

    def test_resolution_is_deterministic(self) -> None:
        guardian = UserConfig(name="parent", telegram_id=1, guardian_of=["kid"])
        kid = UserConfig(name="kid", discord_id=2, monitored=True)
        users = _users(guardian, kid)

        first = resolve_guardian_target(WORKSHOP_ID, users, guardian, "kid")
        second = resolve_guardian_target(WORKSHOP_ID, users, guardian, "kid")

        assert first.identities[0].principal_id == second.identities[0].principal_id
        assert first.identities[0].channel_id == second.identities[0].channel_id

    def test_user_with_both_transports_resolves_both_identities(self) -> None:
        guardian = UserConfig(name="parent", telegram_id=1, guardian_of=["kid"])
        kid = UserConfig(name="kid", telegram_id=3, discord_id=2, monitored=True)
        users = _users(guardian, kid)

        target = resolve_guardian_target(WORKSHOP_ID, users, guardian, "kid")

        transports = {identity.transport for identity in target.identities}
        assert transports == {"telegram", "discord"}

    def test_guardian_is_refused_for_a_user_not_in_guardian_of(self) -> None:
        guardian = UserConfig(name="parent", telegram_id=1, guardian_of=["kid"])
        other = UserConfig(name="stranger", discord_id=9, monitored=True)
        users = _users(guardian, other)

        with pytest.raises(GuardianAccessError):
            resolve_guardian_target(WORKSHOP_ID, users, guardian, "stranger")

    def test_non_guardian_with_no_guardian_of_gets_nothing(self) -> None:
        non_guardian = UserConfig(name="parent", telegram_id=1)
        kid = UserConfig(name="kid", discord_id=2, monitored=True)
        users = _users(non_guardian, kid)

        with pytest.raises(GuardianAccessError):
            resolve_guardian_target(WORKSHOP_ID, users, non_guardian, "kid")

    def test_guardian_of_referencing_unconfigured_user_is_refused(self) -> None:
        guardian = UserConfig(name="parent", telegram_id=1, guardian_of=["ghost"])
        users = _users(guardian)

        with pytest.raises(GuardianAccessError):
            resolve_guardian_target(WORKSHOP_ID, users, guardian, "ghost")

    def test_target_with_no_transport_identity_is_refused(self) -> None:
        # Not constructible via normal users.yaml loading (config.py
        # requires at least one of telegram_id/discord_id), but
        # guardian_access.py must still fail closed rather than return
        # a target with an empty identities tuple if it ever happens.
        guardian = UserConfig(name="parent", telegram_id=1, guardian_of=["ghost"])
        ghost = UserConfig(name="ghost", monitored=True)
        users = _users(guardian, ghost)

        with pytest.raises(GuardianAccessError):
            resolve_guardian_target(WORKSHOP_ID, users, guardian, "ghost")

    def test_empty_target_name_is_refused(self) -> None:
        guardian = UserConfig(name="parent", telegram_id=1, guardian_of=["kid"])
        users = _users(guardian)

        with pytest.raises(GuardianAccessError):
            resolve_guardian_target(WORKSHOP_ID, users, guardian, "")


class TestResolveGuardiansOf:
    def test_monitored_user_with_no_guardian_configured_returns_empty(self) -> None:
        kid = UserConfig(name="kid", discord_id=2, monitored=True)
        users = _users(kid)

        guardians = resolve_guardians_of(WORKSHOP_ID, users, kid)

        assert guardians == ()

    def test_monitored_user_with_one_guardian_resolves_it(self) -> None:
        guardian = UserConfig(name="parent", telegram_id=1, guardian_of=["kid"])
        kid = UserConfig(name="kid", discord_id=2, monitored=True)
        users = _users(guardian, kid)

        guardians = resolve_guardians_of(WORKSHOP_ID, users, kid)

        assert len(guardians) == 1
        assert guardians[0].name == "kid"
        assert guardians[0].guardian_identities[0].transport == "telegram"

    def test_monitored_user_with_two_guardians_resolves_both(self) -> None:
        parent_a = UserConfig(name="parent_a", telegram_id=1, guardian_of=["kid"])
        parent_b = UserConfig(name="parent_b", discord_id=4, guardian_of=["kid"])
        kid = UserConfig(name="kid", discord_id=2, monitored=True)
        users = _users(parent_a, parent_b, kid)

        guardians = resolve_guardians_of(WORKSHOP_ID, users, kid)

        assert len(guardians) == 2

    def test_unmonitored_user_can_still_have_a_guardian_resolved(self) -> None:
        # monitored is a separate knob from guardian_of (see UserConfig's
        # docstring) - resolve_guardians_of doesn't gate on it, the
        # safety-flagging pipeline is expected to check .monitored itself
        # before calling this.
        guardian = UserConfig(name="parent", telegram_id=1, guardian_of=["kid"])
        kid = UserConfig(name="kid", discord_id=2, monitored=False)
        users = _users(guardian, kid)

        guardians = resolve_guardians_of(WORKSHOP_ID, users, kid)

        assert len(guardians) == 1
