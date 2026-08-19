"""Config-driven guardian authorization for reading another user's data.

Bjornheim AI's canonical principal/channel authorization (authorization.py,
client_access.py) is strictly self-scoped by design - no principal can read
another principal's data through it. Guardian monitoring is a deliberate,
narrow exception to that: an admin-configured user (a parent) may read a
specific other configured user's (a kid's) data, never a general ACL system.
Authorization here is entirely static, from users.yaml's guardian_of field -
there is no database-backed grant/revoke flow, matching how role == "admin"
already works for the one other non-self check in this codebase (see
bot.py's "/project unregister" handler).

This module only resolves identity - which principal_id/channel_id a
guardian and their target resolve to, per transport. It does not read
history or memory content; that belongs to the callers who actually need
it (the safety-flagging pipeline, the on-demand transcript deep-dive).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from kai.config import UserConfig
from kai.workshop.bootstrap import bootstrap_human_channel_id, bootstrap_human_principal_id
from kai.workshop.domain import ChannelId, PrincipalId, WorkshopId


class GuardianAccessError(PermissionError):
    """A guardian-access request did not resolve safely."""


@dataclass(frozen=True, slots=True)
class GuardianIdentity:
    """One transport identity resolved to its principal and direct channel."""

    transport: str
    principal_id: PrincipalId
    channel_id: ChannelId


@dataclass(frozen=True, slots=True)
class GuardianTarget:
    """A monitored user resolved for one specific, already-authorized guardian.

    identities covers every transport the target has configured
    (telegram_id and/or discord_id) - a user with both isn't guaranteed to
    have every turn in one place, so callers that need "everything this
    user said" must consult each identity's channel, not just the first.
    guardian_identities is the calling guardian's own resolved identities,
    for delivering an alert or response back to them.
    """

    name: str
    monitored: bool
    identities: tuple[GuardianIdentity, ...]
    guardian_identities: tuple[GuardianIdentity, ...]


def _resolve_identities(workshop_id: WorkshopId, user: UserConfig) -> tuple[GuardianIdentity, ...]:
    identities: list[GuardianIdentity] = []
    for transport, external_subject in (
        ("telegram", str(user.telegram_id) if user.telegram_id is not None else None),
        ("discord", str(user.discord_id) if user.discord_id is not None else None),
    ):
        if external_subject is None:
            continue
        identities.append(
            GuardianIdentity(
                transport=transport,
                principal_id=bootstrap_human_principal_id(workshop_id, transport, external_subject),
                channel_id=bootstrap_human_channel_id(workshop_id, transport, external_subject),
            )
        )
    return tuple(identities)


def _find_user_by_name(user_configs: Iterable[UserConfig], target_name: str) -> UserConfig | None:
    for user_config in user_configs:
        if user_config.name == target_name:
            return user_config
    return None


def resolve_guardian_target(
    workshop_id: WorkshopId,
    user_configs: Iterable[UserConfig],
    guardian_config: UserConfig,
    target_name: str,
) -> GuardianTarget:
    """Resolve target_name's identity for guardian_config, or raise.

    user_configs is every configured user (e.g. Config.user_configs.values())
    - used only to look up target_name by UserConfig.name, since guardian_of
    stores names, not the config_id keys Config.user_configs is keyed by.

    Raises GuardianAccessError if guardian_config.guardian_of does not
    list target_name, if no such user is configured, or if the resolved
    target has no transport identity to read at all. Denial and
    not-found are deliberately not distinguished in the exception type -
    callers must not leak which of "not your ward" vs. "doesn't exist"
    applies, the same fail-closed shape client_access.py's
    _require_direct_human uses.
    """
    if not target_name or target_name not in guardian_config.guardian_of:
        raise GuardianAccessError(f"{guardian_config.name!r} is not a configured guardian of {target_name!r}")
    target_config = _find_user_by_name(user_configs, target_name)
    if target_config is None:
        raise GuardianAccessError(f"{guardian_config.name!r} is not a configured guardian of {target_name!r}")
    identities = _resolve_identities(workshop_id, target_config)
    if not identities:
        raise GuardianAccessError(f"{target_name!r} has no transport identity configured to read")
    guardian_identities = _resolve_identities(workshop_id, guardian_config)
    return GuardianTarget(
        name=target_config.name,
        monitored=target_config.monitored,
        identities=identities,
        guardian_identities=guardian_identities,
    )


def resolve_guardians_of(
    workshop_id: WorkshopId,
    user_configs: Iterable[UserConfig],
    target_config: UserConfig,
) -> tuple[GuardianTarget, ...]:
    """Resolve every configured guardian of target_config, for alert fan-out.

    Returns one GuardianTarget per guardian whose guardian_of lists
    target_config.name - empty if none are configured, which is a normal,
    non-error state (see UserConfig.monitored's docstring): a monitored
    user with no guardian configured simply has no one to alert.
    """
    user_configs = tuple(user_configs)
    resolved: list[GuardianTarget] = []
    for candidate in user_configs:
        if target_config.name in candidate.guardian_of:
            resolved.append(resolve_guardian_target(workshop_id, user_configs, candidate, target_config.name))
    return tuple(resolved)
