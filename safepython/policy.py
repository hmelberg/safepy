"""ProtectionLevel -> resolved policy.

Deliberately mirrors the shape settled in m2py's safestat design
(``docs/superpowers/specs/2026-06-29-safestat-remote-compute-slice-design.md``):
one ordered level per source, resolved *most-restrictive-wins* into a single
policy object that drives every downstream behaviour. safepython does not invent
its own knobs; when folded into m2py this module is replaced by m2py's
``resolve_policy`` and the rest of safepython is unchanged.

The one safepython-specific addition is ``sandbox_allowed``: this package is the
*sandbox* executor (the server runs the user's AST-gated Python). Per the
safestat spec, production ``sensitive`` data must use the translate-to-artifact
frontend instead, so ``sandbox_allowed`` is False there. ``public`` is the
intended home of the sandbox; ``protected`` keeps it enabled as the deliberate
*research* configuration this package exists to explore. We encode the boundary
as a value, not a comment.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ProtectionLevel(str, Enum):
    PUBLIC = "public"
    PROTECTED = "protected"
    SENSITIVE = "sensitive"


_ORDER = {ProtectionLevel.PUBLIC: 0,
          ProtectionLevel.PROTECTED: 1,
          ProtectionLevel.SENSITIVE: 2}


@dataclass(frozen=True)
class Policy:
    level: ProtectionLevel
    auth_required: bool
    log: bool
    min_n: int               # primary suppression threshold (cells with n<min_n blanked)
    round_to: int | None     # rounding base for released counts, or None
    sandbox_allowed: bool     # may the server run user Python directly for this level?

    def require_sandbox(self) -> None:
        """Raise if this policy forbids direct execution of user Python."""
        if not self.sandbox_allowed:
            from .errors import DisclosureError
            raise DisclosureError(
                f"protection level '{self.level.value}' forbids direct execution "
                "of user Python; use the translate-to-artifact frontend"
            )


def resolve_policy(levels: list[ProtectionLevel | str]) -> Policy:
    """Most-restrictive-source-wins resolution to one policy."""
    if not levels:
        levels = [ProtectionLevel.PROTECTED]
    norm = [ProtectionLevel(l) for l in levels]
    level = max(norm, key=lambda l: _ORDER[l])

    if level is ProtectionLevel.PUBLIC:
        return Policy(level, auth_required=False, log=False,
                      min_n=1, round_to=None, sandbox_allowed=True)
    if level is ProtectionLevel.PROTECTED:
        return Policy(level, auth_required=True, log=True,
                      min_n=5, round_to=None, sandbox_allowed=True)
    return Policy(level, auth_required=True, log=True,
                  min_n=5, round_to=10, sandbox_allowed=False)
