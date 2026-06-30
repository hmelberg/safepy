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


class Profile(str, Enum):
    """Which executor runs the code — the single knob separating the two
    security postures. Both share the same gate/runtime/mediator/protect; they
    differ only in what is put in the sandbox namespace.

    OPEN   — real pandas + the raw frame are in scope. Defended by enumeration
             (gate node-whitelist + method deny-list + mediator provenance).
             "Probably safe"; audit surface is all of pandas. For public/local.
    STRICT — only a SafeFrame facade and the safe-verb library are in scope; no
             pandas, no raw frame. Defended by construction: disclosive
             capabilities are not reachable. Audit surface is the small closed
             SafeFrame method list. For protected/sensitive.
    """
    OPEN = "open"
    STRICT = "strict"


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
    profile: Profile          # which executor (OPEN sandbox vs STRICT capability)


def resolve_policy(levels: list[ProtectionLevel | str]) -> Policy:
    """Most-restrictive-source-wins resolution to one policy.

    Profile follows the level: public gets the OPEN sandbox; protected and
    sensitive get the STRICT capability executor. A caller may override the
    profile explicitly (see ``api.run``) for development/testing.
    """
    if not levels:
        levels = [ProtectionLevel.PROTECTED]
    norm = [ProtectionLevel(l) for l in levels]
    level = max(norm, key=lambda l: _ORDER[l])

    if level is ProtectionLevel.PUBLIC:
        return Policy(level, auth_required=False, log=False,
                      min_n=1, round_to=None, profile=Profile.OPEN)
    if level is ProtectionLevel.PROTECTED:
        return Policy(level, auth_required=True, log=True,
                      min_n=5, round_to=None, profile=Profile.OPEN)
    return Policy(level, auth_required=True, log=True,
                  min_n=5, round_to=10, profile=Profile.STRICT)
