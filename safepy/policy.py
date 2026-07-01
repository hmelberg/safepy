"""ProtectionLevel -> resolved policy.

Deliberately mirrors the shape settled in m2py's safestat design
(``docs/superpowers/specs/2026-06-29-safestat-remote-compute-slice-design.md``):
one ordered level per source, resolved *most-restrictive-wins* into a single
policy object that drives every downstream behaviour. safepy does not invent
its own knobs; when folded into m2py this module is replaced by m2py's
``resolve_policy`` and the rest of safepy is unchanged.

The one safepy-specific addition is ``sandbox_allowed``: this package is the
*sandbox* executor (the server runs the user's AST-gated Python). Per the
safestat spec, production ``sensitive`` data must use the translate-to-artifact
frontend instead, so ``sandbox_allowed`` is False there. ``public`` is the
intended home of the sandbox; ``protected`` keeps it enabled as the deliberate
*research* configuration this package exists to explore. We encode the boundary
as a value, not a comment.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
class Suppression:
    """The tunable secondary-disclosure controls (the microdata.no "Tiltak").

    Every field is a lever; ``None``/``False`` means the measure is off. Presets
    below bundle them into named tiers, and any single field can be overridden
    per run. The load-bearing primary control (``min_n`` cell suppression) and
    the count rounding base live here too, so there is one config object.
    """
    min_n: int = 5                          # primary cell threshold
    round_to: int | None = None             # rounding base for released counts
    min_population: int | None = None       # Tiltak 1: min analysis population
    winsorize: tuple | None = None          # Tiltak 2: (low, high) percentiles
    count_noise: float | str | None = None  # Tiltak 3: noise on counts (batch 2)
    max_low_cell_share: float | None = None # Tiltak 5: stop table if exceeded
    min_edit_units: int | None = None       # Tiltak 6: min units an edit may touch
    min_descriptive_n: int | None = None    # Tiltak 7: min pop for mean/std/pctile
    percentile_sig_figs: int | None = None  # Tiltak 8: round pctiles to N sig figs
    intercept_k_anon: int | None = None     # Tiltak 9: hide intercept (batch 3)
    microaggregate: bool = False            # Tiltak 10: micro-agg pctiles (batch 3)


# Named aggressiveness tiers. `standard` turns on the measures implemented so far;
# `microdata` mirrors microdata.no (adds the 1000-person floor; noise and
# micro-aggregation are wired in later batches).
PRESETS: dict[str, Suppression] = {
    "off":      Suppression(min_n=1),
    "light":    Suppression(min_n=5, round_to=None),
    "standard": Suppression(
        min_n=5, round_to=10, winsorize=(0.01, 0.99), max_low_cell_share=0.5,
        min_edit_units=10, min_descriptive_n=10, percentile_sig_figs=3,
        intercept_k_anon=5),
    "microdata": Suppression(
        min_n=5, round_to=10, min_population=1000, winsorize=(0.01, 0.99),
        count_noise=2, max_low_cell_share=0.5, min_edit_units=10,
        min_descriptive_n=10, percentile_sig_figs=3, intercept_k_anon=5),
}

# Which tier each protection level gets by default.
_LEVEL_PRESET = {ProtectionLevel.PUBLIC: "off",
                 ProtectionLevel.PROTECTED: "standard",
                 ProtectionLevel.SENSITIVE: "microdata"}


@dataclass(frozen=True)
class Policy:
    level: ProtectionLevel
    auth_required: bool
    log: bool
    min_n: int               # mirror of suppression.min_n (kept for direct access)
    round_to: int | None     # mirror of suppression.round_to
    profile: Profile          # which executor (OPEN sandbox vs STRICT capability)
    suppression: Suppression  # the full secondary-control config


def resolve_policy(levels: list[ProtectionLevel | str], *,
                   suppression: Suppression | str | None = None) -> Policy:
    """Most-restrictive-source-wins resolution to one policy.

    Profile follows the level: public gets the OPEN sandbox; protected and
    sensitive get the STRICT capability executor. ``suppression`` overrides the
    tier — pass a preset name (``"light"``/``"standard"``/``"microdata"``) or a
    :class:`Suppression` instance to tune aggressiveness. A caller may also
    override the profile explicitly (see ``api.run``).
    """
    if not levels:
        levels = [ProtectionLevel.PROTECTED]
    norm = [ProtectionLevel(l) for l in levels]
    level = max(norm, key=lambda l: _ORDER[l])

    if suppression is None:
        supp = PRESETS[_LEVEL_PRESET[level]]
    elif isinstance(suppression, str):
        if suppression not in PRESETS:
            raise ValueError(f"unknown suppression preset: {suppression!r}")
        supp = PRESETS[suppression]
    else:
        supp = suppression

    profile = Profile.OPEN if level in (ProtectionLevel.PUBLIC, ProtectionLevel.PROTECTED) \
        else Profile.STRICT
    auth = level is not ProtectionLevel.PUBLIC
    return Policy(level, auth_required=auth, log=auth, min_n=supp.min_n,
                  round_to=supp.round_to, profile=profile, suppression=supp)
