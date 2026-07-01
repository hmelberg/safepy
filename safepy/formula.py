"""A whitelisted parser for statsmodels-style formulas.

This is the security-critical piece of the ``smf`` facade. A patsy formula string
is normally ``eval``-ed by patsy, and the AST gate cannot see inside a string
literal — so we must **never** hand the user's string to patsy. Instead we parse
it ourselves into validated pieces (column names, ``C(col)``, ``a:b``
interactions) and reconstruct a *canonical* formula built only from real column
names and a fixed set of operators. Patsy then sees our reconstruction, in which
every token is a validated identifier — it cannot evaluate anything arbitrary.

Supported grammar (deliberately small):

    outcome ~ term ( + term )*            and an optional  - 1 / 0  to drop intercept
    term := col | C(col) | col:col | C(col):col | ...

Transforms (``np.log(x)``, ``I(x**2)``) are intentionally unsupported — do the
transform first with ``assign``. Anything outside the grammar raises
``DisclosureError``.
"""

from __future__ import annotations

import re

from .errors import DisclosureError

_IDENT = re.compile(r"^[A-Za-z_]\w*$")
_C = re.compile(r"^C\(\s*(\w+)\s*\)$")


def _validate(name: str, columns) -> str:
    name = name.strip()
    if not _IDENT.match(name) or name not in columns:
        raise DisclosureError(f"invalid or unknown column in formula: {name!r}")
    return name


def _factor(token: str, columns, base: set) -> str:
    """A single factor: ``col`` or ``C(col)``. Returns its canonical text."""
    m = _C.match(token.strip())
    if m:
        col = _validate(m.group(1), columns)
        base.add(col)
        return f"C({col})"
    col = _validate(token, columns)
    base.add(col)
    return col


def parse_formula(formula: str, columns) -> tuple[str, str, list[str]]:
    """Return ``(outcome, canonical_rhs, base_columns)`` or raise DisclosureError.

    ``canonical_rhs`` is rebuilt from validated tokens and is the *only* thing
    that should be handed to statsmodels.
    """
    if not isinstance(formula, str) or formula.count("~") != 1:
        raise DisclosureError("formula must be a string with exactly one '~'")
    lhs, rhs = formula.split("~")
    outcome = _validate(lhs, columns)

    base: set = {outcome}
    has_intercept = True

    # handle an explicit intercept drop (- 1) anywhere on the rhs
    m = re.search(r"-\s*1\b", rhs)
    if m:
        has_intercept = False
        rhs = rhs[:m.start()] + rhs[m.end():]

    terms: list[str] = []
    for raw in rhs.split("+"):
        t = raw.strip()
        if t == "" or t == "1":
            continue
        if t in ("0",):
            has_intercept = False
            continue
        if ":" in t:
            factors = [_factor(p, columns, base) for p in t.split(":")]
            terms.append(":".join(factors))
        else:
            terms.append(_factor(t, columns, base))

    if not terms:
        raise DisclosureError("formula has no predictors")

    canonical_rhs = " + ".join(terms) + ("" if has_intercept else " - 1")
    return outcome, canonical_rhs, sorted(base)
