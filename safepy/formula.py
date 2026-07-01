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
        if "*" in t:
            # a*b  ->  a + b + a:b  (main effects plus their interaction)
            factors = [_factor(p, columns, base) for p in t.split("*")]
            terms.extend(factors)
            terms.append(":".join(factors))
        elif ":" in t:
            factors = [_factor(p, columns, base) for p in t.split(":")]
            terms.append(":".join(factors))
        else:
            terms.append(_factor(t, columns, base))

    if not terms:
        raise DisclosureError("formula has no predictors")

    canonical_rhs = " + ".join(terms) + ("" if has_intercept else " - 1")
    return outcome, canonical_rhs, sorted(base)


def parse_fixest_formula(formula: str, columns) -> tuple[str, list[str], list[str]]:
    """Parse a pyfixest/fixest formula (with ``|`` sections for fixed effects and
    IV) into ``(canonical_formula, base_columns, fe_columns)``, validating every
    identifier. The canonical formula is rebuilt from validated tokens only, so
    formulaic never sees anything it could evaluate.

    Grammar:  outcome ~ exog [ | fe1 + fe2[^fe3] ] [ | endog ~ instruments ]
    """
    if not isinstance(formula, str) or "~" not in formula:
        raise DisclosureError("formula must be a string containing '~'")
    sections = [s.strip() for s in formula.split("|")]

    lhs, rhs = sections[0].split("~", 1)
    if rhs.strip() in ("", "1"):
        outcome = _validate(lhs, columns)
        rhs_canon, base = "1", {outcome}
    else:
        outcome, rhs_canon, base_list = parse_formula(sections[0], columns)
        base = set(base_list)

    fe_canon = iv_canon = None
    fe_cols: list[str] = []
    for sec in sections[1:]:
        if not sec:
            continue
        if "~" in sec:  # IV part:  endog ~ instruments
            if iv_canon is not None:
                raise DisclosureError("formula has more than one IV part")
            en, ins = sec.split("~", 1)
            endog = [_validate(t, columns) for t in en.split("+")]
            instr = [_validate(t, columns) for t in ins.split("+")]
            base.update(endog); base.update(instr)
            iv_canon = f"{' + '.join(endog)} ~ {' + '.join(instr)}"
        else:  # fixed-effects part
            if fe_canon is not None:
                raise DisclosureError("formula has more than one fixed-effects part")
            facs = []
            for f in sec.split("+"):
                f = f.strip()
                if "^" in f:  # FE interaction
                    cols = [_validate(c, columns) for c in f.split("^")]
                    base.update(cols); fe_cols.extend(cols); facs.append("^".join(cols))
                else:
                    c = _validate(f, columns); base.add(c); fe_cols.append(c); facs.append(c)
            fe_canon = " + ".join(facs)

    canonical = f"{outcome} ~ {rhs_canon}"
    if fe_canon:
        canonical += f" | {fe_canon}"
    if iv_canon:
        canonical += f" | {iv_canon}"
    return canonical, sorted(base), fe_cols
