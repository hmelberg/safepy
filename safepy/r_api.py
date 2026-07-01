"""The R dialect (STRICT) — a *translated* safe surface, never executed.

R is a different language, so it cannot ride the Python AST gate. Instead this
module **parses** a restricted dplyr/base-R surface and **translates** it to the
same backend-neutral release core (:class:`safepy.safe.SafeVerbs`) the pandas and
polars dialects use. User R is never `eval`/`source`-ed — there is no system,
file, or code-execution surface — so the dialect is safe by construction; the
parser is default-deny (only whitelisted verbs/functions are recognised).

First slice (dplyr pipe): ``df |> group_by(g) |> summarise(m = fn(x))`` and
``df |> count(g)``, with an optional leading ``filter(x OP v)``. The ``|>`` and
``%>%`` pipes are both accepted.
"""

from __future__ import annotations

import re

from .errors import DisclosureError, ValidationError
from .result import Released

# dplyr/base-R aggregation function -> safepy agg name.
_AGG_MAP = {
    "mean": "mean", "sum": "sum", "median": "median",
    "sd": "std", "var": "var", "n": "size",
}
_CMP = {
    ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b, "<": lambda a, b: a < b,
    "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
}
_IDENT = re.compile(r"^[A-Za-z_.][\w.]*$")
_STAGE = re.compile(r"^([A-Za-z_][\w.]*)\s*\((.*)\)\s*$", re.S)


def _split_top(s: str, sep_tokens) -> list[str]:
    """Split ``s`` on any token in ``sep_tokens`` that appears at bracket depth 0
    (so separators inside ``(...)`` / strings are ignored). ``sep_tokens`` is a
    list of literal strings tried longest-first."""
    out, depth, cur, i, quote = [], 0, [], 0, None
    toks = sorted(sep_tokens, key=len, reverse=True)
    while i < len(s):
        c = s[i]
        if quote:
            cur.append(c)
            if c == quote:
                quote = None
            i += 1
            continue
        if c in "'\"":
            quote = c; cur.append(c); i += 1; continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if depth == 0:
            hit = next((t for t in toks if s.startswith(t, i)), None)
            if hit:
                out.append("".join(cur)); cur = []; i += len(hit); continue
        cur.append(c); i += 1
    out.append("".join(cur))
    return [p.strip() for p in out]


def _need_col(df, col):
    if col not in df.columns:
        raise DisclosureError(f"unknown column: {col}")
    return col


def _parse_stage(stage: str):
    m = _STAGE.match(stage)
    if not m:
        raise ValidationError(f"could not parse R stage: {stage!r}", kind="syntax")
    return m.group(1), m.group(2).strip()


def _cols(argstr: str) -> list[str]:
    cols = [a.strip() for a in _split_top(argstr, [","]) if a.strip()]
    for c in cols:
        if not _IDENT.match(c):
            raise ValidationError(f"expected a column name, got {c!r}", kind="syntax")
    return cols


def _literal(tok: str):
    tok = tok.strip()
    if (tok[:1], tok[-1:]) in (("'", "'"), ('"', '"')):
        return tok[1:-1]
    try:
        return int(tok)
    except ValueError:
        try:
            return float(tok)
        except ValueError:
            raise ValidationError(f"expected a number or quoted string, got {tok!r}",
                                  kind="syntax")


def _apply_filter(df, argstr: str):
    """Translate a single ``filter(col OP value)`` predicate to a pandas mask.
    The filtered frame stays private — it exits only via the terminal summary."""
    for op in ("==", "!=", ">=", "<=", ">", "<"):
        parts = _split_top(argstr, [op])
        if len(parts) == 2:
            col, val = parts[0].strip(), _literal(parts[1])
            _need_col(df, col)
            return df[_CMP[op](df[col], val)]
    raise ValidationError(f"could not parse filter predicate: {argstr!r}", kind="syntax")


def _summarise(verbs, df, group, argstr: str) -> Released:
    if group is None:
        raise DisclosureError(
            "summarise needs a preceding group_by(...) in this first slice")
    pairs = _split_top(argstr, [","])
    if len(pairs) != 1:
        raise DisclosureError(
            "this first slice supports a single summary, e.g. summarise(m = mean(x))")
    m = re.match(r"^(\w+)\s*=\s*(\w+)\s*\(\s*([\w.]*)\s*\)$", pairs[0].strip())
    if not m:
        raise ValidationError(
            f"summary must be name = fn(col), got {pairs[0]!r}", kind="syntax")
    _name, fn, col = m.group(1), m.group(2), m.group(3)
    if fn not in _AGG_MAP:
        raise DisclosureError(
            f"aggregation '{fn}' is not allowed; choose one of {sorted(_AGG_MAP)}")
    agg = _AGG_MAP[fn]
    value = _need_col(df, col) if col else (group[0] if fn == "n" else None)
    by = group[0] if len(group) == 1 else group
    return verbs.group_agg(df, by, value, agg)


def _count(verbs, df, argstr: str) -> Released:
    cols = _cols(argstr)
    if len(cols) != 1:
        raise DisclosureError("count(col) supports a single column in this first slice")
    _need_col(df, cols[0])
    return verbs.value_counts(df, cols[0])


_MODEL_CALL = re.compile(r"^(lm|glm)\s*\((.*)\)\s*$", re.S)


def _resolve_df(name, sources):
    if name not in sources:
        raise ValidationError(f"unknown data source: {name!r}", kind="name")
    df = sources[name]
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def _model(verbs, code: str, sources: dict) -> Released:
    """``lm(y ~ x1 + x2, data=df)`` / ``glm(y ~ x, family=binomial, data=df)`` ->
    the shared regression verbs (ols/logit/poisson), reusing the same
    per-coefficient suppression as the pandas/polars dialects."""
    m = _MODEL_CALL.match(code)
    kind, argstr = m.group(1), m.group(2).strip()
    formula = data = family = None
    for a in _split_top(argstr, [","]):
        a = a.strip()
        key = a.split("=", 1)[0].strip() if "=" in a else ""
        if key in ("data", "family", "weights", "subset", "na.action"):
            val = a.split("=", 1)[1].strip()
            if key == "data":
                data = val
            elif key == "family":
                family = val
        elif "~" in a:
            formula = a
    if formula is None or data is None:
        raise ValidationError(f"{kind}() needs a formula and data=", kind="syntax")
    df = _resolve_df(data, sources)
    lhs, rhs = formula.split("~", 1)
    y = lhs.strip()
    xs = [t.strip() for t in _split_top(rhs, ["+"])
          if t.strip() and t.strip() not in ("1", "0", ".")]
    if not _IDENT.match(y) or not all(_IDENT.match(x) for x in xs):
        raise ValidationError("formula terms must be column names", kind="syntax")
    if not xs:
        raise DisclosureError("model needs at least one predictor")
    if kind == "lm":
        return verbs.ols(df, y=y, x=xs)
    fam = (family or "").lower()
    if "binomial" in fam:
        return verbs.logit(df, y=y, x=xs)
    if "poisson" in fam:
        return verbs.poisson(df, y=y, x=xs)
    raise DisclosureError("glm supports family = binomial (logit) or poisson")


def translate_r(code: str, verbs, sources: dict) -> Released:
    """Parse a restricted R pipeline and return a suppressed ``Released``."""
    code = code.strip()
    if not code:
        raise ValidationError("empty program", kind="empty")
    if _MODEL_CALL.match(code) and code.split("(", 1)[0].strip() in ("lm", "glm"):
        return _model(verbs, code, sources)
    stages = _split_top(code, ["|>", "%>%"])
    src = stages[0].strip()
    if "<-" in src:                      # `result <- df |> ...`: take the data name
        src = src.split("<-", 1)[1].strip()
    if src not in sources:
        raise ValidationError(f"unknown data source: {src!r}", kind="name")
    df = sources[src]
    if hasattr(df, "to_pandas"):         # accept a polars frame too
        df = df.to_pandas()

    group = None
    for stage in stages[1:]:
        verb, argstr = _parse_stage(stage)
        if verb == "filter":
            df = _apply_filter(df, argstr)
        elif verb == "group_by":
            group = _cols(argstr)
            for c in group:
                _need_col(df, c)
        elif verb in ("summarise", "summarize"):
            return _summarise(verbs, df, group, argstr)
        elif verb == "count":
            return _count(verbs, df, argstr)
        else:
            raise DisclosureError(
                f"R verb '{verb}' is not supported in this first slice "
                "(group_by, summarise, count, filter)")
    raise DisclosureError(
        "R pipeline did not end in a releasable summary (summarise/count)")
