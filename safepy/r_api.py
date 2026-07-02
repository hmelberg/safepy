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


def _filter(sf, argstr: str):
    """``filter(<expr>)`` — evaluate the (possibly compound) predicate against the
    facade and keep matching rows. The filtered frame stays private."""
    from .r_expr import eval_expr, parse
    from .safeframe import SafeColumn
    mask = eval_expr(parse(argstr), sf)
    if not isinstance(mask, SafeColumn):
        raise DisclosureError("filter needs a boolean condition, e.g. filter(age >= 18)")
    return sf[mask]


def _mutate(sf, argstr: str):
    """``mutate(name = <expr>, ...)`` — add derived columns via the expression
    parser; returns a new (private) SafeFrame."""
    from .r_expr import eval_expr, parse
    for pair in _split_top(argstr, [","]):
        m = re.match(r"^\s*([A-Za-z_.][\w.]*)\s*=\s*(?!=)(.*)$", pair, re.S)
        if not m:
            raise ValidationError(f"mutate needs name = expr, got {pair!r}", kind="syntax")
        name, expr = m.group(1), m.group(2).strip()
        sf = sf.assign(**{name: eval_expr(parse(expr), sf)})
    return sf


def _select(sf, argstr: str):
    """``select(a, b)`` keeps columns; ``select(-a, -b)`` drops them."""
    items = [a.strip() for a in _split_top(argstr, [","]) if a.strip()]
    drop = [i[1:].strip() for i in items if i.startswith("-")]
    keep = [i for i in items if not i.startswith("-")]
    if drop and keep:
        raise DisclosureError("select cannot mix kept and dropped (-) columns")
    for c in drop + keep:
        if not _IDENT.match(c):
            raise ValidationError(f"expected a column name in select, got {c!r}", kind="syntax")
        _need_col(sf._df, c)
    return sf.drop(drop) if drop else sf[keep]


def _rename(sf, argstr: str):
    """``rename(new = old, ...)``."""
    mapping = {}
    for pair in _split_top(argstr, [","]):
        m = re.match(r"^\s*([A-Za-z_.][\w.]*)\s*=\s*([A-Za-z_.][\w.]*)\s*$", pair)
        if not m:
            raise ValidationError(f"rename needs new = old, got {pair!r}", kind="syntax")
        new, old = m.group(1), m.group(2)
        _need_col(sf._df, old)
        mapping[old] = new
    return sf.rename(columns=mapping)


def _arrange(sf, argstr: str):
    """``arrange(a, desc(b))`` — reorder rows (non-disclosive; no head/pull to read
    the order back out)."""
    by, ascending = [], []
    for term in _split_top(argstr, [","]):
        term = term.strip()
        m = re.match(r"^desc\s*\(\s*([A-Za-z_.][\w.]*)\s*\)$", term)
        col, asc = (m.group(1), False) if m else (term, True)
        if not _IDENT.match(col):
            raise ValidationError(f"arrange takes column names, got {term!r}", kind="syntax")
        _need_col(sf._df, col)
        by.append(col); ascending.append(asc)
    return sf.sort_values(by, ascending=ascending)


def _distinct(sf, argstr: str):
    """``distinct()`` / ``distinct(cols)`` — drop duplicate rows."""
    subset = _cols(argstr) if argstr.strip() else None
    for c in subset or []:
        _need_col(sf._df, c)
    return sf.drop_duplicates(subset=subset)


def _summarise(sf, group, argstr: str) -> Released:
    if group is None:
        raise DisclosureError("summarise needs a preceding group_by(...)")
    by = group[0] if len(group) == 1 else group
    specs = []
    for pair in _split_top(argstr, [","]):
        m = re.match(r"^(\w+)\s*=\s*(\w+)\s*\(\s*([\w.]*)\s*\)$", pair.strip())
        if not m:
            raise ValidationError(
                f"summary must be name = fn(col), got {pair!r}", kind="syntax")
        name, fn, col = m.group(1), m.group(2), m.group(3)
        if fn not in _AGG_MAP:
            raise DisclosureError(
                f"aggregation '{fn}' is not allowed; choose one of {sorted(_AGG_MAP)}")
        value = _need_col(sf._df, col) if col else (group[0] if fn == "n" else None)
        specs.append((name, _AGG_MAP[fn], value))
    if len(specs) == 1:
        return sf._verbs.group_agg(sf._df, by, specs[0][2], specs[0][1])
    # multi-stat -> a frame, one column per named summary (each cell suppressed on
    # its group count, via the shared group_agg release).
    rels = [sf._verbs.group_agg(sf._df, by, value, agg) for _n, agg, value in specs]
    index = rels[0].payload["index"]
    dicts = [dict(zip(r.payload["index"], r.payload["values"])) for r in rels]
    columns = [n for n, _a, _v in specs]
    data = [[d.get(g) for d in dicts] for g in index]
    return Released(
        {"type": "frame", "columns": columns, "index": index, "data": data},
        audit={"kind": "table", "verb": "r.summarise", "by": by, "stats": columns,
               "backend": "pandas"})


def _count(sf, argstr: str) -> Released:
    cols = _cols(argstr)
    if len(cols) != 1:
        raise DisclosureError("count(col) supports a single column in this dialect")
    _need_col(sf._df, cols[0])
    return sf._verbs.value_counts(sf._df, cols[0])


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


_R_REDUCERS = {"mean": "mean", "sum": "sum", "median": "median",
               "sd": "std", "var": "var"}
_CALL_RE = re.compile(r"^([A-Za-z_.][\w.]*)\s*\((.*)\)\s*$", re.S)
_DOLLAR_RE = re.compile(r"^([A-Za-z_.][\w.]*)\$([A-Za-z_.][\w.]*)$")


def _aggregate(verbs, argstr: str, sources: dict) -> Released:
    """``aggregate(y ~ g1 + g2, data=df, FUN=mean)`` -> grouped aggregation."""
    formula = data = fun = None
    for a in _split_top(argstr, [","]):
        km = re.match(r"^(data|FUN|by|subset|na\.action)\s*=\s*(.+)$", a.strip(), re.S)
        if km:
            if km.group(1) == "data":
                data = km.group(2).strip()
            elif km.group(1) == "FUN":
                fun = km.group(2).strip()
        elif "~" in a:
            formula = a
    if formula is None or data is None:
        raise ValidationError("aggregate needs a formula and data=", kind="syntax")
    lhs, rhs = formula.split("~", 1)
    y = lhs.strip()
    bys = [t.strip() for t in _split_top(rhs, ["+"]) if t.strip()]
    if not _IDENT.match(y) or not all(_IDENT.match(b) for b in bys):
        raise ValidationError("aggregate formula terms must be column names", kind="syntax")
    fn = (fun or "mean").strip()
    if fn not in _AGG_MAP:
        raise DisclosureError(f"FUN '{fn}' is not allowed; choose one of {sorted(_AGG_MAP)}")
    df = _resolve_df(data, sources)
    for c in [y] + bys:
        _need_col(df, c)
    by = bys[0] if len(bys) == 1 else bys
    return verbs.group_agg(df, by, y, _AGG_MAP[fn])


def _table(verbs, argstr: str, sources: dict) -> Released:
    """``table(df$x)`` -> value_counts; ``table(df$x, df$y)`` -> crosstab."""
    cols, frame = [], None
    for a in _split_top(argstr, [","]):
        dm = _DOLLAR_RE.match(a.strip())
        if dm:
            frame = frame or dm.group(1)
            cols.append(dm.group(2))
        else:
            raise ValidationError(f"table takes df$col arguments, got {a.strip()!r}",
                                  kind="syntax")
    if frame is None:
        raise ValidationError("table needs df$col (which data frame?)", kind="syntax")
    df = _resolve_df(frame, sources)
    for c in cols:
        _need_col(df, c)
    if len(cols) == 1:
        return verbs.value_counts(df, cols[0])
    if len(cols) == 2:
        return verbs.crosstab(df, cols[0], cols[1])
    raise DisclosureError("table supports one or two columns")


def _base_reducer(verbs, fname: str, argstr: str, sources: dict) -> Released:
    """``mean(df$x)`` / ``sum`` / ``median`` / ``sd`` / ``var`` -> a suppressed
    whole-column scalar via the shared SafeColumn reducer."""
    from .safeframe import SafeFrame
    dm = _DOLLAR_RE.match(argstr.strip())
    if not dm:
        raise DisclosureError(f"{fname}(df$col) needs a single column, e.g. {fname}(df$salary)")
    df = _resolve_df(dm.group(1), sources)
    _need_col(df, dm.group(2))
    return getattr(SafeFrame(df, verbs)[dm.group(2)], _R_REDUCERS[fname])()


def _base_r(verbs, code: str, sources: dict) -> Released:
    """A single (non-pipeline) base-R call: lm/glm, aggregate, table, or a
    whole-column reducer. Default-deny — anything else is refused."""
    m = _CALL_RE.match(code)
    if not m:
        raise DisclosureError(
            "expected an R pipeline (df |> ...) or a supported base-R call "
            "(aggregate/table/lm/glm/mean/...)")
    fname, argstr = m.group(1), m.group(2).strip()
    if fname in ("lm", "glm"):
        return _model(verbs, code, sources)
    if fname == "aggregate":
        return _aggregate(verbs, argstr, sources)
    if fname == "table":
        return _table(verbs, argstr, sources)
    if fname in _R_REDUCERS:
        return _base_reducer(verbs, fname, argstr, sources)
    raise DisclosureError(f"base-R function '{fname}' is not supported in safepy's R dialect")


def translate_r(code: str, verbs, sources: dict) -> Released:
    """Parse a restricted R pipeline or base-R call and return a suppressed
    ``Released``. R is only ever parsed, never executed."""
    code = code.strip()
    if not code:
        raise ValidationError("empty program", kind="empty")
    am = re.match(r"^[A-Za-z_.][\w.]*\s*<-\s*(.+)$", code, re.S)   # `name <- expr`
    if am:
        code = am.group(1).strip()
    stages = _split_top(code, ["|>", "%>%"])
    if len(stages) == 1:                          # not a pipeline -> base-R call
        return _base_r(verbs, code, sources)
    from .safeframe import SafeFrame
    sf = SafeFrame(_resolve_df(stages[0].strip(), sources), verbs)   # option 1 reuse

    group = None
    for stage in stages[1:]:
        verb, argstr = _parse_stage(stage)
        if verb == "filter":
            sf = _filter(sf, argstr)
        elif verb == "mutate":
            sf = _mutate(sf, argstr)
        elif verb == "select":
            sf = _select(sf, argstr)
        elif verb == "rename":
            sf = _rename(sf, argstr)
        elif verb == "arrange":
            sf = _arrange(sf, argstr)
        elif verb == "distinct":
            sf = _distinct(sf, argstr)
        elif verb == "group_by":
            group = _cols(argstr)
            for c in group:
                _need_col(sf._df, c)
        elif verb in ("summarise", "summarize"):
            return _summarise(sf, group, argstr)
        elif verb == "count":
            return _count(sf, argstr)
        else:
            raise DisclosureError(
                f"R verb '{verb}' is not supported (group_by, summarise, count, "
                "filter, mutate, select, rename, arrange, distinct)")
    raise DisclosureError(
        "R pipeline did not end in a releasable summary (summarise/count)")
