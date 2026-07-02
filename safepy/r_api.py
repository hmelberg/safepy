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


_JOINS = {"left_join": "left", "inner_join": "inner",
          "right_join": "right", "full_join": "outer"}


def _unquote(tok: str) -> str:
    tok = tok.strip()
    if tok[:1] in "'\"" and tok[-1:] == tok[:1]:
        return tok[1:-1]
    raise ValidationError(f"join key must be a quoted column name, got {tok!r}",
                          kind="syntax")


def _parse_by(val: str):
    """``by = "col"`` or ``by = c("a", "b")`` -> a column name or list of names."""
    cm = re.match(r"^c\s*\((.*)\)$", val.strip(), re.S)
    if cm:
        return [_unquote(x) for x in _split_top(cm.group(1), [","]) if x.strip()]
    return _unquote(val)


def _join(sf, argstr: str, env: dict, how: str):
    """``left_join(other, by = "key")`` -> SafeFrame.merge. The joined frame stays
    private; it exits only via a suppressed aggregate."""
    from .safeframe import SafeFrame
    args = _split_top(argstr, [","])
    if not args or not _IDENT.match(args[0].strip()):
        raise ValidationError("join needs another data frame as its first argument",
                              kind="syntax")
    other = _env_df(args[0].strip(), env)
    on = None
    for a in args[1:]:
        m = re.match(r"^by\s*=\s*(.+)$", a.strip(), re.S)
        if m:
            on = _parse_by(m.group(1))
    if on is None:                      # natural join: infer shared columns
        on = [c for c in sf._df.columns if c in other.columns]
        if not on:
            raise DisclosureError("join found no common columns; specify by =")
    keys = [on] if isinstance(on, str) else list(on)
    for c in keys:
        if c not in sf._df.columns or c not in other.columns:
            raise DisclosureError(f"join key not in both frames: {c}")
    return sf.merge(SafeFrame(other, sf._verbs), on=on, how=how)


def _col_token(t: str) -> str:
    t = t.strip()
    if t[:1] in "'\"" and t[-1:] == t[:1]:
        return t[1:-1]
    if _IDENT.match(t):
        return t
    raise ValidationError(f"expected a column name, got {t!r}", kind="syntax")


def _name_list(val: str):
    """Parse ``c(a, b)`` / ``c("a", "b")`` / a single bare-or-quoted name into a
    list of column names. Tidyselect helpers (starts_with, ranges) are refused."""
    cm = re.match(r"^c\s*\((.*)\)$", val.strip(), re.S)
    if not cm:
        return [_col_token(val)]
    return [_col_token(x) for x in _split_top(cm.group(1), [","]) if x.strip()]


def _pivot_longer(sf, argstr: str):
    """``pivot_longer(cols = c(a, b), names_to = "name", values_to = "value")``
    -> melt (wide -> long). Non-``cols`` columns become the id vars."""
    cols = names_to = values_to = None
    for a in _split_top(argstr, [","]):
        m = re.match(r"^\s*(cols|names_to|values_to)\s*=\s*(.+)$", a.strip(), re.S)
        if not m:
            raise ValidationError("pivot_longer takes cols=, names_to=, values_to=",
                                  kind="syntax")
        key, val = m.group(1), m.group(2).strip()
        if key == "cols":
            cols = _name_list(val)
        elif key == "names_to":
            names_to = _col_token(val)
        else:
            values_to = _col_token(val)
    if not cols:
        raise ValidationError("pivot_longer needs cols = c(...)", kind="syntax")
    for c in cols:
        _need_col(sf._df, c)
    id_vars = [c for c in sf._df.columns if c not in cols]
    return sf.melt(id_vars=id_vars, value_vars=cols,
                   var_name=names_to or "name", value_name=values_to or "value")


def _pivot_wider(sf, argstr: str):
    """``pivot_wider(names_from = key, values_from = value)`` -> pivot (long ->
    wide). The remaining columns are the id columns (kept as columns, tibble-style)."""
    from .safeframe import SafeFrame
    names_from = values_from = None
    for a in _split_top(argstr, [","]):
        m = re.match(r"^\s*(names_from|values_from|id_cols|values_fill)\s*=\s*(.+)$",
                     a.strip(), re.S)
        if not m:
            raise ValidationError("pivot_wider takes names_from=, values_from=",
                                  kind="syntax")
        key, val = m.group(1), m.group(2).strip()
        if key == "names_from":
            names_from = _col_token(val)
        elif key == "values_from":
            values_from = _col_token(val)
    if names_from is None or values_from is None:
        raise ValidationError("pivot_wider needs names_from= and values_from=", kind="syntax")
    _need_col(sf._df, names_from)
    _need_col(sf._df, values_from)
    index = [c for c in sf._df.columns if c not in (names_from, values_from)]
    if not index:
        raise DisclosureError("pivot_wider needs at least one id column")
    wide = sf._df.pivot(index=index, columns=names_from, values=values_from).reset_index()
    wide.columns.name = None
    return SafeFrame(wide, sf._verbs)


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


def _env_df(name, env):
    """Resolve a name (a source or an assigned intermediate) to its pandas frame.
    Refuses a name bound to a *result* (a Released), which is not a data frame."""
    from .safeframe import SafeFrame
    if name not in env:
        raise ValidationError(f"unknown data source: {name!r}", kind="name")
    v = env[name]
    if not isinstance(v, SafeFrame):
        raise DisclosureError(f"'{name}' is a result, not a data frame")
    return v._df


def _model(verbs, code: str, env: dict) -> Released:
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
    df = _env_df(data, env)
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


def _aggregate(verbs, argstr: str, env: dict) -> Released:
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
    df = _env_df(data, env)
    for c in [y] + bys:
        _need_col(df, c)
    by = bys[0] if len(bys) == 1 else bys
    return verbs.group_agg(df, by, y, _AGG_MAP[fn])


def _table(verbs, argstr: str, env: dict) -> Released:
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
    df = _env_df(frame, env)
    for c in cols:
        _need_col(df, c)
    if len(cols) == 1:
        return verbs.value_counts(df, cols[0])
    if len(cols) == 2:
        return verbs.crosstab(df, cols[0], cols[1])
    raise DisclosureError("table supports one or two columns")


def _base_reducer(verbs, fname: str, argstr: str, env: dict) -> Released:
    """``mean(df$x)`` / ``sum`` / ``median`` / ``sd`` / ``var`` -> a suppressed
    whole-column scalar via the shared SafeColumn reducer."""
    from .safeframe import SafeFrame
    dm = _DOLLAR_RE.match(argstr.strip())
    if not dm:
        raise DisclosureError(f"{fname}(df$col) needs a single column, e.g. {fname}(df$salary)")
    df = _env_df(dm.group(1), env)
    _need_col(df, dm.group(2))
    return getattr(SafeFrame(df, verbs)[dm.group(2)], _R_REDUCERS[fname])()


def _base_r(verbs, code: str, env: dict) -> Released:
    """A single (non-pipeline) base-R call: lm/glm, aggregate, table, or a
    whole-column reducer. Default-deny — anything else is refused."""
    m = _CALL_RE.match(code)
    if not m:
        raise DisclosureError(
            "expected an R pipeline (df |> ...) or a supported base-R call "
            "(aggregate/table/lm/glm/mean/...)")
    fname, argstr = m.group(1), m.group(2).strip()
    if fname in ("lm", "glm"):
        return _model(verbs, code, env)
    if fname == "aggregate":
        return _aggregate(verbs, argstr, env)
    if fname == "table":
        return _table(verbs, argstr, env)
    if fname in _R_REDUCERS:
        return _base_reducer(verbs, fname, argstr, env)
    raise DisclosureError(f"base-R function '{fname}' is not supported in safepy's R dialect")


# statement-continuation tokens: a physical line ending with one of these (or a
# next line starting with one) continues the current statement.
_CONT_TRAIL = ("|>", "%>%", "+", "-", "*", "/", "^", "&", "|", ",", "~", "<-",
               "(", "<", ">", "<=", ">=", "==", "!=")
_CONT_LEAD = ("|>", "%>%", "+", "*", "/", "^", "&", "|", ",", ")")


def _strip_comment(line: str) -> str:
    """Drop an R ``#`` comment (respecting string literals)."""
    out, quote = [], None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch; out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    return "".join(out)


def _depth(s: str) -> int:
    d, quote = 0, None
    for ch in s:
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch in "([{":
            d += 1
        elif ch in ")]}":
            d -= 1
    return d


def _split_statements(code: str) -> list[str]:
    """Split an R script into statements, merging continuation lines (unbalanced
    brackets, a trailing pipe/operator, or a next line that leads with one) and
    honouring ``;`` separators. Comments are stripped."""
    lines = [_strip_comment(l) for l in code.split("\n")]
    stmts, buf = [], ""
    for i, line in enumerate(lines):
        buf = (buf + "\n" + line) if buf else line
        b = buf.strip()
        if not b:
            buf = ""
            continue
        if _depth(buf) > 0 or b.endswith(_CONT_TRAIL):
            continue
        nxt = next((lines[j].strip() for j in range(i + 1, len(lines))
                    if lines[j].strip()), "")
        if nxt.startswith(_CONT_LEAD):
            continue
        for part in _split_top(buf, [";"]):
            if part.strip():
                stmts.append(part.strip())
        buf = ""
    for part in _split_top(buf, [";"]):              # trailing (unterminated) buffer
        if part.strip():
            stmts.append(part.strip())
    return stmts


def _eval_statement(stmt: str, verbs, env: dict):
    """Evaluate one statement to a SafeFrame (shaping) or a Released (terminal /
    a bare name that refers to a bound result)."""
    from .safeframe import SafeFrame
    stages = _split_top(stmt, ["|>", "%>%"])
    if len(stages) == 1:
        s = stages[0].strip()
        if _IDENT.match(s):                          # a bare name -> its binding
            if s not in env:
                raise ValidationError(f"unknown name: {s!r}", kind="name")
            return env[s]
        return _base_r(verbs, s, env)                # a base-R call
    src = stages[0].strip()
    if src not in env:
        raise ValidationError(f"unknown data source: {src!r}", kind="name")
    if not isinstance(env[src], SafeFrame):
        raise DisclosureError(f"cannot pipe from '{src}' (it is a result, not a frame)")
    sf = env[src]
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
        elif verb == "pivot_longer":
            sf = _pivot_longer(sf, argstr)
        elif verb == "pivot_wider":
            sf = _pivot_wider(sf, argstr)
        elif verb in _JOINS:
            sf = _join(sf, argstr, env, _JOINS[verb])
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
                "filter, mutate, select, rename, arrange, distinct, "
                "pivot_longer, pivot_wider, "
                "left_join/inner_join/right_join/full_join)")
    return sf                                        # shaping-only pipeline -> a frame


def translate_r(code: str, verbs, sources: dict) -> Released:
    """Parse a restricted R script (one or more statements) and return a suppressed
    ``Released``. R is only ever parsed, never executed. Assignments (``name <-
    expr``) bind intermediate frames in an environment; the final statement is the
    released result."""
    from .safeframe import SafeFrame
    if not code.strip():
        raise ValidationError("empty program", kind="empty")
    env = {name: SafeFrame(df.to_pandas() if hasattr(df, "to_pandas") else df, verbs)
           for name, df in sources.items()}
    result = None
    for stmt in _split_statements(code):
        am = re.match(r"^([A-Za-z_.][\w.]*)\s*<-\s*(.+)$", stmt, re.S)   # name <- expr
        if am:
            result = _eval_statement(am.group(2).strip(), verbs, env)
            env[am.group(1)] = result
        else:
            result = _eval_statement(stmt, verbs, env)
    if result is None:
        raise DisclosureError("R script did not end in a releasable result")
    return result   # the last statement's value; a bare SafeFrame is refused by the mediator
