"""The DuckDB (SQL) dialect — STRICT: gated execution over the shared release core.

Unlike R (pure translation), SQL *is* executed — but only after a static gate, in
a locked-down engine, and released only through the shared audited suppressor:

1. **Static gate on the parsed AST.** ``json_serialize_sql`` parses the user SQL
   *without executing it*. The gate then enforces: exactly one SELECT statement;
   every function anywhere in the tree is on a whitelist (default-deny — this
   kills ``min``/``max``/``quantile``/``string_agg``/``read_csv`` even inside
   subqueries); no window functions; and the *outer* select list contains only
   GROUP BY keys and whitelisted aggregates (so raw rows / DISTINCT dumps /
   scalar-subquery leaks can never be the result shape).
2. **Locked execution.** ``enable_external_access=false`` (+ configuration lock)
   kills COPY/ATTACH/INSTALL/httpfs; only the registered private frames are
   visible. When the policy winsorizes (Tiltak 2), numeric columns are winsorized
   *at registration*, so every moment aggregate the SQL computes is capped —
   matching the pandas dialect's global-quantile winsorize on unfiltered queries
   (a WHERE + moment-stat combination sees full-table caps rather than
   subset caps; slightly more conservative than the pandas dialect).
3. **Release through the shared core.** The outer select is rewritten (JSON
   surgery on the AST, then ``json_deserialize_sql``) so each aggregate carries a
   paired per-group ``count``; each aggregate column is then released via
   ``SafeVerbs._release_group_agg`` — the identical suppressor the pandas and
   polars dialects use. Inner shaping (subqueries/CTEs/joins/LIMIT) is a private
   intermediate: shape down to one row and the paired count is 1, so the cell is
   suppressed.
"""

from __future__ import annotations

import copy
import json

import pandas as pd

from .errors import DisclosureError, ValidationError
from .result import Released
from .safe import SafeVerbs, _winsorize_col

# SQL aggregate -> the release-core agg name (drives the suppression floor).
# Extremes/order stats/listification (min/max/quantile/first/last/string_agg/
# list/histogram/mode/arg_*) are absent on purpose: default-deny.
_SQL_AGGS = {
    "avg": "mean", "mean": "mean", "sum": "sum",
    "count": "count", "count_star": "size",
    "median": "median",
    "stddev": "std", "stddev_samp": "std",
    "var_samp": "var", "variance": "var",
}

# Scalar/element-wise functions allowed anywhere in the query (default-deny:
# anything not listed — including table functions like read_csv — is refused).
_SCALAR_FUNCS = frozenset({
    # arithmetic operators (serialized as functions)
    "+", "-", "*", "/", "//", "%", "^",
    # numeric
    "abs", "round", "floor", "ceil", "ceiling", "ln", "log", "log2", "log10",
    "exp", "sqrt", "pow", "power", "sign", "greatest", "least",
    # string (element-wise)
    "lower", "upper", "substr", "substring", "length", "trim", "ltrim", "rtrim",
    "replace", "concat", "||", "left", "right", "lpad", "rpad", "contains",
    "starts_with", "ends_with", "strpos", "instr", "strip_accents",
    # date/time parts
    "year", "month", "day", "quarter", "week", "dayofweek", "dayofyear",
    "hour", "minute", "second", "date_part", "datepart", "date_trunc",
    "date_diff", "datediff", "age", "strftime", "strptime",
    # null handling / conditionals
    "coalesce", "ifnull", "nullif", "if",
})

_ALLOWED_FUNCS = _SCALAR_FUNCS | frozenset(_SQL_AGGS)


def _connect(sources: dict, policy):
    try:
        import duckdb
    except ImportError:  # pragma: no cover
        raise DisclosureError("the 'duckdb' package is required for the duckdb dialect")
    con = duckdb.connect()
    con.execute("SET enable_external_access=false")
    for name, df in sources.items():
        pdf = df.to_pandas() if hasattr(df, "to_pandas") else pd.DataFrame(df)
        # Tiltak 2 at the data boundary: every numeric column is winsorized before
        # the SQL engine ever sees it (matches the pandas dialect's global-quantile
        # winsorize for moment stats; median/count are unaffected by tail caps).
        if policy.suppression.winsorize is not None:
            for c in pdf.columns:
                pdf = _winsorize_col(pdf, c, policy)
        con.register(name, pdf)
    try:
        con.execute("SET lock_configuration=true")   # freeze settings (defence in depth)
    except Exception:  # pragma: no cover - older duckdb without the flag
        pass
    return con


def _parse(con, sql: str):
    """Parse (never execute) the SQL into a serialized AST; enforce single SELECT."""
    try:
        raw = con.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0]
    except Exception:
        raise ValidationError("could not parse SQL", kind="parse")
    ast = json.loads(raw)
    if ast.get("error"):
        raise ValidationError(
            f"could not parse SQL: {ast.get('error_message', 'syntax error')}",
            kind="parse")
    stmts = ast.get("statements") or []
    if len(stmts) != 1:
        raise DisclosureError("exactly one SQL statement is allowed")
    node = (stmts[0] or {}).get("node") or {}
    if node.get("type") != "SELECT_NODE":
        raise DisclosureError("only SELECT statements are allowed in safepy's duckdb dialect")
    return ast, node


def _walk(obj):
    """Default-deny walk over the whole AST: every function anywhere must be
    whitelisted; window functions and sampling are refused outright."""
    if isinstance(obj, dict):
        cls = obj.get("class")
        typ = obj.get("type")
        if isinstance(typ, str) and typ.startswith("WINDOW"):
            raise DisclosureError("window functions are not supported in safepy's duckdb dialect")
        if cls == "FUNCTION":
            fn = (obj.get("function_name") or "").lower()
            if fn not in _ALLOWED_FUNCS:
                raise DisclosureError(f"SQL function '{fn}' is not allowed in safepy")
            if obj.get("distinct") and fn != "count":
                raise DisclosureError(
                    "DISTINCT is only supported inside count(DISTINCT col)")
        if obj.get("sample"):
            raise DisclosureError("USING SAMPLE is not supported")
        for v in obj.values():
            _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v)


def _colname(item) -> str:
    names = item.get("column_names") or []
    if not names:
        raise ValidationError("unsupported column reference", kind="syntax")
    return names[-1]


def _sig(expr) -> str:
    """A canonical signature of an expression (ignoring source location and alias)
    so a select item can be matched to a GROUP BY expression."""
    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in sorted(o.items())
                    if k not in ("query_location", "alias")}
        if isinstance(o, list):
            return [strip(v) for v in o]
        return o
    return json.dumps(strip(expr), sort_keys=True)


def _is_agg(item) -> bool:
    return (item.get("class") == "FUNCTION"
            and (item.get("function_name") or "").lower() in _SQL_AGGS)


def _prepare_outer(node):
    """Validate the outer select (the release boundary) and return
    ``(group_out_names, agg_items)``. Only GROUP BY keys and safe aggregates may
    appear; ORDER BY may only reference group keys (ordering by an aggregate leaks
    the exact, unrounded values); HAVING/QUALIFY are refused (they filter on exact
    aggregate values — an oracle). Missing group keys are auto-added to the output;
    GROUP BY expressions are supported."""
    if node.get("having"):
        raise DisclosureError(
            "HAVING is not supported: it filters on exact aggregate values "
            "(an oracle on unrounded results). Filter rows with WHERE instead.")
    if node.get("qualify"):
        raise DisclosureError("QUALIFY is not supported")

    group_exprs = (node.get("group_expressions")
                   or (node.get("groups") or {}).get("group_expressions") or [])
    group_sigs = {_sig(g) for g in group_exprs}
    group_cols = {_colname(g) for g in group_exprs if g.get("class") == "COLUMN_REF"}

    # ORDER BY: only group keys (a plain group column, or a group expression).
    for m in node.get("modifiers") or []:
        t = m.get("type")
        if t == "LIMIT_MODIFIER":
            continue
        if t != "ORDER_MODIFIER":
            raise DisclosureError(f"SQL modifier '{t}' is not supported")
        for order in m.get("orders") or []:
            e = order.get("expression") or {}
            ok = (e.get("class") == "COLUMN_REF" and _colname(e) in group_cols) \
                or _sig(e) in group_sigs
            if not ok:
                raise DisclosureError(
                    "ORDER BY may only reference GROUP BY keys; ordering by an "
                    "aggregate would leak the exact (unrounded) values")

    select_list = node.get("select_list") or []
    aggs, present_group_sigs = [], set()
    for it in select_list:
        if (it.get("alias") or "").startswith("__"):
            raise ValidationError("aliases may not start with '__'", kind="name")
        if _is_agg(it):
            aggs.append(it)
            continue
        # otherwise it must be a group key (plain column or a group expression)
        is_col = it.get("class") == "COLUMN_REF" and _colname(it) in group_cols
        if is_col or _sig(it) in group_sigs:
            present_group_sigs.add(_sig(it) if not is_col else _colname(it))
            if not it.get("alias"):
                it["alias"] = _colname(it) if is_col else "__grp"
            continue
        raise DisclosureError(
            "each SELECT item must be a GROUP BY key or a safe aggregate "
            "(avg/sum/count/count(DISTINCT)/median/stddev/var_samp); "
            "bare columns and arithmetic on aggregates are not released")

    if not aggs:
        raise DisclosureError(
            "the query must compute at least one aggregate; raw rows or distinct "
            "values are never released")

    # auto-add any GROUP BY key missing from the select, so it becomes a row label.
    group_out = []
    for j, g in enumerate(group_exprs):
        key = _colname(g) if g.get("class") == "COLUMN_REF" else _sig(g)
        # find an existing select item for this group expr to reuse its alias
        alias = None
        for it in select_list:
            it_key = (_colname(it) if it.get("class") == "COLUMN_REF"
                      and _colname(it) in group_cols else _sig(it))
            if not _is_agg(it) and it_key == key:
                alias = it.get("alias")
                break
        if alias is None:
            alias = _colname(g) if g.get("class") == "COLUMN_REF" else f"__grp_{j}"
            add = copy.deepcopy(g)
            add["alias"] = alias
            select_list.append(add)
        group_out.append(alias)
    return group_out, aggs


def _agg_arg_name(a) -> str:
    """A readable name for what the aggregate aggregates (for audit/labels)."""
    ch = a.get("children") or []
    if ch and ch[0].get("class") == "COLUMN_REF":
        return _colname(ch[0])
    return "*" if (a.get("function_name") or "").lower() == "count_star" else "expr"


def _inject_counts(node, aggs):
    """Pair each aggregate with a ``count`` over the same argument (``count(*)``
    pairs with itself; ``count(DISTINCT x)`` pairs with a plain ``count(x)`` — the
    contributing group size). Group keys are already aliased by _prepare_outer."""
    labels = []                     # (value_col, user_alias_or_None, fname, argname)
    for i, a in enumerate(aggs):
        user_alias = a.get("alias") or None
        a["alias"] = f"__val_{i}"
        c = copy.deepcopy(a)
        if (c.get("function_name") or "").lower() != "count_star":
            c["function_name"] = "count"
            c["distinct"] = False
        c["alias"] = f"__cnt_{i}"
        node["select_list"].append(c)
        labels.append((f"__val_{i}", user_alias, (a.get("function_name") or "").lower(),
                       _agg_arg_name(a)))
    return labels


def _deserialize(con, ast) -> str:
    row = con.execute("SELECT json_deserialize_sql(CAST(? AS JSON))",
                      [json.dumps(ast)]).fetchone()
    return row[0]


def run_sql(code: str, verbs: SafeVerbs, sources: dict) -> Released:
    """Gate, execute (locked), and release one SQL SELECT through the shared core."""
    if not code.strip():
        raise ValidationError("empty program", kind="empty")
    con = _connect(sources, verbs._policy)
    ast, node = _parse(con, code)
    _walk(ast["statements"])                     # default-deny, whole tree
    group_cols, aggs = _prepare_outer(node)
    labels = _inject_counts(node, aggs)
    res = con.execute(_deserialize(con, ast)).df()

    by = group_cols[0] if len(group_cols) == 1 else (group_cols or None)
    if group_cols:
        # match pandas groupby(observed=True): null group keys are dropped (a lone
        # null-key group would be an unpaired small cell).
        res = res.dropna(subset=group_cols)
    idx = res.set_index(group_cols) if group_cols else None
    rels = []
    for i, (vcol, user_alias, fname, argname) in enumerate(labels):
        if idx is not None:
            table = pd.to_numeric(idx[vcol], errors="coerce")
            counts = idx[f"__cnt_{i}"].fillna(0).astype(int)
        else:                                    # whole-table aggregate: one row
            label = user_alias or argname
            table = pd.to_numeric(pd.Series([res[vcol].iloc[0]], index=[label]),
                                  errors="coerce")
            counts = pd.Series([int(res[f"__cnt_{i}"].iloc[0])], index=[label])
        rel = verbs._release_group_agg(table, counts, agg=_SQL_AGGS[fname], by=by,
                                       value=argname, backend="duckdb")
        if user_alias:
            rel.payload["name"] = user_alias
        rels.append(rel)
    if len(rels) == 1:
        return rels[0]
    index = rels[0].payload["index"]
    dicts = [dict(zip(r.payload["index"], r.payload["values"])) for r in rels]
    columns = [r.payload["name"] for r in rels]
    data = [[d.get(g) for d in dicts] for g in index]
    return Released(
        {"type": "frame", "columns": columns, "index": index, "data": data},
        audit={"kind": "table", "verb": "sql_agg_compound", "by": by,
               "stats": columns, "backend": "duckdb"})
