"""The polars dialect (STRICT mode) — M1-b: polars *surface*, pandas *backend*.

Users write genuine polars call shapes::

    import polars as pl
    df.filter(pl.col("age") >= 18).group_by("sex").agg(pl.col("salary").mean())

Real polars evaluates the *shaping* (``filter``/``select``/expressions), so we
never reimplement the polars expression algebra. At the *terminal* aggregation
the shaped-but-still-private frame is converted to pandas and handed to the
existing, audited :class:`safepy.safe.SafeVerbs` — so suppression (min_n, counts,
winsorization, the Tiltak measures) is byte-identical to the pandas dialect.

Security model (STRICT): the facade *is* the boundary. ``SafePolarsFrame`` /
``SafeExpr`` expose only non-disclosive shaping plus the safe reducers; the
value-ordered / row-identity surface (``head``/``get_column``/``sort``/``max``/
``top_k``/``gather``/``to_pandas`` …) simply does not exist. The real objects
live in ``_pl`` / ``_expr`` (private; the AST gate blocks ``_`` access).
"""

from __future__ import annotations

import polars as pl

from .errors import DisclosureError
from .result import Released
from .safe import SafeVerbs

# Reducers that are safe *given* a minimum contributing count — mirrors the
# pandas SafeColumn/_ALLOWED_AGGS surface. Value-ordered reducers (max/min/
# quantile/top_k/arg_max …) are deliberately absent: they return individuals.
_SAFE_REDUCERS = frozenset({"mean", "sum", "count", "median", "std", "var"})

# Grouped moment aggregates that winsorization (Tiltak 2) affects — must mirror
# safe._WINSOR_AGGS so the native path winsorizes exactly where the pandas path does.
_WINSOR_AGGS = frozenset({"mean", "std", "var", "sum"})

# aggfuncs allowed for pivot_table — mirrors safe._ALLOWED_AGGS.
_SAFE_AGGFUNCS = frozenset({"mean", "sum", "count", "size", "median", "std", "var"})


def _eager(frame):
    """Materialize a LazyFrame; pass an eager frame through. Shaping stays lazy;
    we collect only at the conversion boundary (native aggregate / to_pandas)."""
    return frame.collect() if isinstance(frame, pl.LazyFrame) else frame


def _schema(frame):
    """Column schema (name -> dtype) for an eager or lazy frame, without a
    (warning-triggering) implicit materialization."""
    return frame.collect_schema() if isinstance(frame, pl.LazyFrame) else frame.schema


def _winsorize_polars(pl_df: pl.DataFrame, value: str, agg: str, policy) -> pl.DataFrame:
    """Global winsorization of ``value`` (Tiltak 2), computed natively in polars.
    Uses linear-interpolation quantiles to match pandas/``protect.winsorize``
    byte-for-byte (verified numerically). No-op when winsorize is off, the agg is
    not moment-based, or the column is non-numeric/boolean."""
    w = policy.suppression.winsorize
    if w is None or agg not in _WINSOR_AGGS:
        return pl_df
    dtype = _schema(pl_df)[value]
    if not dtype.is_numeric() or dtype == pl.Boolean:
        return pl_df
    lo = _eager(pl_df.select(pl.col(value).quantile(float(w[0]), interpolation="linear"))).item()
    hi = _eager(pl_df.select(pl.col(value).quantile(float(w[1]), interpolation="linear"))).item()
    return pl_df.with_columns(pl.col(value).clip(lo, hi))


def _native_pivot_table(pl_df, values: str, i0: str, c0: str, aggfunc: str):
    """Compute a 2-D value table and its per-cell contributing counts natively in
    polars (``group_by(i0, c0).agg`` then pivot), returning aligned pandas
    DataFrames with sorted axes — matching pandas ``pivot_table`` byte-for-byte
    (verified numerically). Single index column, single pivot column."""
    # drop null index/pivot keys to match pandas pivot_table (value nulls kept).
    frame = _eager(pl_df).drop_nulls([i0, c0])
    vexpr = pl.len() if aggfunc == "size" else getattr(pl.col(values), aggfunc)()
    grp = frame.group_by(i0, c0).agg(vexpr.alias("__v"), pl.col(values).count().alias("__n"))
    tab = grp.pivot(on=c0, index=i0, values="__v").to_pandas().set_index(i0)
    counts = grp.pivot(on=c0, index=i0, values="__n").to_pandas().set_index(i0)
    tab = tab.sort_index().sort_index(axis=1)
    counts = counts.reindex_like(tab)
    tab.index.name = counts.index.name = i0
    tab.columns.name = counts.columns.name = c0
    return tab, counts


def _native_frame_reduce(pl_df, stat: str, policy):
    """Per-column ``(name, raw aggregate, non-null count)`` computed natively in
    polars, ready for the backend-neutral ``safeframe._release_frame_reduce``.
    Numeric-only for moment/median stats (matches pandas ``select_dtypes('number')``,
    which excludes boolean); all columns for count/nunique. Moment stats are
    winsorized (Tiltak 2) exactly where the pandas path winsorizes."""
    frame = _eager(pl_df)
    schema = frame.schema
    if stat in ("count", "nunique"):
        cols = list(schema.names())
    else:
        cols = [c for c, dt in schema.items() if dt.is_numeric() and dt != pl.Boolean]
    if not cols:
        raise DisclosureError(f"{stat} needs at least one numeric column")
    per_col = []
    for c in cols:
        n = int(frame.select(pl.col(c).count()).item())         # non-null count
        if stat == "count":
            raw = n
        elif stat == "nunique":
            raw = int(frame.select(pl.col(c).drop_nulls().n_unique()).item())
        else:
            work = _winsorize_polars(frame, c, stat, policy)
            raw = work.select(getattr(pl.col(c), stat)()).item()
        per_col.append((c, raw, n))
    return per_col


def _native_group_agg(pl_df: pl.DataFrame, by: list, value: str, agg: str, policy):
    """Compute the per-group aggregate and paired row counts **natively in polars**,
    returning them as pandas Series (small — one row per group) sharing a group
    index, ready for the backend-neutral ``SafeVerbs._release_group_agg``. Only the
    aggregate crosses to pandas; the private per-row frame stays in polars."""
    # drop rows with a null group key so we match pandas groupby(observed=True),
    # which excludes them — a lone null-key group would be an unpaired small cell.
    pl_df = pl_df.drop_nulls(by)
    counts = (_eager(pl_df.group_by(by).agg(pl.len().alias("__n")))
              .to_pandas().set_index(by)["__n"])
    if agg == "size":
        return counts.copy(), counts
    work = _winsorize_polars(pl_df, value, agg, policy)
    expr = getattr(pl.col(value), agg)()          # mean/sum/std/var/median/count
    table = (_eager(work.group_by(by).agg(expr.alias("__v")))
             .to_pandas().set_index(by)["__v"])
    return table, counts


class SafeExpr:
    """A wrapped ``pl.Expr``. Shaping operations return a new SafeExpr carrying a
    real polars expression; the safe reducers additionally record ``(col, agg)``
    so the terminal ``agg()`` can route through the suppression backend."""

    _is_polars_intermediate = True

    def __init__(self, expr: pl.Expr, *, col: str | None = None, agg: str | None = None,
                 pre: pl.Expr | None = None, name: str | None = None):
        self._expr = expr
        self._col = col          # source column name, if this is a simple column ref
        self._agg = agg          # reducer name, if a safe reducer was applied
        self._pre = expr if pre is None else pre   # the expression *before* the reducer
        self._name = name        # output name from .alias(), if any

    def _shape(self, expr) -> "SafeExpr":
        return SafeExpr(expr)    # a derived/shaping expr is no longer a simple column

    @property
    def str(self):
        return _SafeExprStr(self._expr)

    @property
    def dt(self):
        return _SafeExprDt(self._expr)

    @staticmethod
    def _val(o):
        return o._expr if isinstance(o, SafeExpr) else o

    # -- comparisons -> boolean expression (for filter) --
    def __gt__(self, o): return self._shape(self._expr > self._val(o))
    def __ge__(self, o): return self._shape(self._expr >= self._val(o))
    def __lt__(self, o): return self._shape(self._expr < self._val(o))
    def __le__(self, o): return self._shape(self._expr <= self._val(o))
    def __eq__(self, o): return self._shape(self._expr == self._val(o))
    def __ne__(self, o): return self._shape(self._expr != self._val(o))
    __hash__ = None

    # -- arithmetic -> derived expression --
    def __add__(self, o): return self._shape(self._expr + self._val(o))
    def __sub__(self, o): return self._shape(self._expr - self._val(o))
    def __mul__(self, o): return self._shape(self._expr * self._val(o))
    def __truediv__(self, o): return self._shape(self._expr / self._val(o))
    def __radd__(self, o): return self._shape(self._val(o) + self._expr)
    def __rsub__(self, o): return self._shape(self._val(o) - self._expr)
    def __rmul__(self, o): return self._shape(self._val(o) * self._expr)
    def __neg__(self): return self._shape(-self._expr)

    # -- boolean combinators on masks --
    def __and__(self, o): return self._shape(self._expr & self._val(o))
    def __or__(self, o): return self._shape(self._expr | self._val(o))
    def __invert__(self): return self._shape(~self._expr)

    # -- safe reducers -> a terminal SafeExpr (col + agg captured) --
    def mean(self): return self._reduce("mean")
    def sum(self): return self._reduce("sum")
    def count(self): return self._reduce("count")
    def median(self): return self._reduce("median")
    def std(self): return self._reduce("std")
    def var(self): return self._reduce("var")

    def _reduce(self, agg: str) -> "SafeExpr":
        # ``_pre`` is the expression to aggregate (a column ref or a derived expr);
        # the terminal verb materializes it and routes the reduction through the
        # shared SafeVerbs/SafeColumn suppression path.
        return SafeExpr(getattr(self._expr, agg)(), col=self._col, agg=agg,
                        pre=self._expr, name=self._name)

    def alias(self, name: str) -> "SafeExpr":
        """Name the output column (``pl.col('a').alias('b')``). Preserves the
        source column / reducer so a named aggregation still routes correctly."""
        if not isinstance(name, str):
            raise DisclosureError("alias takes a single name (a string)")
        return SafeExpr(self._expr.alias(name), col=self._col, agg=self._agg,
                        pre=self._pre, name=name)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raise DisclosureError(
            f"expr.{name} is not available in safepy's polars dialect; the safe "
            f"reducers are {sorted(_SAFE_REDUCERS)}")

    def __repr__(self):
        return f"<SafeExpr col={self._col!r} agg={self._agg!r}>"


class _SafeExprStr:
    """``pl.col(...).str`` — element-wise string transforms. Each returns a
    derived (private) SafeExpr; aggregating forms are not exposed. Mirrors the
    pandas ``SafeColumn.str`` whitelist."""

    def __init__(self, expr: pl.Expr):
        self._expr = expr.str

    def _d(self, e) -> SafeExpr:
        return SafeExpr(e)

    def to_uppercase(self): return self._d(self._expr.to_uppercase())
    def to_lowercase(self): return self._d(self._expr.to_lowercase())
    def slice(self, offset, length=None): return self._d(self._expr.slice(offset, length))
    def len_chars(self): return self._d(self._expr.len_chars())
    def strip_chars(self, characters=None): return self._d(self._expr.strip_chars(characters))
    def replace(self, pattern, value, *, literal=True):
        return self._d(self._expr.replace(pattern, value, literal=bool(literal)))
    def zfill(self, length): return self._d(self._expr.zfill(length))
    # search -> boolean mask (literal by default: ReDoS-safe, matches SafeColumn.str)
    def contains(self, pattern, *, literal=True):
        return self._d(self._expr.contains(pattern, literal=bool(literal)))
    def starts_with(self, prefix): return self._d(self._expr.starts_with(prefix))
    def ends_with(self, suffix): return self._d(self._expr.ends_with(suffix))

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raise DisclosureError(f"str.{name} is not available in safepy's polars dialect")


class _SafeExprDt:
    """``pl.col(...).dt`` — element-wise datetime parts. Each returns a derived
    (private) SafeExpr. Mirrors the pandas ``SafeColumn.dt`` whitelist."""

    _PARTS = frozenset({
        "year", "month", "day", "quarter", "week", "weekday", "ordinal_day",
        "hour", "minute", "second", "millisecond", "microsecond",
    })

    def __init__(self, expr: pl.Expr):
        self._expr = expr.dt

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._PARTS:
            raise DisclosureError(f"dt.{name} is not available in safepy's polars dialect")
        return lambda: SafeExpr(getattr(self._expr, name)())


class _SafeThen:
    """The result of ``pl.when(...).then(...)`` — awaits ``.otherwise(...)`` (or a
    chained ``.when(...)``) to become a SafeExpr."""

    def __init__(self, then):
        self._then = then

    def when(self, condition) -> "_SafeWhen":
        return _SafeWhen(self._then.when(SafeExpr._val(condition)))

    def otherwise(self, value) -> SafeExpr:
        return SafeExpr(self._then.otherwise(SafeExpr._val(value)))


class _SafeWhen:
    def __init__(self, when):
        self._when = when

    def then(self, value) -> _SafeThen:
        return _SafeThen(self._when.then(SafeExpr._val(value)))


class SafePl:
    """The ``pl`` facade injected for ``import polars as pl``. Only expression
    constructors are exposed; everything else (read_csv, DataFrame, Series …) is
    refused."""

    def col(self, name):
        if not isinstance(name, str):
            raise DisclosureError("pl.col takes a single column name (a string)")
        return SafeExpr(pl.col(name), col=name)

    def lit(self, value):
        return SafeExpr(pl.lit(value))

    def when(self, condition) -> _SafeWhen:
        """``pl.when(cond).then(a).otherwise(b)`` — the element-wise conditional
        (polars analog of ``np.where``); non-disclosive, so it returns a derived
        expression."""
        return _SafeWhen(pl.when(SafeExpr._val(condition)))

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raise DisclosureError(f"pl.{name} is not available in safepy's polars dialect")


class SafePolarsGroupBy:
    """``df.group_by(*by)`` — only ``agg`` of safe reducers, each a suppressed
    table routed through the pandas backend."""

    _is_polars_intermediate = True

    def __init__(self, pl_df: pl.DataFrame, by, verbs: SafeVerbs):
        self._pl, self._by, self._verbs = pl_df, list(by), verbs

    def len(self):
        """``df.group_by(by).len()`` — the row count per group (polars-idiomatic),
        suppressed below min_n like any other count."""
        table, counts = _native_group_agg(self._pl, self._by, self._by[0], "size",
                                          self._verbs._policy)
        by = self._by[0] if len(self._by) == 1 else self._by
        return self._verbs._release_group_agg(table, counts, agg="size", by=by,
                                              value=self._by[0], backend="polars")

    def agg(self, *exprs):
        if not exprs:
            raise DisclosureError("agg needs at least one reducer, e.g. pl.col('salary').mean()")
        rels = [self._one_agg(e) for e in exprs]
        if len(rels) == 1:
            return rels[0]
        return self._combine(rels)

    def _one_agg(self, e) -> Released:
        """One reducer -> a suppressed series Released. The aggregate + counts are
        computed natively in polars (only the small result crosses to pandas), then
        released through the shared, audited SafeVerbs suppressor."""
        if not isinstance(e, SafeExpr) or e._agg is None:
            raise DisclosureError(
                "agg needs reducers over columns, e.g. pl.col('salary').mean()")
        policy = self._verbs._policy
        by_audit = self._by[0] if len(self._by) == 1 else self._by
        if e._col is not None and e._name is None:   # simple column: fast path
            frame, value = self._pl, e._col
        else:                                        # derived / aliased: materialize it
            value = e._name or "value"
            base = e._pre if e._col is None else pl.col(e._col)
            frame = self._pl.with_columns(base.alias(value))
        table, counts = _native_group_agg(frame, self._by, value, e._agg, policy)
        rel = self._verbs._release_group_agg(table, counts, agg=e._agg, by=by_audit,
                                             value=value, backend="polars")
        if e._name:                                  # honor the alias as the output name
            rel.payload["name"] = e._name
        return rel

    def _combine(self, rels) -> Released:
        """Assemble several suppressed per-group series into one frame, aligned on
        the (shared) group index. Each column keeps its own cell suppression, so a
        group below min_n is blanked across every column."""
        index = rels[0].payload["index"]
        columns = [r.payload["name"] for r in rels]
        dicts = [dict(zip(r.payload["index"], r.payload["values"])) for r in rels]
        data = [[d.get(g) for d in dicts] for g in index]
        return Released(
            {"type": "frame", "columns": columns, "index": index, "data": data},
            audit={"kind": "table", "verb": "group_agg_compound", "by": self._by,
                   "stats": columns, "backend": "polars"})


# Terminal (Released-returning) verbs on the pandas SafeFrame that are safe to
# delegate to from the polars facade — models, causal/survival, hypothesis tests,
# correlation/description, and per-column frame reducers. Intermediate-returning
# verbs (assign/where/groupby/merge/sort_values/…) are deliberately excluded, and
# ``propensity`` (returns a private SafeColumn) is excluded too.
_DELEGATED_VERBS = frozenset({
    "ols", "logit", "poisson", "cox", "kaplan_meier", "logrank", "weibull_aft",
    "lognormal_aft", "loglogistic_aft", "rmst", "feols", "iv", "ate", "refute_ate",
    "synthetic_control", "corr", "cov", "describe", "ttest", "mannwhitney",
    "anova", "chisq", "corr_test",
})


class SafePolarsFrame:
    """The only polars data object user code can touch in STRICT mode."""

    _is_polars_intermediate = True
    _is_polars_safeframe = True     # distinguishes a source frame for the catalog

    def __init__(self, pl_df: pl.DataFrame, verbs: SafeVerbs):
        self._pl = pl_df
        self._verbs = verbs

    def filter(self, predicate) -> "SafePolarsFrame":
        if not isinstance(predicate, SafeExpr):
            raise DisclosureError("filter needs a boolean expression, e.g. pl.col('age') >= 18")
        return SafePolarsFrame(self._pl.filter(predicate._expr), self._verbs)

    def _as_expr(self, e) -> SafeExpr:
        """Normalize a select/with_columns argument: a column name -> a column
        reference; a SafeExpr passes through. Anything else is refused."""
        if isinstance(e, str):
            if e not in _schema(self._pl).names():
                raise DisclosureError(f"unknown column: {e}")
            return SafeExpr(pl.col(e), col=e)
        if isinstance(e, SafeExpr):
            return e
        raise DisclosureError("expected a column name or a pl.col(...) expression")

    def select(self, *exprs):
        """``df.select(...)``. With plain columns / derived expressions this is a
        (non-disclosive) column selection -> a private SafePolarsFrame. With a
        single reducer (``df.select(pl.col('salary').mean())``) it is a
        whole-frame aggregation -> a suppressed scalar, routed through the same
        SafeColumn reducer the pandas dialect uses."""
        norm = [self._as_expr(e) for e in exprs]
        if not norm:
            raise DisclosureError("select needs at least one column or expression")
        terminals = [e for e in norm if e._agg is not None]
        if terminals:
            if len(terminals) != len(norm):
                raise DisclosureError("select cannot mix aggregations with plain columns")
            if len(terminals) == 1:                  # whole-frame reducer -> scalar
                return self._scalar_agg(terminals[0])
            # multiple whole-frame reducers -> a series of named suppressed scalars
            names = [e._name or f"{e._agg}({e._col or 'value'})" for e in terminals]
            values = [self._scalar_agg(e).payload.get("value") for e in terminals]
            return Released(
                {"type": "series", "name": "aggregate", "index": names, "values": values},
                audit={"kind": "table", "verb": "select_aggregate", "stats": names,
                       "backend": "pandas"})
        return SafePolarsFrame(self._pl.select(*[e._expr for e in norm]), self._verbs)

    def _scalar_agg(self, e) -> Released:
        """One whole-frame reducer -> a suppressed scalar Released, via the shared
        SafeColumn reducer (identical to the pandas dialect)."""
        from .safeframe import SafeColumn
        if e._col is not None and e._name is None:   # simple column: convert only it
            series = _eager(self._pl.select(e._col)).to_pandas().iloc[:, 0]
        else:                                        # derived / aliased: materialize it
            base = e._pre if e._col is None else pl.col(e._col)
            series = _eager(self._pl.select(base.alias(e._name or "value"))).to_pandas().iloc[:, 0]
        return getattr(SafeColumn(series, self._verbs), e._agg)()

    def with_columns(self, *exprs, **named) -> "SafePolarsFrame":
        """Add derived columns (polars analog of pandas ``assign``). Returns a
        private SafePolarsFrame; the derived values exit only via a suppressed
        aggregate, so derivation is safe by construction."""
        out = [self._as_expr(e)._expr for e in exprs]
        out += [self._as_expr(e)._expr.alias(name) for name, e in named.items()]
        return SafePolarsFrame(self._pl.with_columns(*out), self._verbs)

    # -- whole-frame per-column reducers: computed natively in polars, released
    #    through the shared safeframe._release_frame_reduce (suppression/noise/round). --
    def mean(self): return self._frame_reduce("mean")
    def sum(self): return self._frame_reduce("sum")
    def median(self): return self._frame_reduce("median")
    def std(self): return self._frame_reduce("std")
    def var(self): return self._frame_reduce("var")
    def count(self): return self._frame_reduce("count")
    def nunique(self): return self._frame_reduce("nunique")

    def _frame_reduce(self, stat: str) -> Released:
        from .safeframe import _release_frame_reduce
        policy = self._verbs._policy
        per_col = _native_frame_reduce(self._pl, stat, policy)
        return _release_frame_reduce(policy, stat, per_col,
                                     rounded=(stat != "nunique"), backend="polars")

    def value_counts(self, col: str) -> Released:
        """Suppressed frequency table of one column — counts computed natively in
        polars (``group_by(col).len()``), released through the shared suppressor.
        Nulls are dropped to match pandas ``value_counts``."""
        if not isinstance(col, str) or col not in _schema(self._pl).names():
            raise DisclosureError(f"unknown column: {col}")
        counts = (_eager(self._pl.drop_nulls(col).group_by(col).agg(pl.len().alias("__n")))
                  .to_pandas().set_index(col)["__n"].sort_values(ascending=False))
        counts.index.name = col
        return self._verbs._release_value_counts(counts, col=col, backend="polars")

    def crosstab(self, row: str, col: str) -> Released:
        """Suppressed 2-D frequency table — counts computed natively in polars
        (``group_by(row, col).len()`` then pivot), released through the shared
        suppressor. Axes are sorted and empty cells filled with 0 to match
        pandas ``crosstab``."""
        names = _schema(self._pl).names()
        for c in (row, col):
            if not isinstance(c, str) or c not in names:
                raise DisclosureError(f"unknown column: {c}")
        long = _eager(self._pl.drop_nulls([row, col]).group_by(row, col)
                      .agg(pl.len().alias("__n")))
        tab = long.pivot(on=col, index=row, values="__n").to_pandas().set_index(row)
        tab = tab.reindex(sorted(tab.index)).sort_index(axis=1).fillna(0).astype(int)
        tab.index.name, tab.columns.name = row, col
        return self._verbs._release_crosstab(tab, row=row, col=col, backend="polars")

    def pivot_table(self, *, values, index, columns=None, aggfunc="mean",
                    min_n=None, round=None) -> Released:
        """``df.pivot_table(...)`` — a 2-D aggregation. Computed natively in polars
        for the single-index / single-column case (the common shape); multi-index,
        multi-column, or ``columns=None`` fall back to the pandas backend, which is
        equally safe (same audited release)."""
        names = _schema(self._pl).names()
        idx = [index] if isinstance(index, str) else list(index)
        cols = None if columns is None else ([columns] if isinstance(columns, str) else list(columns))
        for c in idx + (cols or []) + [values]:
            if c not in names:
                raise DisclosureError(f"unknown column: {c}")
        if cols is None or len(idx) != 1 or len(cols) != 1:      # complex shape -> pandas
            return self._safeframe().pivot_table(
                values=values, index=index, columns=columns, aggfunc=aggfunc,
                min_n=min_n, round=round)
        if aggfunc not in _SAFE_AGGFUNCS:
            raise DisclosureError(
                f"aggfunc '{aggfunc}' is not allowed; choose one of {sorted(_SAFE_AGGFUNCS)}")
        tab, counts = _native_pivot_table(self._pl, values, idx[0], cols[0], aggfunc)
        return self._verbs._release_pivot_table(
            tab, counts, aggfunc=aggfunc, index=idx, columns=cols, values=values,
            min_n=min_n, round=round, backend="polars")

    def group_by(self, *by) -> SafePolarsGroupBy:
        cols = [c for grp in by for c in ([grp] if isinstance(grp, str) else grp)]
        names = _schema(self._pl).names()
        for c in cols:
            if c not in names:
                raise DisclosureError(f"unknown column: {c}")
        return SafePolarsGroupBy(self._pl, cols, self._verbs)

    def _safeframe(self):
        """A pandas SafeFrame over the (converted) data — the delegation target for
        verbs that are backend-neutral and already audited on the pandas facade.
        A LazyFrame source is collected here."""
        from .safeframe import SafeFrame
        return SafeFrame(_eager(self._pl).to_pandas(), self._verbs)

    def _catalog_raw(self):
        """``(n_rows, [(name, dtype, n_missing)])`` for the schema catalog. Collects
        a LazyFrame source (row/missing counts need the data)."""
        frame = _eager(self._pl)
        nulls = dict(zip(frame.columns, frame.null_count().row(0)))
        return frame.height, [(str(c), str(dt), int(nulls[c])) for c, dt in frame.schema.items()]

    def __getattr__(self, name):
        # Terminal, Released-returning verbs are delegated to the audited pandas
        # SafeFrame. Only this explicit whitelist is reachable — never the
        # intermediate-returning shaping verbs (assign/where/groupby/merge/…), so
        # the polars facade stays the security boundary.
        if name.startswith("_"):
            raise AttributeError(name)
        if name in _DELEGATED_VERBS:
            return getattr(self._safeframe(), name)
        raise DisclosureError(
            f"{name!r} is not a supported method in safepy's polars dialect")

    def __len__(self):
        raise DisclosureError("len() on a frame is not allowed; use a count aggregation")

    def __repr__(self):
        return f"<SafePolarsFrame cols={self._pl.columns}>"
