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
        by = self._by[0] if len(self._by) == 1 else self._by
        return self._verbs.group_agg(self._pl.to_pandas(), by, self._by[0], "size")

    def agg(self, *exprs):
        if not exprs:
            raise DisclosureError("agg needs at least one reducer, e.g. pl.col('salary').mean()")
        rels = [self._one_agg(e) for e in exprs]
        if len(rels) == 1:
            return rels[0]
        return self._combine(rels)

    def _one_agg(self, e) -> Released:
        """One reducer -> a suppressed series Released, routed through SafeVerbs."""
        if not isinstance(e, SafeExpr) or e._agg is None:
            raise DisclosureError(
                "agg needs reducers over columns, e.g. pl.col('salary').mean()")
        by = self._by[0] if len(self._by) == 1 else self._by
        if e._col is not None and e._name is None:   # simple column: fast path
            return self._verbs.group_agg(self._pl.to_pandas(), by, e._col, e._agg)
        # derived expression, or an aliased column: materialize a value column so
        # the released name follows the alias.
        name = e._name or "value"
        base = e._pre if e._col is None else pl.col(e._col)
        pdf = self._pl.with_columns(base.alias(name)).to_pandas()
        rel = self._verbs.group_agg(pdf, by, name, e._agg)
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
                   "stats": columns, "backend": "pandas"})


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
            if e not in self._pl.columns:
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
        if e._col is not None and e._name is None:   # simple column: fast path
            series = self._pl.to_pandas()[e._col]
        else:                                        # derived / aliased: materialize it
            base = e._pre if e._col is None else pl.col(e._col)
            series = self._pl.select(base.alias(e._name or "value")).to_pandas().iloc[:, 0]
        return getattr(SafeColumn(series, self._verbs), e._agg)()

    def with_columns(self, *exprs, **named) -> "SafePolarsFrame":
        """Add derived columns (polars analog of pandas ``assign``). Returns a
        private SafePolarsFrame; the derived values exit only via a suppressed
        aggregate, so derivation is safe by construction."""
        out = [self._as_expr(e)._expr for e in exprs]
        out += [self._as_expr(e)._expr.alias(name) for name, e in named.items()]
        return SafePolarsFrame(self._pl.with_columns(*out), self._verbs)

    def group_by(self, *by) -> SafePolarsGroupBy:
        cols = [c for grp in by for c in ([grp] if isinstance(grp, str) else grp)]
        for c in cols:
            if c not in self._pl.columns:
                raise DisclosureError(f"unknown column: {c}")
        return SafePolarsGroupBy(self._pl, cols, self._verbs)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raise DisclosureError(
            f"{name!r} is not a supported method in safepy's polars dialect")

    def __len__(self):
        raise DisclosureError("len() on a frame is not allowed; use a count aggregation")

    def __repr__(self):
        return f"<SafePolarsFrame cols={self._pl.columns}>"
