"""SafeFrame — the STRICT-profile capability facade, pandas-shaped.

In STRICT mode the sandbox namespace contains no pandas and no raw frame — only
a ``SafeFrame`` wrapping the private data, plus the curated ``safe`` verbs. The
facade mirrors pandas' *call shapes* (``df['salary']``, ``df[df['age'] >= 18]``,
``df.groupby('sex')['salary'].mean()``) so familiar code runs, while the
disclosive verbs (``head``/``iloc``/``values``/``max``/``describe``) simply do
not exist.

The load-bearing invariant (see DESIGN.md):

    **Safe* types never reveal a value; only a Released aggregate exits.**

Shaping/intermediate operations (selection, masks, arithmetic, derived columns)
are non-disclosive, so the facade can be generous there. Disclosure is possible
only at *reduction*, and every reducer we expose (mean/sum/count/median/std/var,
value_counts, group aggregations, models) goes through ``min_n`` suppression —
which is sound because *we* own the method and therefore know the provenance and
the contributing count. Extremes/positional reducers are omitted entirely.

The raw objects live in ``_df`` / ``_s`` (private); the AST gate blocks any user
access to a ``_``-prefixed attribute, so user code can never reach them.
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd

from ._payload import frame_payload, series_payload
from .errors import DisclosureError
from .result import Released
from .safe import SafeVerbs
from .stats import _num

try:
    import protect
except ImportError:  # pragma: no cover
    protect = None

# ── assign() expression compiler (trusted, whitelisted — never eval) ──────────
_EXPR_FUNCS = {
    "log": np.log, "log10": np.log10, "exp": np.exp, "sqrt": np.sqrt,
    "abs": np.abs, "floor": np.floor, "ceil": np.ceil,
}
_EXPR_OPS = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
    ast.Mod: lambda a, b: a % b, ast.Pow: lambda a, b: a ** b,
}
_CMP_OPS = {
    ">": lambda a, b: a > b, "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
}
# whole-column reducers that are safe given a minimum contributing count
_SAFE_REDUCERS = frozenset({"mean", "sum", "count", "median", "std", "var"})


def _compile_expr(df: pd.DataFrame, expr: str):
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        raise DisclosureError(f"could not parse expression: {expr!r}")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in df.columns:
                raise DisclosureError(f"unknown column in expression: {node.id}")
            return df[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _EXPR_OPS:
            return _EXPR_OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in _EXPR_FUNCS and len(node.args) == 1 and not node.keywords:
            return _EXPR_FUNCS[node.func.id](ev(node.args[0]))
        raise DisclosureError("unsupported expression element in assign()")

    return ev(tree.body)


def _unwrap_val(x):
    """Unwrap a SafeColumn to its Series for use inside an operation."""
    return x._s if isinstance(x, SafeColumn) else x


def _order_stat(s, kind, k, *, q=None, winsorize=None):
    """Return ``(value_or_None, support)`` for an order statistic under the rule:
    releasable iff ``min(#<=v, #>=v) >= min_n`` (at least min_n observations at or
    beyond the value). Winsorizing pulls an extreme to a shared quantile bound."""
    if winsorize is not None and kind in ("min", "max"):
        p = float(winsorize)
        v = float(s.quantile(1 - p if kind == "max" else p))
    elif kind == "max":
        v = float(s.max())
    elif kind == "min":
        v = float(s.min())
    else:
        v = float(s.quantile(q))
    support = min(int((s <= v).sum()), int((s >= v).sum()))
    return (v if support >= k else None), support


def _nice_edges(s, bins):
    """Equal-width bin edges snapped to round numbers, so exact min/max are not
    revealed by the boundaries."""
    import math
    lo, hi = float(s.min()), float(s.max())
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return [lo, lo + 1]
    raw = (hi - lo) / max(int(bins), 1)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    step = next((m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag), 10 * mag)
    start = math.floor(lo / step) * step
    edges, x = [], start
    while x <= hi + step * 0.5:
        edges.append(round(x, 10))
        x += step
    return edges


class _ColumnPlot:
    """``SafeColumn.plot`` — only ``.hist()`` (a suppressed binned frequency) is
    available; other kinds would plot raw values, so they are refused."""

    def __init__(self, col):
        self._c = col

    def hist(self, bins=10, **kw):
        return self._c._hist(bins, kw.get("percent", False))

    def box(self, **kw):
        return self._c._box(kw.get("winsorize"))

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raise DisclosureError(
            f"plot.{name} on a raw column would reveal individual values; "
            "aggregate first (e.g. value_counts().plot.bar()), or use .hist()/.box()")


# ─────────────────────────────────────────────────────────────────────────────
class SafeColumn:
    """A column (pandas Series) that never reveals its values.

    Supports the non-disclosive surface — comparisons (→ boolean mask),
    arithmetic (→ derived column), membership/null tests, a small ``.dt``
    accessor — plus the safe reducers, which return a suppressed ``Released``
    scalar. It deliberately implements no ``__repr__``/``__iter__``/``values``/
    ``tolist``/``max``/``min``/``quantile``/``__getitem__``/scalar coercion.
    """

    _is_safecolumn = True

    def __init__(self, s: pd.Series, verbs: SafeVerbs):
        self._s = s
        self._verbs = verbs

    def _col(self, s) -> "SafeColumn":
        return SafeColumn(s, self._verbs)

    # -- comparisons -> boolean mask (a SafeColumn of dtype bool) --
    def __gt__(self, o): return self._col(self._s > _unwrap_val(o))
    def __ge__(self, o): return self._col(self._s >= _unwrap_val(o))
    def __lt__(self, o): return self._col(self._s < _unwrap_val(o))
    def __le__(self, o): return self._col(self._s <= _unwrap_val(o))
    def __eq__(self, o): return self._col(self._s == _unwrap_val(o))
    def __ne__(self, o): return self._col(self._s != _unwrap_val(o))
    __hash__ = None  # defining __eq__ makes columns unhashable; explicit

    # -- arithmetic -> derived column --
    def __add__(self, o): return self._col(self._s + _unwrap_val(o))
    def __sub__(self, o): return self._col(self._s - _unwrap_val(o))
    def __mul__(self, o): return self._col(self._s * _unwrap_val(o))
    def __truediv__(self, o): return self._col(self._s / _unwrap_val(o))
    def __mod__(self, o): return self._col(self._s % _unwrap_val(o))
    def __pow__(self, o): return self._col(self._s ** _unwrap_val(o))
    def __radd__(self, o): return self._col(_unwrap_val(o) + self._s)
    def __rsub__(self, o): return self._col(_unwrap_val(o) - self._s)
    def __rmul__(self, o): return self._col(_unwrap_val(o) * self._s)
    def __rtruediv__(self, o): return self._col(_unwrap_val(o) / self._s)
    def __neg__(self): return self._col(-self._s)

    # -- boolean combinators on masks --
    def __and__(self, o): return self._col(self._s & _unwrap_val(o))
    def __or__(self, o): return self._col(self._s | _unwrap_val(o))
    def __invert__(self): return self._col(~self._s)

    # -- membership / null tests -> mask --
    def isin(self, values): return self._col(self._s.isin(list(values)))
    def isna(self): return self._col(self._s.isna())
    def notna(self): return self._col(self._s.notna())
    def between(self, lo, hi): return self._col(self._s.between(lo, hi))

    # -- light transforms -> derived column --
    def astype(self, dtype): return self._col(self._s.astype(dtype))
    def round(self, n=0): return self._col(self._s.round(n))
    def clip(self, lower=None, upper=None): return self._col(self._s.clip(lower, upper))

    @property
    def dt(self): return _SafeDt(self._s, self._verbs)

    # -- safe reducers -> Released scalar (suppressed if support < min_n) --
    def mean(self): return self._reduce("mean")
    def sum(self): return self._reduce("sum")
    def count(self): return self._reduce("count")
    def median(self): return self._reduce("median")
    def std(self): return self._reduce("std")
    def var(self): return self._reduce("var")

    def _reduce(self, stat: str) -> Released:
        k = self._verbs._policy.min_n
        n = int(self._s.notna().sum())
        raw = self._s.count() if stat == "count" else getattr(self._s, stat)()
        suppressed = n < k
        value = None
        if not suppressed:
            value = _num(raw)
            rt = self._verbs._policy.round_to
            if rt is not None and value is not None:
                value = float(round(value / rt) * rt)
        return Released({"type": "scalar", "stat": stat, "value": value, "n": (None if suppressed else n)},
                        audit={"kind": "scalar", "verb": f"column.{stat}", "min_n": k,
                               "suppressed": suppressed, "backend": "pandas"})

    # -- order statistics: released only if >= min_n observations lie at/beyond
    #    the value (min(#<=v, #>=v) >= min_n). Median/quartiles pass; extremes
    #    pass only if shared/winsorized. See _order_stat.
    def _numeric(self):
        if not pd.api.types.is_numeric_dtype(self._s):
            raise DisclosureError(
                "min/max/describe/box are for numeric columns; use value_counts() "
                "for categories")
        return self._s.dropna()

    def max(self, *, winsorize=None): return self._extreme("max", winsorize)
    def min(self, *, winsorize=None): return self._extreme("min", winsorize)

    def _extreme(self, kind, winsorize):
        s = self._numeric()
        k = self._verbs._policy.min_n
        value, support = _order_stat(s, kind, k, winsorize=winsorize)
        return Released(
            {"type": "scalar", "stat": kind, "value": value,
             "n": support if value is not None else None},
            audit={"kind": "scalar", "verb": kind, "min_n": k, "support": support,
                   "winsorized": winsorize, "suppressed": value is None, "backend": "pandas"})

    def quantile(self, q, *, winsorize=None):
        s = self._numeric()
        k = self._verbs._policy.min_n
        value, support = _order_stat(s, "q", k, q=q, winsorize=winsorize)
        return Released(
            {"type": "scalar", "stat": f"q{q}", "value": value,
             "n": support if value is not None else None},
            audit={"kind": "scalar", "verb": "quantile", "q": q, "min_n": k,
                   "support": support, "suppressed": value is None, "backend": "pandas"})

    def describe(self, *, winsorize=None) -> Released:
        s = self._numeric()
        k = self._verbs._policy.min_n
        n = int(s.shape[0])
        agg = (lambda f: float(f) if n >= k else None)
        stats = {
            "count": n if n >= k else None,
            "mean": agg(s.mean()), "std": agg(s.std()),
            "min": _order_stat(s, "min", k, winsorize=winsorize)[0],
            "25%": _order_stat(s, "q", k, q=0.25)[0],
            "50%": _order_stat(s, "q", k, q=0.50)[0],
            "75%": _order_stat(s, "q", k, q=0.75)[0],
            "max": _order_stat(s, "max", k, winsorize=winsorize)[0],
        }
        return Released({"type": "describe", "name": str(self._s.name), "stats": stats},
                        audit={"kind": "table", "verb": "describe", "min_n": k,
                               "winsorized": winsorize, "backend": "pandas"})

    def boxplot(self, *, winsorize=None):
        return self._box(winsorize)

    def _box(self, winsorize):
        from .charts import chart_released
        s = self._numeric()
        k = self._verbs._policy.min_n
        stats = {
            "min": _order_stat(s, "min", k, winsorize=winsorize)[0],
            "q1": _order_stat(s, "q", k, q=0.25)[0],
            "median": _order_stat(s, "q", k, q=0.50)[0],
            "q3": _order_stat(s, "q", k, q=0.75)[0],
            "max": _order_stat(s, "max", k, winsorize=winsorize)[0],
        }
        return chart_released("box", {"type": "box", "name": str(self._s.name), "stats": stats},
                              {"verb": "boxplot", "min_n": k, "winsorized": winsorize,
                               "outliers": "omitted", "backend": "pandas"})

    # -- histogram: a raw-data plot, redirected to a suppressed binned frequency --
    def hist(self, bins=10, *, percent=False):
        return self._hist(bins, percent)

    @property
    def plot(self):
        return _ColumnPlot(self)

    def _hist(self, bins, percent):
        from .charts import chart_released
        s = self._s.dropna()
        k = self._verbs._policy.min_n
        edges = list(bins) if isinstance(bins, (list, tuple)) else _nice_edges(s, int(bins))
        cats = pd.cut(s, bins=edges, include_lowest=True)
        counts = cats.value_counts().sort_index()
        total = int(counts.sum())
        idx = [str(i) for i in counts.index]
        vals, suppressed = [], 0
        for v in counts.to_numpy():
            v = int(v)
            if v < k:
                vals.append(None); suppressed += 1
            elif percent:
                vals.append(round(100.0 * v / total, 2) if total else None)
            else:
                vals.append(v)
        data = {"type": "series", "name": "percent" if percent else "count",
                "index": idx, "values": vals}
        return chart_released("hist", data, {"verb": "hist", "min_n": k,
                              "bins": len(idx), "bins_suppressed": suppressed,
                              "backend": "pandas"})

    # -- frequency table of this column -> Released table --
    def value_counts(self, *, min_n=None, round=None) -> Released:
        if protect is None:
            raise DisclosureError("the 'protect' package is required")
        k = self._verbs._min_n(min_n)
        counts = self._s.value_counts()
        safe = protect.suppress(counts, counts=counts, min_n=k, round=self._verbs._round(round))
        return Released(series_payload(safe, name=f"count({self._s.name})"), audit={
            "kind": "table", "verb": "value_counts", "min_n": k,
            "cells_suppressed": int((counts < k).sum()), "backend": "pandas"})

    # -- guards against value leakage / ambiguous coercion --
    def __bool__(self):
        raise DisclosureError("a column has no single truth value; build a mask and filter instead")

    def __len__(self):
        raise DisclosureError("len() on a column is not allowed; use .count()")

    def __repr__(self):
        return f"<SafeColumn name={self._s.name!r}>"


class _SafeDt:
    """A minimal ``.dt`` accessor: returns derived SafeColumns, never values."""

    def __init__(self, s: pd.Series, verbs: SafeVerbs):
        self._s = pd.to_datetime(s)
        self._verbs = verbs

    def _part(self, name):
        return SafeColumn(getattr(self._s.dt, name), self._verbs)

    @property
    def year(self): return self._part("year")
    @property
    def month(self): return self._part("month")
    @property
    def day(self): return self._part("day")
    @property
    def quarter(self): return self._part("quarter")
    @property
    def dayofweek(self): return self._part("dayofweek")
    @property
    def hour(self): return self._part("hour")


# ─────────────────────────────────────────────────────────────────────────────
class SafeSeriesGroupBy:
    """``df.groupby(by)[value]`` — only aggregations, each a suppressed table."""

    def __init__(self, df, by, value, verbs: SafeVerbs):
        self._df, self._by, self._value, self._verbs = df, by, value, verbs

    def mean(self, **kw): return self._agg("mean", **kw)
    def sum(self, **kw): return self._agg("sum", **kw)
    def count(self, **kw): return self._agg("count", **kw)
    def median(self, **kw): return self._agg("median", **kw)
    def std(self, **kw): return self._agg("std", **kw)
    def var(self, **kw): return self._agg("var", **kw)
    def size(self, **kw): return self._agg("size", **kw)

    def _agg(self, agg, **kw):
        return self._verbs.group_agg(self._df, self._by, self._value, agg, **kw)


class SafeGroupBy:
    """``df.groupby(by)``. Index a column to get the pandas chaining shape
    ``groupby(by)[value].mean()``; the legacy ``groupby(by).mean(value)`` shape
    is also accepted."""

    def __init__(self, df, by, verbs: SafeVerbs):
        self._df, self._by, self._verbs = df, by, verbs

    def __getitem__(self, value):
        if not isinstance(value, str):
            raise DisclosureError("select a single column by name, e.g. groupby(...)['salary']")
        return SafeSeriesGroupBy(self._df, self._by, value, self._verbs)

    def __getattr__(self, name):
        # df.groupby('sex').salary  ==  df.groupby('sex')['salary']
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._df.columns:
            return SafeSeriesGroupBy(self._df, self._by, name, self._verbs)
        raise DisclosureError(f"{name!r} is not a column")

    # pandas-shaped chaining handles aggregation via __getitem__; these support
    # the legacy explicit-value shape groupby(by).mean('salary').
    def mean(self, value, **kw): return self._verbs.group_agg(self._df, self._by, value, "mean", **kw)
    def sum(self, value, **kw): return self._verbs.group_agg(self._df, self._by, value, "sum", **kw)
    def count(self, value, **kw): return self._verbs.group_agg(self._df, self._by, value, "count", **kw)
    def median(self, value, **kw): return self._verbs.group_agg(self._df, self._by, value, "median", **kw)
    def std(self, value, **kw): return self._verbs.group_agg(self._df, self._by, value, "std", **kw)
    def var(self, value, **kw): return self._verbs.group_agg(self._df, self._by, value, "var", **kw)

    def size(self, **kw):
        col = self._by if isinstance(self._by, str) else self._by[0]
        return self._verbs.group_agg(self._df, self._by, col, "size", **kw)


# ─────────────────────────────────────────────────────────────────────────────
class SafeFrame:
    """The only data object user code can touch in STRICT mode."""

    _is_safeframe = True

    def __init__(self, df: pd.DataFrame, verbs: SafeVerbs):
        self._df = df
        self._verbs = verbs

    # -- selection / masking (pandas shapes) --
    def __getitem__(self, key):
        if isinstance(key, str):
            if key not in self._df.columns:
                raise DisclosureError(f"unknown column: {key}")
            return SafeColumn(self._df[key], self._verbs)
        if isinstance(key, list):
            for c in key:
                if not isinstance(c, str) or c not in self._df.columns:
                    raise DisclosureError(f"unknown column: {c}")
            return SafeFrame(self._df[key], self._verbs)
        if isinstance(key, SafeColumn):
            mask = key._s
            if mask.dtype != bool:
                raise DisclosureError("can only index a frame with a boolean mask")
            return SafeFrame(self._df[mask], self._verbs)
        raise DisclosureError("unsupported index; use a column name, list of names, or boolean mask")

    def __getattr__(self, name):
        # attribute column access: df.salary == df['salary']. Methods are found
        # by normal lookup first, so they always win over a same-named column.
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._df.columns:
            return SafeColumn(self._df[name], self._verbs)
        raise DisclosureError(
            f"{name!r} is not a column or supported method (a column named like a "
            "method must be accessed as df['{0}'])".format(name))

    # -- shaping verbs: return another SafeFrame --
    def where(self, col: str, op: str, value) -> "SafeFrame":
        if op not in _CMP_OPS:
            raise DisclosureError(f"unknown comparison operator: {op!r}")
        if col not in self._df.columns:
            raise DisclosureError(f"unknown column: {col}")
        mask = _CMP_OPS[op](self._df[col], value)
        return SafeFrame(self._df[mask], self._verbs)

    def rename(self, columns: dict) -> "SafeFrame":
        if not isinstance(columns, dict):
            raise DisclosureError("rename needs columns={old: new}")
        for c in columns:
            if c not in self._df.columns:
                raise DisclosureError(f"unknown column: {c}")
        return SafeFrame(self._df.rename(columns=columns), self._verbs)

    def fillna(self, value) -> "SafeFrame":
        return SafeFrame(self._df.fillna(value), self._verbs)

    def dropna(self, subset=None) -> "SafeFrame":
        if subset is not None:
            for c in subset:
                if c not in self._df.columns:
                    raise DisclosureError(f"unknown column: {c}")
        return SafeFrame(self._df.dropna(subset=subset), self._verbs)

    def drop(self, columns) -> "SafeFrame":
        cols = [columns] if isinstance(columns, str) else list(columns)
        for c in cols:
            if c not in self._df.columns:
                raise DisclosureError(f"unknown column: {c}")
        return SafeFrame(self._df.drop(columns=cols), self._verbs)

    def assign(self, *args, **kwargs) -> "SafeFrame":
        """Add derived columns. Two forms:

          df.assign(logwage=np.log(df['wage']))   # natural pandas: name=SafeColumn
          df.assign('logwage', 'log(wage)')        # legacy: whitelisted expr string
        """
        df = self._df
        if args:
            name, expr = args
            df = df.assign(**{name: _compile_expr(self._df, expr)})
        for name, val in kwargs.items():
            series = val._s if isinstance(val, SafeColumn) else val
            df = df.assign(**{name: series})
        return SafeFrame(df, self._verbs)

    # -- terminal verbs: return a suppressed Released aggregate --
    def groupby(self, by) -> SafeGroupBy:
        return SafeGroupBy(self._df, by, self._verbs)

    def value_counts(self, col: str, **kw) -> Released:
        return self._verbs.value_counts(self._df, col, **kw)

    def crosstab(self, row: str, col: str, **kw) -> Released:
        return self._verbs.crosstab(self._df, row, col, **kw)

    def corr(self, **kw) -> Released:
        """Correlation matrix over numeric columns (aggregate; released only if
        the frame has at least ``min_n`` rows)."""
        num = self._df.select_dtypes("number")
        if num.shape[1] < 2:
            raise DisclosureError("corr needs at least two numeric columns")
        k = self._verbs._policy.min_n
        if len(self._df) < k:
            raise DisclosureError("too few rows to release a correlation matrix")
        c = num.corr().round(3)
        return Released(frame_payload(c), audit={
            "kind": "table", "verb": "corr", "min_n": k, "backend": "pandas"})

    # -- regression / survival: delegate to the safe-verb library --
    def ols(self, *, y, x, **kw) -> Released:
        return self._verbs.ols(self._df, y=y, x=x, **kw)

    def logit(self, *, y, x, **kw) -> Released:
        return self._verbs.logit(self._df, y=y, x=x, **kw)

    def poisson(self, *, y, x, **kw) -> Released:
        return self._verbs.poisson(self._df, y=y, x=x, **kw)

    def cox(self, *, duration, event, x, **kw) -> Released:
        return self._verbs.cox(self._df, duration=duration, event=event, x=x, **kw)

    def kaplan_meier(self, *, duration, event, by=None, **kw) -> Released:
        return self._verbs.kaplan_meier(self._df, duration=duration, event=event, by=by, **kw)

    def logrank(self, *, duration, event, by, **kw) -> Released:
        return self._verbs.logrank(self._df, duration=duration, event=event, by=by, **kw)

    def weibull_aft(self, *, duration, event, x, **kw) -> Released:
        return self._verbs.weibull_aft(self._df, duration=duration, event=event, x=x, **kw)

    def lognormal_aft(self, *, duration, event, x, **kw) -> Released:
        return self._verbs.lognormal_aft(self._df, duration=duration, event=event, x=x, **kw)

    def loglogistic_aft(self, *, duration, event, x, **kw) -> Released:
        return self._verbs.loglogistic_aft(self._df, duration=duration, event=event, x=x, **kw)

    def rmst(self, *, duration, event, t, by=None, **kw) -> Released:
        return self._verbs.rmst(self._df, duration=duration, event=event, t=t, by=by, **kw)

    def feols(self, fml=None, *, y=None, x=None, fe=None, cluster=None,
              vcov=None, data=None, **kw):
        # library-faithful string form: df.feols("y ~ x | fe").summary()
        if isinstance(fml, str):
            from .pyfixest_api import SafePyfixest
            v = vcov if vcov is not None else ({"CRV1": cluster} if cluster else None)
            return SafePyfixest().feols(fml, self, vcov=v)
        # keyword convenience form -> a released coefficient table
        if y is None:
            raise DisclosureError("feols needs a formula string or y=/x=")
        return self._verbs.feols(self._df, y=y, x=x, fe=fe, cluster=cluster)

    def iv(self, *, y, x=None, endog, instruments, fe=None, cluster=None, **kw) -> Released:
        return self._verbs.iv(self._df, y=y, x=x, endog=endog, instruments=instruments,
                              fe=fe, cluster=cluster, **kw)

    def ate(self, *, outcome, treatment, confounders, method="weighting", **kw) -> Released:
        return self._verbs.ate(self._df, outcome=outcome, treatment=treatment,
                               confounders=confounders, method=method, **kw)

    def __len__(self):
        raise DisclosureError("len() on a frame is not allowed; use a count aggregation")

    def __repr__(self):
        return f"<SafeFrame cols={list(self._df.columns)}>"

    # deliberately NO __iter__, to_*, head, values, iloc, describe, ...
