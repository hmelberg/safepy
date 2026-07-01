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
from .safe import SafeVerbs, _stop_if_too_sparse
from .stats import _num

try:
    import protect
except ImportError:  # pragma: no cover
    protect = None

# ── assign() expression compiler (trusted, whitelisted — never eval) ──────────
# Mirrors the SafeColumn/np/str surface so microdata-style string expressions
# (assign('y', 'log(x)'), assign('c', 'substr(muni, 0, 2)')) match the pandas
# path. Every element is element-wise -> a column; nothing aggregates or reveals.
def _f_where(c, a, b):
    idx = getattr(c, "index", None)
    idx = idx if idx is not None else getattr(a, "index", getattr(b, "index", None))
    val = lambda v: v.to_numpy() if hasattr(v, "to_numpy") else v
    return pd.Series(np.where(val(c), val(a), val(b)), index=idx)


_EXPR_CALLS = {
    # unary math
    **{n: (lambda f: (lambda a: f(a)))(getattr(np, m)) for n, m in {
        "log": "log", "log2": "log2", "log10": "log10", "log1p": "log1p",
        "exp": "exp", "expm1": "expm1", "sqrt": "sqrt", "cbrt": "cbrt",
        "square": "square", "abs": "abs", "sign": "sign", "floor": "floor",
        "ceil": "ceil", "trunc": "trunc", "rint": "rint", "sin": "sin",
        "cos": "cos", "tan": "tan", "sinh": "sinh", "cosh": "cosh",
        "tanh": "tanh", "radians": "radians", "degrees": "degrees",
    }.items()},
    # multi-arg math
    "minimum": lambda a, b: np.minimum(a, b),
    "maximum": lambda a, b: np.maximum(a, b),
    "power": lambda a, b: np.power(a, b),
    "mod": lambda a, b: np.mod(a, b),
    "round": lambda a, n=0: np.round(a, int(n)),
    "where": _f_where,
    # string functions (element-wise)
    "substr": lambda s, start, length: s.str.slice(int(start), int(start) + int(length)),
    "upper": lambda s: s.str.upper(),
    "lower": lambda s: s.str.lower(),
    "title": lambda s: s.str.title(),
    "strip": lambda s: s.str.strip(),
    "strlen": lambda s: s.str.len(),
    "zfill": lambda s, w: s.str.zfill(int(w)),
    "replace": lambda s, a, b: s.str.replace(a, b, regex=False),
    "concat": lambda a, b: a.astype(str).str.cat(b.astype(str)),
}
_EXPR_OPS = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
    ast.Mod: lambda a, b: a % b, ast.Pow: lambda a, b: a ** b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.BitAnd: lambda a, b: a & b, ast.BitOr: lambda a, b: a | b,
}
_EXPR_CMP = {
    ast.Gt: lambda a, b: a > b, ast.Lt: lambda a, b: a < b,
    ast.GtE: lambda a, b: a >= b, ast.LtE: lambda a, b: a <= b,
    ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b,
}
_CMP_OPS = {
    ">": lambda a, b: a > b, "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
}
# whole-column reducers that are safe given a minimum contributing count
_SAFE_REDUCERS = frozenset({"mean", "sum", "count", "median", "std", "var"})
# aggregations allowed when *building a derived dataset* via summarise (extremes
# excluded — the derived frame is private, but this keeps derived columns tame)
_SUMMARISE_FUNCS = frozenset({"mean", "sum", "count", "size", "median", "std", "var"})


def _compile_expr(df: pd.DataFrame, expr: str):
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        raise DisclosureError(f"could not parse expression: {expr!r}")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str, bool)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in df.columns:
                raise DisclosureError(f"unknown column in expression: {node.id}")
            return df[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _EXPR_OPS:
            return _EXPR_OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _EXPR_CMP:
            return _EXPR_CMP[type(node.ops[0])](ev(node.left), ev(node.comparators[0]))
        if isinstance(node, ast.BoolOp):
            vals = [ev(v) for v in node.values]
            acc = vals[0]
            for v in vals[1:]:
                acc = (acc & v) if isinstance(node.op, ast.And) else (acc | v)
            return acc
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -ev(node.operand)
            if isinstance(node.op, (ast.Invert, ast.Not)):
                return ~ev(node.operand)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            fn = _EXPR_CALLS.get(node.func.id)
            if fn is None:
                raise DisclosureError(f"function '{node.func.id}' is not available in assign()")
            try:
                return fn(*[ev(a) for a in node.args])
            except (TypeError, AttributeError):
                raise DisclosureError(f"invalid arguments to '{node.func.id}' in assign()")
        raise DisclosureError("unsupported expression element in assign()")

    return ev(tree.body)


def _unwrap_val(x):
    """Unwrap a SafeColumn to its Series for use inside an operation."""
    return x._s if isinstance(x, SafeColumn) else x


def _check_recode(to_replace, value):
    """Guard ``replace`` inputs: allow scalars, lists, and {old: new} mappings,
    but never a callable (which would be arbitrary code) or a regex pattern.
    Recoding is a value→value relabel, so it stays non-disclosive."""
    def walk(v):
        if callable(v):
            raise DisclosureError(
                "replace does not accept functions; pass a value or a {old: new} mapping")
        if isinstance(v, dict):
            for k, sub in v.items():
                walk(k); walk(sub)
        elif isinstance(v, (list, tuple)):
            for sub in v:
                walk(sub)
    walk(to_replace)
    walk(value)


def _check_edit(old: pd.Series, new: pd.Series, verbs):
    """Tiltak 6: an edit (recode/replace/fill/clip/where) may not change a number
    of units in ``[1, k)`` or ``(n-k, n)`` — changing a tiny group, or all but a
    tiny group, could single individuals out. Changing all rows or none is fine.
    The message never states the affected count (that count is itself data)."""
    k = verbs._policy.suppression.min_edit_units
    if not k:
        return
    o, nw = old.to_numpy(), new.to_numpy()
    same = (o == nw) | (pd.isna(o) & pd.isna(nw))
    changed = int((~same).sum())
    n = len(o)
    if changed == 0 or changed == n:
        return
    if changed < k or changed > n - k:
        raise DisclosureError(
            f"this edit changes fewer than {k} units (or all but fewer than {k}); "
            "such edits are not allowed because they can single individuals out. "
            "Change all rows, none, or at least this many.")


def _check_frame_edit(old_df: pd.DataFrame, new_df: pd.DataFrame, verbs):
    """Tiltak 6 for a frame-wide recode: applied column by column (each recode
    term must satisfy the rule independently)."""
    if not verbs._policy.suppression.min_edit_units:
        return
    for c in new_df.columns:
        if c in old_df.columns:
            _check_edit(old_df[c], new_df[c], verbs)


def _sig_round(v, sig_figs):
    """Round to ``sig_figs`` significant figures (Tiltak 8: percentile/median
    values are actual individual values, so they are shown coarsely)."""
    import math
    if v is None or sig_figs is None or v == 0 or not math.isfinite(v):
        return v
    d = int(sig_figs) - int(math.floor(math.log10(abs(v)))) - 1
    return round(v, d)


def _descriptive_k(policy):
    """The suppression threshold for *descriptive* statistics (mean/std/percentile
    /skew...): the primary ``min_n`` raised by the Tiltak 7 descriptive-population
    floor and the Tiltak 1 minimum-population floor. Counts/sums use plain min_n."""
    s = policy.suppression
    return max(policy.min_n, s.min_descriptive_n or 0, s.min_population or 0)


# Moment/magnitude stats that winsorization (Tiltak 2) affects. Order statistics
# (median/quartiles) are deliberately excluded — they are not winsorized.
_WINSOR_STATS = frozenset({"mean", "std", "var", "sum", "sem", "skew", "kurt"})


def _winsor_p(policy):
    """The single tail probability for order-stat winsorization, or None. The
    policy stores (low, high) percentiles; the lower one is the symmetric p."""
    w = policy.suppression.winsorize
    return None if w is None else float(w[0])


def _winsorized_series(s, policy):
    """Return ``s`` with its tails capped per the policy (Tiltak 2), using
    ``protect.winsorize``; unchanged if winsorization is off or the column is not
    numeric (categorical variables are never winsorized)."""
    w = policy.suppression.winsorize
    if (w is None or protect is None or not pd.api.types.is_numeric_dtype(s)
            or pd.api.types.is_bool_dtype(s)):     # bools/indicators are not winsorized
        return s
    capped = protect.winsorize(s.to_frame("v"), "v", limits=(float(w[0]), float(w[1])))
    return capped["v"]


def _order_stat(s, kind, k, *, q=None, winsorize=None, sig_figs=None):
    """Return ``(value_or_None, support)`` for an order statistic under the rule:
    releasable iff ``min(#<=v, #>=v) >= min_n`` (at least min_n observations at or
    beyond the value). Winsorizing pulls an extreme to a shared quantile bound;
    ``sig_figs`` (Tiltak 8) coarsens percentile/median values before release."""
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
    if support < k:
        return None, support
    return _sig_round(v, sig_figs), support


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
    def clip(self, lower=None, upper=None): return self._edit(self._s.clip(lower, upper))
    def abs(self): return self._col(self._s.abs())
    def fillna(self, value): return self._edit(self._s.fillna(_unwrap_val(value)))

    def _edit(self, new_s) -> "SafeColumn":
        """Return a recoded column after the Tiltak 6 edit-size check."""
        _check_edit(self._s, new_s, self._verbs)
        return self._col(new_s)

    def where(self, cond, other=np.nan):
        """Keep values where ``cond`` is True, else ``other`` — the idiomatic
        conditional recode. ``cond`` is a boolean SafeColumn."""
        if not isinstance(cond, SafeColumn):
            raise DisclosureError("where needs a boolean column as its condition")
        return self._edit(self._s.where(cond._s, _unwrap_val(other)))

    def mask(self, cond, other=np.nan):
        """Inverse of ``where``: replace where ``cond`` is True."""
        if not isinstance(cond, SafeColumn):
            raise DisclosureError("mask needs a boolean column as its condition")
        return self._edit(self._s.mask(cond._s, _unwrap_val(other)))

    def replace(self, to_replace, value=None):
        """Recode values: replace({'M': 'Male'}) or replace('M', 'Male'). A
        value→value relabel, so the column stays private."""
        _check_recode(to_replace, value)
        s = self._s.replace(to_replace) if value is None else self._s.replace(to_replace, value)
        return self._edit(s)

    def map(self, mapping):
        """Recode via a {old: new} mapping (like replace, but unmatched → NaN).
        Only a dict is accepted — a callable would be arbitrary code."""
        if not isinstance(mapping, dict):
            raise DisclosureError(
                "map takes a {old: new} dict (a function would be arbitrary code); "
                "use replace/where for other recoding")
        return self._edit(self._s.map(mapping))

    # NB: no rank() — it is an order-statistic differencing primitive (rank +
    # filter + sum isolates one value-ordered individual across two above-min_n
    # queries) and is gate-denied until the multi-query audit layer exists.

    # -- per-row derived columns (order-dependent; sort_values first for panels).
    #    Each returns a private SafeColumn -> released only via a suppressed
    #    aggregation, so no new disclosure path. --
    def shift(self, periods=1): return self._col(self._s.shift(periods))

    def diff(self, periods=1): return self._col(self._s.diff(periods))

    def _num_series(self, verb):
        if not pd.api.types.is_numeric_dtype(self._s):
            raise DisclosureError(f"{verb} is for numeric columns")
        return self._s

    def cumsum(self): return self._col(self._num_series("cumsum").cumsum())
    def cumprod(self): return self._col(self._num_series("cumprod").cumprod())
    def cummax(self): return self._col(self._num_series("cummax").cummax())
    def cummin(self): return self._col(self._num_series("cummin").cummin())
    def pct_change(self, periods=1):
        return self._col(self._num_series("pct_change").pct_change(periods))
    def ffill(self): return self._col(self._s.ffill())
    def bfill(self): return self._col(self._s.bfill())

    def interpolate(self, method="linear"):
        if not isinstance(method, str):
            raise DisclosureError("interpolate takes a method name, not a function")
        return self._col(self._num_series("interpolate").interpolate(method=method))

    @property
    def dt(self): return _SafeDt(self._s, self._verbs)

    @property
    def str(self):
        if pd.api.types.is_numeric_dtype(self._s):
            raise DisclosureError(".str requires a text column; this column is numeric")
        return _SafeStr(self._s, self._verbs)

    # -- safe reducers -> Released scalar (suppressed if support < min_n) --
    def mean(self): return self._reduce("mean")
    def sum(self): return self._reduce("sum")
    def count(self): return self._reduce("count")
    def median(self): return self._reduce("median")
    def std(self): return self._reduce("std")
    def var(self): return self._reduce("var")
    # shape / precision stats — dimensionless, so round_to (a magnitude coarsener)
    # is not applied; min_n suppression still holds.
    def sem(self): return self._reduce("sem", apply_round=False)
    def skew(self): return self._reduce("skew", apply_round=False)
    def kurt(self): return self._reduce("kurt", apply_round=False)
    kurtosis = kurt

    def nunique(self) -> Released:
        """Count of distinct values — an aggregate, suppressed if the column has
        fewer than ``min_n`` non-null rows."""
        k = self._verbs._policy.min_n
        n = int(self._s.notna().sum())
        suppressed = n < k
        return Released(
            {"type": "scalar", "stat": "nunique",
             "value": None if suppressed else int(self._s.nunique()),
             "n": None if suppressed else n},
            audit={"kind": "scalar", "verb": "column.nunique", "min_n": k,
                   "suppressed": suppressed, "backend": "pandas"})

    def _reduce(self, stat: str, *, apply_round=True) -> Released:
        policy = self._verbs._policy
        # counts/sums use the primary min_n; descriptive stats get the higher
        # descriptive-population floor (Tiltak 7/1).
        k = policy.min_n if stat in ("count", "sum") else _descriptive_k(policy)
        n = int(self._s.notna().sum())
        # Tiltak 2: moment stats are computed on the winsorized column.
        src = _winsorized_series(self._s, policy) if stat in _WINSOR_STATS else self._s
        raw = src.count() if stat == "count" else getattr(src, stat)()
        suppressed = n < k
        value = None
        if not suppressed:
            value = _num(raw)
            sf = policy.suppression.percentile_sig_figs
            rt = policy.round_to
            if stat == "median" and sf:            # Tiltak 8: percentile precision
                value = _sig_round(value, sf)
            elif apply_round and rt is not None and value is not None:
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
        k = _descriptive_k(self._verbs._policy)
        sf = self._verbs._policy.suppression.percentile_sig_figs
        w = winsorize if winsorize is not None else _winsor_p(self._verbs._policy)
        value, support = _order_stat(s, kind, k, winsorize=w, sig_figs=sf)
        return Released(
            {"type": "scalar", "stat": kind, "value": value,
             "n": support if value is not None else None},
            audit={"kind": "scalar", "verb": kind, "min_n": k, "support": support,
                   "winsorized": winsorize, "suppressed": value is None, "backend": "pandas"})

    def quantile(self, q, *, winsorize=None):
        s = self._numeric()
        k = _descriptive_k(self._verbs._policy)
        sf = self._verbs._policy.suppression.percentile_sig_figs
        value, support = _order_stat(s, "q", k, q=q, winsorize=winsorize, sig_figs=sf)
        return Released(
            {"type": "scalar", "stat": f"q{q}", "value": value,
             "n": support if value is not None else None},
            audit={"kind": "scalar", "verb": "quantile", "q": q, "min_n": k,
                   "support": support, "suppressed": value is None, "backend": "pandas"})

    def describe(self, *, winsorize=None) -> Released:
        s = self._numeric()
        policy = self._verbs._policy
        k = _descriptive_k(policy)
        sf = policy.suppression.percentile_sig_figs
        w = winsorize if winsorize is not None else _winsor_p(policy)   # Tiltak 2 default
        sw = _winsorized_series(s, policy)                              # for mean/std
        n = int(s.shape[0])
        agg = (lambda f: float(f) if n >= k else None)      # mean/std: not sig-rounded
        stats = {
            "count": n if n >= policy.min_n else None,
            "mean": agg(sw.mean()), "std": agg(sw.std()),
            "min": _order_stat(s, "min", k, winsorize=w, sig_figs=sf)[0],
            "25%": _order_stat(s, "q", k, q=0.25, sig_figs=sf)[0],
            "50%": _order_stat(s, "q", k, q=0.50, sig_figs=sf)[0],
            "75%": _order_stat(s, "q", k, q=0.75, sig_figs=sf)[0],
            "max": _order_stat(s, "max", k, winsorize=w, sig_figs=sf)[0],
        }
        return Released({"type": "describe", "name": str(self._s.name), "stats": stats},
                        audit={"kind": "table", "verb": "describe", "min_n": k,
                               "winsorized": winsorize, "backend": "pandas"})

    def boxplot(self, *, winsorize=None):
        return self._box(winsorize)

    def _box(self, winsorize):
        from .charts import chart_released
        s = self._numeric()
        k = _descriptive_k(self._verbs._policy)
        sf = self._verbs._policy.suppression.percentile_sig_figs
        w = winsorize if winsorize is not None else _winsor_p(self._verbs._policy)
        stats = {
            "min": _order_stat(s, "min", k, winsorize=w, sig_figs=sf)[0],
            "q1": _order_stat(s, "q", k, q=0.25, sig_figs=sf)[0],
            "median": _order_stat(s, "q", k, q=0.50, sig_figs=sf)[0],
            "q3": _order_stat(s, "q", k, q=0.75, sig_figs=sf)[0],
            "max": _order_stat(s, "max", k, winsorize=w, sig_figs=sf)[0],
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
        # Tiltak 2: winsorize before binning so extreme values fold into the tail
        # bins rather than revealing themselves as lone points.
        s = _winsorized_series(self._s.dropna(), self._verbs._policy)
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
        _stop_if_too_sparse(counts.to_numpy(), self._verbs._policy)
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
        # keep datetime/timedelta as-is (so `d1 - d2` -> .dt.days works); coerce
        # strings to datetime.
        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s):
            self._s = s
        else:
            self._s = pd.to_datetime(s)
        self._verbs = verbs

    def _part(self, name):
        return SafeColumn(getattr(self._s.dt, name), self._verbs)

    @property
    def days(self): return self._part("days")             # for a timedelta column

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
    def weekday(self): return self._part("dayofweek")
    @property
    def dayofyear(self): return self._part("dayofyear")
    @property
    def days_in_month(self): return self._part("days_in_month")
    @property
    def is_month_start(self): return self._part("is_month_start")
    @property
    def is_month_end(self): return self._part("is_month_end")
    @property
    def hour(self): return self._part("hour")
    @property
    def minute(self): return self._part("minute")
    @property
    def second(self): return self._part("second")

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raise DisclosureError(f"dt.{name} is not available in safepy")


class _SafeStr:
    """A ``.str`` accessor mirroring pandas string functions. Element-wise, so
    every method returns a private SafeColumn (or a boolean mask). Aggregating
    forms (e.g. ``.str.cat()`` with no other column, which joins all rows into
    one string) are not exposed; anything outside the whitelist is refused."""

    def __init__(self, s: pd.Series, verbs: SafeVerbs):
        self._s = s
        self._verbs = verbs

    def _col(self, s):
        return SafeColumn(s, self._verbs)

    # -- substring / indexing --
    def slice(self, start=None, stop=None, step=None):
        return self._col(self._s.str.slice(start, stop, step))

    def substr(self, start, length):
        """microdata-style: `length` chars from position `start` (0-indexed)."""
        return self._col(self._s.str.slice(start, start + length))

    # -- case / trim --
    def upper(self): return self._col(self._s.str.upper())
    def lower(self): return self._col(self._s.str.lower())
    def title(self): return self._col(self._s.str.title())
    def capitalize(self): return self._col(self._s.str.capitalize())
    def strip(self, chars=None): return self._col(self._s.str.strip(chars))
    def lstrip(self, chars=None): return self._col(self._s.str.lstrip(chars))
    def rstrip(self, chars=None): return self._col(self._s.str.rstrip(chars))

    # -- length / pad --
    def len(self): return self._col(self._s.str.len())
    def pad(self, width, side="left", fillchar=" "):
        return self._col(self._s.str.pad(width, side=side, fillchar=fillchar))
    def zfill(self, width): return self._col(self._s.str.zfill(width))

    # -- replace / concat --
    def replace(self, pat, repl, *, regex=False):
        return self._col(self._s.str.replace(pat, repl, regex=bool(regex)))

    def cat(self, other, *, sep=""):
        # element-wise (row-wise) concat only; joining all rows would disclose.
        if not isinstance(other, SafeColumn):
            raise DisclosureError("str.cat needs another column to concatenate row-wise")
        return self._col(self._s.str.cat(other._s, sep=sep))

    # -- search -> boolean mask (regex off by default: literal, and ReDoS-safe) --
    def contains(self, pat, *, regex=False):
        return self._col(self._s.str.contains(pat, regex=bool(regex), na=False))
    def startswith(self, pat): return self._col(self._s.str.startswith(pat, na=False))
    def endswith(self, pat): return self._col(self._s.str.endswith(pat, na=False))
    def find(self, sub): return self._col(self._s.str.find(sub))

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raise DisclosureError(f"str.{name} is not available in safepy")


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

    def describe(self, **kw):
        return self._verbs.group_describe(self._df, self._by, self._value)

    def agg(self, func, **kw):
        """``groupby(by)[value].agg('mean')`` or ``.agg(['mean', 'std'])``. Only
        stat *names* are accepted (a callable would be arbitrary code); each cell
        is suppressed on its group count like every other aggregation."""
        if callable(func) or (isinstance(func, (list, tuple)) and any(callable(f) for f in func)):
            raise DisclosureError(
                "agg takes a stat name like 'mean' or ['mean', 'std'], not a function")
        if isinstance(func, str):
            return self._agg(func, **kw)
        return self._verbs.group_agg_multi(self._df, self._by, self._value, list(func), **kw)

    aggregate = agg

    def transform(self, func):
        """``groupby(by)[value].transform('mean')`` — the group statistic
        broadcast back to each row. Returns a private SafeColumn (e.g. for
        within-group demeaning); only stat names are accepted."""
        if not isinstance(func, str) or func not in _SUMMARISE_FUNCS:
            raise DisclosureError(
                f"transform takes a stat name from {sorted(_SUMMARISE_FUNCS)} "
                "(a function would be arbitrary code)")
        out = self._df.groupby(self._by, observed=True)[self._value].transform(func)
        return SafeColumn(out, self._verbs)

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

    # -- dataset-producing verbs: return a new (private) SafeFrame --
    def merge(self, other, *, on, how="inner") -> "SafeFrame":
        """Join with another dataset in scope. The result is a private SafeFrame
        (disclosure control still applies when it is finally released)."""
        if not isinstance(other, SafeFrame):
            raise DisclosureError("merge needs another dataset (a SafeFrame)")
        if how not in ("inner", "left", "right", "outer"):
            raise DisclosureError(f"unknown join type: {how!r}")
        keys = [on] if isinstance(on, str) else list(on)
        for c in keys:
            if c not in self._df.columns or c not in other._df.columns:
                raise DisclosureError(f"join key not in both datasets: {c}")
        merged = self._df.merge(other._df, on=keys, how=how, suffixes=("", "_y"))
        return SafeFrame(merged, self._verbs)

    def summarise(self, by, **aggs) -> "SafeFrame":
        """Group-aggregate into a new dataset: summarise('region',
        mean_pay=('salary','mean'), n=('salary','count')). Returns a SafeFrame
        (use pandas-style named aggregations)."""
        if not aggs:
            raise DisclosureError("summarise needs at least one name=(column, func)")
        bys = [by] if isinstance(by, str) else list(by)
        for b in bys:
            if b not in self._df.columns:
                raise DisclosureError(f"unknown column: {b}")
        spec = {}
        for name, pair in aggs.items():
            if not (isinstance(pair, tuple) and len(pair) == 2):
                raise DisclosureError(f"'{name}' must be (column, func)")
            col, func = pair
            if col not in self._df.columns:
                raise DisclosureError(f"unknown column: {col}")
            if func not in _SUMMARISE_FUNCS:
                raise DisclosureError(
                    f"agg '{func}' is not allowed; choose one of {sorted(_SUMMARISE_FUNCS)}")
            spec[name] = (col, func)
        out = self._df.groupby(bys, observed=True).agg(**spec).reset_index()
        return SafeFrame(out, self._verbs)

    summarize = summarise  # US spelling

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
        out = self._df.fillna(value)
        _check_frame_edit(self._df, out, self._verbs)
        return SafeFrame(out, self._verbs)

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

    def replace(self, to_replace, value=None) -> "SafeFrame":
        """Recode values across the frame: replace({'sex': {'M': 'Male'}}) or
        replace('N/A', None). A value→value relabel, so the frame stays private."""
        _check_recode(to_replace, value)
        out = self._df.replace(to_replace) if value is None else self._df.replace(to_replace, value)
        _check_frame_edit(self._df, out, self._verbs)
        return SafeFrame(out, self._verbs)

    def sort_values(self, by, *, ascending=True) -> "SafeFrame":
        """Reorder rows (useful before shift/diff/cumsum on panel data). The frame
        stays private — there is no head/iloc to read the ordering back out."""
        cols = [by] if isinstance(by, str) else list(by)
        for c in cols:
            if c not in self._df.columns:
                raise DisclosureError(f"unknown column: {c}")
        return SafeFrame(self._df.sort_values(by=cols, ascending=ascending), self._verbs)

    def drop_duplicates(self, subset=None) -> "SafeFrame":
        if subset is not None:
            cols = [subset] if isinstance(subset, str) else list(subset)
            for c in cols:
                if c not in self._df.columns:
                    raise DisclosureError(f"unknown column: {c}")
            subset = cols
        return SafeFrame(self._df.drop_duplicates(subset=subset), self._verbs)

    # -- reshapes: rearrange rows/columns. The result is a private Safe object
    #    whose values were already private and still exit only through a
    #    suppressed aggregation, so reshaping is safe by construction. --
    def _reshaped(self, obj):
        return SafeFrame(obj, self._verbs) if isinstance(obj, pd.DataFrame) \
            else SafeColumn(obj, self._verbs)

    def _need(self, *cols):
        for c in cols:
            if c is not None and c not in self._df.columns:
                raise DisclosureError(f"unknown column: {c}")

    def melt(self, *, id_vars=None, value_vars=None, var_name=None, value_name="value"):
        ids = [id_vars] if isinstance(id_vars, str) else list(id_vars or [])
        vals = [value_vars] if isinstance(value_vars, str) else list(value_vars or [])
        self._need(*ids, *vals)
        return self._reshaped(self._df.melt(
            id_vars=ids or None, value_vars=vals or None,
            var_name=var_name, value_name=value_name))

    def pivot(self, *, index, columns, values):
        """Raw reshape (no aggregation). Safe because the result is a private
        SafeFrame; a *released* pivot with individual values in cells is still
        blocked at the exit (each reducer checks its own count). For a released
        cross-tab use pivot_table, which suppresses cells."""
        self._need(index, columns, values)
        return self._reshaped(self._df.pivot(index=index, columns=columns, values=values))

    def explode(self, column):
        self._need(column)
        return self._reshaped(self._df.explode(column))

    def stack(self, **kw):
        return self._reshaped(self._df.stack())

    def unstack(self, level=-1, **kw):
        return self._reshaped(self._df.unstack(level))

    # -- frame-wide transforms: another private SafeFrame --
    def astype(self, dtype) -> "SafeFrame":
        return SafeFrame(self._df.astype(dtype), self._verbs)

    def round(self, decimals=0) -> "SafeFrame":
        return SafeFrame(self._df.round(decimals), self._verbs)

    def clip(self, lower=None, upper=None) -> "SafeFrame":
        out = self._df.copy()
        num = out.select_dtypes("number").columns
        out[num] = out[num].clip(lower=lower, upper=upper)
        return SafeFrame(out, self._verbs)

    def select_dtypes(self, include=None, exclude=None) -> "SafeFrame":
        return SafeFrame(self._df.select_dtypes(include=include, exclude=exclude), self._verbs)

    def filter(self, items=None, like=None, regex=None) -> "SafeFrame":
        """Select columns by name (items / like / regex). Column selection is
        non-disclosive, so the result is another SafeFrame."""
        return SafeFrame(self._df.filter(items=items, like=like, regex=regex, axis=1),
                         self._verbs)

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

    def pivot_table(self, *, values, index, columns=None, aggfunc="mean", **kw) -> Released:
        return self._verbs.pivot_table(self._df, values=values, index=index,
                                       columns=columns, aggfunc=aggfunc, **kw)

    # -- hypothesis tests --
    def ttest(self, *, value, by=None, mu=0.0, **kw) -> Released:
        return self._verbs.ttest(self._df, value=value, by=by, mu=mu, **kw)

    def mannwhitney(self, *, value, by, **kw) -> Released:
        return self._verbs.mannwhitney(self._df, value=value, by=by, **kw)

    def anova(self, *, value, by, **kw) -> Released:
        return self._verbs.anova(self._df, value=value, by=by, **kw)

    def chisq(self, *, row, col, **kw) -> Released:
        return self._verbs.chisq(self._df, row=row, col=col, **kw)

    def corr_test(self, *, x, y, method="pearson", **kw) -> Released:
        return self._verbs.corr_test(self._df, x=x, y=y, method=method, **kw)

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

    def cov(self, **kw) -> Released:
        """Covariance matrix over numeric columns (a function of releasable
        std/corr; released only if the frame has at least ``min_n`` rows)."""
        num = self._df.select_dtypes("number")
        if num.shape[1] < 2:
            raise DisclosureError("cov needs at least two numeric columns")
        k = self._verbs._policy.min_n
        if len(self._df) < k:
            raise DisclosureError("too few rows to release a covariance matrix")
        return Released(frame_payload(num.cov().round(3)), audit={
            "kind": "table", "verb": "cov", "min_n": k, "backend": "pandas"})

    # -- column-wise reducers over the whole frame -> a suppressed Released series.
    #    Each column is suppressed on its own non-null count (the release rule is
    #    per-column, exactly like a single SafeColumn reducer). --
    def mean(self, **kw): return self._frame_reduce("mean")
    def sum(self, **kw): return self._frame_reduce("sum")
    def median(self, **kw): return self._frame_reduce("median")
    def std(self, **kw): return self._frame_reduce("std")
    def var(self, **kw): return self._frame_reduce("var")
    def count(self, **kw): return self._frame_reduce("count", numeric_only=False)
    def nunique(self, **kw): return self._frame_reduce("nunique", numeric_only=False, rounded=False)

    def _frame_reduce(self, stat, *, numeric_only=True, rounded=True) -> Released:
        cols = (self._df.select_dtypes("number") if numeric_only else self._df)
        if cols.shape[1] == 0:
            raise DisclosureError(f"{stat} needs at least one numeric column")
        policy = self._verbs._policy
        # counts/distinct use min_n; descriptive stats get the higher floor.
        k = policy.min_n if stat in ("count", "sum", "nunique") else _descriptive_k(policy)
        sf = policy.suppression.percentile_sig_figs
        rt = policy.round_to
        index, values, suppressed = [], [], 0
        for c in cols.columns:
            s = cols[c]
            n = int(s.notna().sum())
            index.append(str(c))
            if n < k:
                values.append(None); suppressed += 1
                continue
            src = _winsorized_series(s, policy) if stat in _WINSOR_STATS else s
            v = s.count() if stat == "count" else (
                int(s.nunique()) if stat == "nunique" else getattr(src, stat)())
            v = _num(v)
            if stat == "median" and sf:
                v = _sig_round(v, sf)
            elif rounded and rt is not None and v is not None:
                v = float(round(v / rt) * rt)
            values.append(v)
        return Released({"type": "series", "name": stat, "index": index, "values": values},
                        audit={"kind": "table", "verb": f"frame.{stat}", "min_n": k,
                               "cols_suppressed": suppressed, "backend": "pandas"})

    def describe(self, *, winsorize=None) -> Released:
        """Per-column summary table with the same order-statistic guards as the
        column-level ``describe`` (min/max/quartiles released only when enough
        observations lie at/beyond them)."""
        num = self._df.select_dtypes("number")
        if num.shape[1] == 0:
            raise DisclosureError("describe needs at least one numeric column")
        k = _descriptive_k(self._verbs._policy)
        sf = self._verbs._policy.suppression.percentile_sig_figs
        kc = self._verbs._policy.min_n
        data = {}
        for c in num.columns:
            s = num[c].dropna()
            n = int(s.shape[0])
            agg = (lambda f: float(f) if n >= k else None)      # mean/std: not sig-rounded
            data[str(c)] = {
                "count": n if n >= kc else None,
                "mean": agg(s.mean()), "std": agg(s.std()),
                "min": _order_stat(s, "min", k, winsorize=winsorize, sig_figs=sf)[0],
                "25%": _order_stat(s, "q", k, q=0.25, sig_figs=sf)[0],
                "50%": _order_stat(s, "q", k, q=0.50, sig_figs=sf)[0],
                "75%": _order_stat(s, "q", k, q=0.75, sig_figs=sf)[0],
                "max": _order_stat(s, "max", k, winsorize=winsorize, sig_figs=sf)[0],
            }
        tab = pd.DataFrame(data).reindex(
            ["count", "mean", "std", "min", "25%", "50%", "75%", "max"])
        return Released(frame_payload(tab), audit={
            "kind": "table", "verb": "describe", "min_n": k,
            "winsorized": winsorize, "backend": "pandas"})

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

    def refute_ate(self, *, outcome, treatment, confounders, method="weighting",
                   refuter="placebo", **kw) -> Released:
        return self._verbs.refute_ate(self._df, outcome=outcome, treatment=treatment,
                                      confounders=confounders, method=method,
                                      refuter=refuter, **kw)

    def propensity(self, *, treatment, confounders, **kw):
        return self._verbs.propensity(self._df, treatment=treatment,
                                      confounders=confounders, **kw)

    def synthetic_control(self, *, unit, time, outcome, treated_unit, treatment_time,
                          predictors=None, controls=None, unit_size=None, **kw) -> Released:
        return self._verbs.synthetic_control(
            self._df, unit=unit, time=time, outcome=outcome, treated_unit=treated_unit,
            treatment_time=treatment_time, predictors=predictors, controls=controls,
            unit_size=unit_size, **kw)

    def __len__(self):
        raise DisclosureError("len() on a frame is not allowed; use a count aggregation")

    def __repr__(self):
        return f"<SafeFrame cols={list(self._df.columns)}>"

    # deliberately NO __iter__, to_*, head, values, iloc, describe, ...
