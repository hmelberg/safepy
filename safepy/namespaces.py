"""Look-alike ``pd`` and ``np`` namespaces for STRICT mode.

These let familiar module-level calls work — ``np.log(df['wage'])``,
``pd.crosstab(df['sex'], df['region'])``, ``pd.cut(df['age'], bins)`` — while
staying inside the facade: every function takes and returns ``SafeColumn``s (or a
suppressed ``Released`` table), never raw values. Anything we haven't implemented
raises a clear "not available" error rather than a bare ``AttributeError``.

They borrow the policy from whichever ``SafeColumn`` is passed in (a column
already carries its ``SafeVerbs``), so there is no separate wiring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._payload import frame_payload
from .errors import DisclosureError
from .result import Released
from .safeframe import SafeColumn

try:
    import protect
except ImportError:  # pragma: no cover
    protect = None


def _arr(v):
    """Underlying array/scalar for use inside a vectorised op."""
    return v._s.values if isinstance(v, SafeColumn) else v


class SafeNp:
    """A whitelist of element-wise numpy functions over SafeColumns."""

    # constants commonly used in bins/expressions
    inf = np.inf
    nan = np.nan
    pi = np.pi
    e = np.e

    def _u(self, x, fn):
        if isinstance(x, SafeColumn):
            return SafeColumn(fn(x._s), x._verbs)
        return float(fn(x))

    def _b(self, a, b, fn):
        col = a if isinstance(a, SafeColumn) else (b if isinstance(b, SafeColumn) else None)
        if col is None:
            return float(fn(a, b))
        res = fn(_arr(a), _arr(b))
        if not isinstance(res, pd.Series):
            res = pd.Series(res, index=col._s.index)
        return SafeColumn(res, col._verbs)

    # element-wise unary functions
    def log(self, x): return self._u(x, np.log)
    def log2(self, x): return self._u(x, np.log2)
    def log10(self, x): return self._u(x, np.log10)
    def log1p(self, x): return self._u(x, np.log1p)
    def exp(self, x): return self._u(x, np.exp)
    def expm1(self, x): return self._u(x, np.expm1)
    def sqrt(self, x): return self._u(x, np.sqrt)
    def cbrt(self, x): return self._u(x, np.cbrt)
    def square(self, x): return self._u(x, np.square)
    def abs(self, x): return self._u(x, np.abs)
    def sign(self, x): return self._u(x, np.sign)
    def floor(self, x): return self._u(x, np.floor)
    def ceil(self, x): return self._u(x, np.ceil)
    def trunc(self, x): return self._u(x, np.trunc)
    def rint(self, x): return self._u(x, np.rint)
    def sin(self, x): return self._u(x, np.sin)
    def cos(self, x): return self._u(x, np.cos)
    def tan(self, x): return self._u(x, np.tan)
    def arcsin(self, x): return self._u(x, np.arcsin)
    def arccos(self, x): return self._u(x, np.arccos)
    def arctan(self, x): return self._u(x, np.arctan)
    def sinh(self, x): return self._u(x, np.sinh)
    def cosh(self, x): return self._u(x, np.cosh)
    def tanh(self, x): return self._u(x, np.tanh)
    def radians(self, x): return self._u(x, np.radians)
    def degrees(self, x): return self._u(x, np.degrees)

    def round(self, x, decimals=0):
        if isinstance(x, SafeColumn):
            return SafeColumn(np.round(x._s, decimals), x._verbs)
        return float(np.round(x, decimals))

    # element-wise binary functions
    def minimum(self, a, b): return self._b(a, b, np.minimum)
    def maximum(self, a, b): return self._b(a, b, np.maximum)
    def power(self, a, b): return self._b(a, b, np.power)
    def mod(self, a, b): return self._b(a, b, np.mod)
    def hypot(self, a, b): return self._b(a, b, np.hypot)
    def arctan2(self, a, b): return self._b(a, b, np.arctan2)

    def where(self, cond, a, b):
        if not isinstance(cond, SafeColumn):
            raise DisclosureError("np.where needs a boolean column as its condition")
        s = pd.Series(np.where(cond._s.values, _arr(a), _arr(b)), index=cond._s.index)
        return SafeColumn(s, cond._verbs)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raise DisclosureError(
            f"np.{name} is not available in safepy (only element-wise functions "
            "over columns are supported)")


class SafePd:
    """A whitelist of pandas module-level helpers over SafeColumns."""

    def to_datetime(self, x, **kw):
        if not isinstance(x, SafeColumn):
            raise DisclosureError("pd.to_datetime needs a column")
        return SafeColumn(pd.to_datetime(x._s, **kw), x._verbs)

    def to_numeric(self, x, **kw):
        if not isinstance(x, SafeColumn):
            raise DisclosureError("pd.to_numeric needs a column")
        return SafeColumn(pd.to_numeric(x._s, **kw), x._verbs)

    def cut(self, x, bins, *, labels=None, right=True):
        if not isinstance(x, SafeColumn):
            raise DisclosureError("pd.cut needs a column")
        return SafeColumn(pd.cut(x._s, bins, labels=labels, right=right), x._verbs)

    def qcut(self, x, q, *, labels=None):
        if not isinstance(x, SafeColumn):
            raise DisclosureError("pd.qcut needs a column")
        return SafeColumn(pd.qcut(x._s, q, labels=labels), x._verbs)

    def crosstab(self, row, col, *, min_n=None, round=None) -> Released:
        if not (isinstance(row, SafeColumn) and isinstance(col, SafeColumn)):
            raise DisclosureError("pd.crosstab needs two columns")
        if protect is None:
            raise DisclosureError("the 'protect' package is required")
        verbs = row._verbs
        k = verbs._min_n(min_n)
        tab = pd.crosstab(row._s, col._s)
        safe = protect.suppress(tab, counts=tab, min_n=k, round=verbs._round(round))
        return Released(frame_payload(safe), audit={
            "kind": "table", "verb": "crosstab", "min_n": k, "backend": "pandas"})

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raise DisclosureError(
            f"pd.{name} is not available in safepy (it could construct or reveal "
            "raw data); use a column helper or a safe verb")
