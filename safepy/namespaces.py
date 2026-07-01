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

    def log(self, x): return self._u(x, np.log)
    def log10(self, x): return self._u(x, np.log10)
    def log1p(self, x): return self._u(x, np.log1p)
    def exp(self, x): return self._u(x, np.exp)
    def sqrt(self, x): return self._u(x, np.sqrt)
    def abs(self, x): return self._u(x, np.abs)
    def floor(self, x): return self._u(x, np.floor)
    def ceil(self, x): return self._u(x, np.ceil)

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
