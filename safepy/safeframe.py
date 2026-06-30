"""SafeFrame — the STRICT-profile capability facade.

The whole security argument for the strict profile lives in this file's *size*.
In STRICT mode the sandbox namespace contains no pandas and no raw frame — only
a ``SafeFrame`` wrapping the private data. A SafeFrame exposes nothing but the
methods below. ``data.head()`` is not "blocked"; it simply does not exist. So
the surface a human must audit to certify the strict profile is exactly this
list of methods, not all of pandas.

Two invariants make that argument hold:

1. **No method returns raw rows.** Terminal verbs return a ``Released``
   aggregate (already suppressed via ``protect``); shaping verbs (``where``,
   ``assign``) return another ``SafeFrame``. Nothing hands back a Series of
   individual values, and there is no ``__getitem__``, ``__iter__``, or
   ``__repr__`` that exposes the frame.
2. **The raw frame is reachable only by trusted code.** It lives in ``_df``;
   the AST gate blocks any user access to a ``_``-prefixed attribute, so user
   code can never reach ``data._df``.

A dangling ``SafeFrame`` returned as the final result is refused by the mediator
(see ``adapters/safeframe_adapter``): you must end on an aggregation.
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd

from .errors import DisclosureError
from .result import Released
from .safe import SafeVerbs

# Element-wise functions allowed inside assign() expressions, mapped to numpy.
# This is a *trusted* whitelist compiler (we translate the AST ourselves, we do
# not eval the string), so the usual "string DSL = hidden code" danger does not
# apply — cf. patsy formulas, which ARE eval'd and are therefore unsafe.
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


def _compile_expr(df: pd.DataFrame, expr: str):
    """Evaluate a whitelisted arithmetic expression over df's columns.

    Allowed: column names, numeric literals, + - * / % **, unary minus, and the
    functions in _EXPR_FUNCS. Anything else raises DisclosureError. No attribute
    access, no subscripts, no calls except the whitelisted funcs.
    """
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


class SafeGroupBy:
    """The result of ``SafeFrame.groupby(...)``. Only aggregations, and each one
    returns a suppressed ``Released`` aggregate via the safe-verb library."""

    def __init__(self, df: pd.DataFrame, by, verbs: SafeVerbs):
        self._df = df
        self._by = by
        self._verbs = verbs

    def mean(self, value: str, **kw) -> Released:  return self._agg("mean", value, **kw)
    def sum(self, value: str, **kw) -> Released:   return self._agg("sum", value, **kw)
    def count(self, value: str, **kw) -> Released: return self._agg("count", value, **kw)
    def median(self, value: str, **kw) -> Released: return self._agg("median", value, **kw)
    def std(self, value: str, **kw) -> Released:   return self._agg("std", value, **kw)
    def var(self, value: str, **kw) -> Released:   return self._agg("var", value, **kw)
    def size(self, **kw) -> Released:
        # group sizes are a frequency table; reuse group_agg's 'size'
        return self._verbs.group_agg(self._df, self._by, self._by if isinstance(self._by, str)
                                     else self._by[0], "size", **kw)

    def _agg(self, agg: str, value: str, **kw) -> Released:
        return self._verbs.group_agg(self._df, self._by, value, agg, **kw)


class SafeFrame:
    """The only data object user code can touch in STRICT mode."""

    _is_safeframe = True  # duck-type marker used by safe._unwrap

    def __init__(self, df: pd.DataFrame, verbs: SafeVerbs):
        self._df = df
        self._verbs = verbs

    # -- shaping verbs: return another SafeFrame (no disclosure) --
    def where(self, col: str, op: str, value) -> "SafeFrame":
        """Row filter by a single column comparison. Returns a SafeFrame."""
        if op not in _CMP_OPS:
            raise DisclosureError(f"unknown comparison operator: {op!r}")
        if col not in self._df.columns:
            raise DisclosureError(f"unknown column: {col}")
        mask = _CMP_OPS[op](self._df[col], value)
        return SafeFrame(self._df[mask], self._verbs)

    def assign(self, name: str, expr: str) -> "SafeFrame":
        """Add a derived column from a whitelisted arithmetic expression."""
        series = _compile_expr(self._df, expr)
        return SafeFrame(self._df.assign(**{name: series}), self._verbs)

    # -- terminal verbs: return a suppressed Released aggregate --
    def groupby(self, by) -> SafeGroupBy:
        return SafeGroupBy(self._df, by, self._verbs)

    def value_counts(self, col: str, **kw) -> Released:
        return self._verbs.value_counts(self._df, col, **kw)

    def crosstab(self, row: str, col: str, **kw) -> Released:
        return self._verbs.crosstab(self._df, row, col, **kw)

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

    # deliberately NO __getitem__, __iter__, __repr__, to_*, head, values, ...
