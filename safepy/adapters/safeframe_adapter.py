"""Refuse a dangling facade intermediate returned as the final result.

In STRICT mode, selection/shaping operations return intermediates — a
``SafeFrame`` (from ``where``/``assign``/mask), a ``SafeColumn`` (from
``df['col']``), or a grouped object (from ``groupby``). None of these is a
releasable result; ending on one is refused with guidance, so the failure is a
clear message rather than a leak. Only a ``Released`` aggregate exits.
"""

from __future__ import annotations

from typing import Any

from ..errors import DisclosureError
from ..policy import Policy
from ..result import SafeResult
from ..safeframe import SafeColumn, SafeFrame, SafeGroupBy, SafeSeriesGroupBy
from . import base

_INTERMEDIATES = (SafeFrame, SafeColumn, SafeGroupBy, SafeSeriesGroupBy)


class SafeFrameAdapter:
    name = "safeframe"

    def claims(self, result: Any) -> bool:
        return isinstance(result, _INTERMEDIATES)

    def make_safe(self, result: Any, policy: Policy) -> SafeResult:
        kind = type(result).__name__
        raise DisclosureError(
            f"a {kind} is an intermediate, not a releasable result. End on an "
            "aggregation, e.g. df.groupby('sex')['salary'].mean(), "
            "df['region'].value_counts(), or df.ols(y=..., x=[...])."
        )


base.register(SafeFrameAdapter())
