"""Curated safe verbs — the trusted release path.

These are the seed of the phase-2 SafeFrame facade. Each verb computes an
aggregate *together with the group counts it needs to be suppressible*, runs
``protect.suppress``, and returns a :class:`Released` value the mediator trusts.

Crucially, the verbs are bound to the active :class:`Policy`: ``min_n`` defaults
to the policy floor and a caller may only make it *stricter*, never weaker. The
sandbox sees an instance as ``safe``.

User code may compute freely with raw pandas for intermediate steps, but the
*released* value must come from one of these verbs — raw pandas results are
default-denied by the mediator, because their provenance can't be verified (a
table of means whose values happen to be integers is indistinguishable from a
table of counts).
"""

from __future__ import annotations

import pandas as pd

from ._payload import series_payload, frame_payload
from .errors import DisclosureError
from .policy import Policy
from .result import Released

try:
    import protect
except ImportError:  # pragma: no cover
    protect = None

# Safe *given* a minimum group size. Extremes (max/min/quantile) are absent on
# purpose: they return individual values regardless of group size.
_ALLOWED_AGGS = frozenset({"mean", "sum", "count", "size", "median", "std", "var"})


class SafeVerbs:
    """Policy-bound safe verbs injected into the sandbox as ``safe``."""

    def __init__(self, policy: Policy):
        self._policy = policy

    def _min_n(self, requested) -> int:
        floor = self._policy.min_n
        return floor if requested is None else max(int(requested), floor)

    def _round(self, requested):
        return self._policy.round_to if requested is None else requested

    def group_agg(self, df: pd.DataFrame, by, value: str, agg: str = "mean",
                  *, min_n=None, round=None) -> Released:
        """``df.groupby(by)[value].agg(...)`` with paired counts and suppression."""
        if protect is None:
            raise DisclosureError("the 'protect' package is required")
        if agg not in _ALLOWED_AGGS:
            raise DisclosureError(
                f"agg '{agg}' is not allowed; choose one of {sorted(_ALLOWED_AGGS)}")
        k = self._min_n(min_n)
        grouped = df.groupby(by, observed=True)[value]
        table = grouped.size() if agg == "size" else getattr(grouped, agg)()
        counts = grouped.size()
        safe = protect.suppress(table, counts=counts, min_n=k, round=self._round(round))
        return Released(series_payload(safe, name=f"{agg}({value})"), audit={
            "kind": "table", "verb": "group_agg", "agg": agg, "by": by,
            "value": value, "min_n": k, "groups": int(len(counts)),
            "cells_suppressed": int((counts < k).sum()), "backend": "pandas"})

    def value_counts(self, df: pd.DataFrame, col: str, *, min_n=None, round=None) -> Released:
        """Suppressed frequency table of one column."""
        if protect is None:
            raise DisclosureError("the 'protect' package is required")
        k = self._min_n(min_n)
        counts = df[col].value_counts()
        safe = protect.suppress(counts, counts=counts, min_n=k, round=self._round(round))
        return Released(series_payload(safe, name=f"count({col})"), audit={
            "kind": "table", "verb": "value_counts", "col": col, "min_n": k,
            "cells_suppressed": int((counts < k).sum()), "backend": "pandas"})

    def crosstab(self, df: pd.DataFrame, row: str, col: str,
                 *, min_n=None, round=None) -> Released:
        """Suppressed frequency cross-tabulation of two columns."""
        if protect is None:
            raise DisclosureError("the 'protect' package is required")
        k = self._min_n(min_n)
        tab = pd.crosstab(df[row], df[col])
        safe = protect.suppress(tab, counts=tab, min_n=k, round=self._round(round))
        return Released(frame_payload(safe), audit={
            "kind": "table", "verb": "crosstab", "row": row, "col": col,
            "min_n": k, "backend": "pandas"})
