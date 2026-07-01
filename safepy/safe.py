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
from .stats import StatsMixin

try:
    import protect
except ImportError:  # pragma: no cover
    protect = None

# Safe *given* a minimum group size. Extremes (max/min/quantile) are absent on
# purpose: they return individual values regardless of group size.
_ALLOWED_AGGS = frozenset({"mean", "sum", "count", "size", "median", "std", "var"})


def _unwrap(df):
    """Accept either a raw frame (OPEN profile) or a SafeFrame (STRICT profile).

    Trusted code may reach the underlying frame; user code cannot (the gate
    blocks ``_df`` as a private attribute). Duck-typed to avoid importing
    SafeFrame here (it imports this module)."""
    return df._df if getattr(df, "_is_safeframe", False) else df


class SafeVerbs(StatsMixin):
    """Policy-bound safe verbs injected into the sandbox as ``safe``.

    Tabular verbs (group_agg/value_counts/crosstab) live here; regression and
    survival verbs (ols/logit/poisson/cox/kaplan_meier) come from StatsMixin.
    """

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
        df = _unwrap(df)
        k = self._min_n(min_n)
        grouped = df.groupby(by, observed=True)[value]
        table = grouped.size() if agg == "size" else getattr(grouped, agg)()
        counts = grouped.size()
        safe = protect.suppress(table, counts=counts, min_n=k, round=self._round(round))
        return Released(series_payload(safe, name=f"{agg}({value})"), audit={
            "kind": "table", "verb": "group_agg", "agg": agg, "by": by,
            "value": value, "min_n": k, "groups": int(len(counts)),
            "cells_suppressed": int((counts < k).sum()), "backend": "pandas"})

    def group_agg_multi(self, df: pd.DataFrame, by, value: str, stats,
                        *, min_n=None, round=None) -> Released:
        """``df.groupby(by)[value].agg(['mean', 'std'])`` — a multi-stat table.
        Every stat for a group shares that group's row count, so the whole row is
        suppressed when the group is below ``min_n``."""
        if protect is None:
            raise DisclosureError("the 'protect' package is required")
        stats = list(stats)
        bad = [s for s in stats if s not in _ALLOWED_AGGS]
        if bad:
            raise DisclosureError(
                f"agg {bad} not allowed; choose from {sorted(_ALLOWED_AGGS)}")
        df = _unwrap(df)
        k = self._min_n(min_n)
        grouped = df.groupby(by, observed=True)[value]
        counts = grouped.size()
        table = grouped.agg(["size" if s == "size" else s for s in stats])
        if isinstance(table, pd.Series):        # single stat -> frame
            table = table.to_frame(stats[0])
        else:
            table.columns = stats
        counts_df = pd.DataFrame({c: counts for c in table.columns})
        safe = protect.suppress(table, counts=counts_df, min_n=k, round=self._round(round))
        return Released(frame_payload(safe), audit={
            "kind": "table", "verb": "group_agg_multi", "by": by, "value": value,
            "stats": stats, "min_n": k, "groups": int(len(counts)),
            "rows_suppressed": int((counts < k).sum()), "backend": "pandas"})

    def value_counts(self, df: pd.DataFrame, col: str, *, min_n=None, round=None) -> Released:
        """Suppressed frequency table of one column."""
        if protect is None:
            raise DisclosureError("the 'protect' package is required")
        df = _unwrap(df)
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
        df = _unwrap(df)
        k = self._min_n(min_n)
        tab = pd.crosstab(df[row], df[col])
        safe = protect.suppress(tab, counts=tab, min_n=k, round=self._round(round))
        return Released(frame_payload(safe), audit={
            "kind": "table", "verb": "crosstab", "row": row, "col": col,
            "min_n": k, "backend": "pandas"})

    def pivot_table(self, df: pd.DataFrame, *, values: str, index, columns=None,
                    aggfunc: str = "mean", min_n=None, round=None) -> Released:
        """``df.pivot_table(...)`` — a 2-D aggregation (crosstab generalised to any
        value + aggfunc). Each cell is paired with its contributing row count and
        suppressed below ``min_n``; the raw ``pivot`` reshape (no aggfunc) is
        refused by the gate because it would place individual values in cells."""
        if protect is None:
            raise DisclosureError("the 'protect' package is required")
        if aggfunc not in _ALLOWED_AGGS:
            raise DisclosureError(
                f"aggfunc '{aggfunc}' is not allowed; choose one of {sorted(_ALLOWED_AGGS)}")
        df = _unwrap(df)
        k = self._min_n(min_n)
        idx = [index] if isinstance(index, str) else list(index)
        cols = None if columns is None else ([columns] if isinstance(columns, str) else list(columns))
        for c in idx + (cols or []) + [values]:
            if c not in df.columns:
                raise DisclosureError(f"unknown column: {c}")
        func = "size" if aggfunc == "size" else aggfunc
        tab = df.pivot_table(values=values, index=idx, columns=cols, aggfunc=func)
        # contributing (non-null) count per cell — the basis for suppression
        counts = df.pivot_table(values=values, index=idx, columns=cols,
                                aggfunc="count").reindex_like(tab)
        safe = protect.suppress(tab, counts=counts, min_n=k, round=self._round(round))
        return Released(frame_payload(safe), audit={
            "kind": "table", "verb": "pivot_table", "aggfunc": aggfunc,
            "index": idx, "columns": cols, "values": values, "min_n": k,
            "cells_suppressed": int((counts < k).sum().sum()), "backend": "pandas"})
