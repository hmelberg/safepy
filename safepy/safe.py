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

import hashlib
import os

import numpy as np
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

# Tiltak 3 (støylegging): counts of units are perturbed by deterministic
# cell-key noise. The noise for a cell is a function of the cell's label and a
# secret salt, so the SAME cell yields the SAME noise on every query — repeating
# a query cannot average the noise away. protect.noise is positional (not
# cell-keyed) and so cannot provide this; the salted-hash seeding here is the one
# piece of SDC orchestration safepy layers on top of protect's primitives.
#
# The salt MUST be secret in production (safepy is open source, so a known salt
# lets an attacker subtract the noise). Set SAFEPY_NOISE_SALT in the deployment.
_NOISE_SALT = os.environ.get("SAFEPY_NOISE_SALT", "safepy-default-INSECURE-salt")


def _cell_noise(key, step: int) -> int:
    """Deterministic integer noise in ``[-step, step]`` for a cell identified by
    ``key`` (a label or (row, col) tuple)."""
    digest = hashlib.blake2b(f"{_NOISE_SALT}\x00{key!r}".encode(), digest_size=8).digest()
    rng = np.random.default_rng(int.from_bytes(digest, "big"))
    return int(rng.integers(-step, step + 1))


def _noise_count_table(counts, policy):
    """Add cell-key noise to a table of counts (Series or DataFrame), flooring at
    0 and leaving suppressed (NaN) cells untouched (Tiltak 3)."""
    step = policy.suppression.count_noise
    if not step:
        return counts
    step = int(step)
    noise1 = lambda key, v: v if pd.isna(v) else float(max(0, int(round(v)) + _cell_noise(key, step)))
    if isinstance(counts, pd.Series):
        return pd.Series({i: noise1(i, v) for i, v in counts.items()}, name=counts.name)
    out = counts.astype(float).copy()
    for i in out.index:
        for c in out.columns:
            out.at[i, c] = noise1((i, c), out.at[i, c])
    return out


def _noised_counts(counts: pd.Series, policy) -> pd.Series:
    """The noised version of a group-size Series (per group label), used to keep
    a released sum consistent with its count: noised_sum = mean * noised_count."""
    step = policy.suppression.count_noise
    if not step:
        return counts
    step = int(step)
    return pd.Series({i: max(0, int(round(v)) + _cell_noise(i, step)) for i, v in counts.items()},
                     name=counts.name)

# Safe *given* a minimum group size. Extremes (max/min/quantile) are absent on
# purpose: they return individual values regardless of group size.
_ALLOWED_AGGS = frozenset({"mean", "sum", "count", "size", "median", "std", "var"})


def _unwrap(df):
    """Accept either a raw frame (OPEN profile) or a SafeFrame (STRICT profile).

    Trusted code may reach the underlying frame; user code cannot (the gate
    blocks ``_df`` as a private attribute). Duck-typed to avoid importing
    SafeFrame here (it imports this module)."""
    return df._df if getattr(df, "_is_safeframe", False) else df


# Grouped moment aggregates winsorization affects (Tiltak 2).
_WINSOR_AGGS = frozenset({"mean", "std", "var", "sum"})


def _winsorize_col(df, col, policy):
    """Return ``df`` with ``col`` winsorized per the policy (Tiltak 2), or
    unchanged if off / non-numeric. Group sizes are taken from the original df, so
    only the values (not the counts) are capped."""
    w = policy.suppression.winsorize
    if (w is None or protect is None or not pd.api.types.is_numeric_dtype(df[col])
            or pd.api.types.is_bool_dtype(df[col])):   # bools/indicators not winsorized
        return df
    return protect.winsorize(df, col, limits=(float(w[0]), float(w[1])))


def _agg_min_n(policy, agg):
    """Suppression threshold for a grouped aggregate: counts/sums/size use the
    primary ``min_n``; descriptive aggregates (mean/median/std/var) get the
    higher Tiltak 7/1 descriptive-population floor."""
    if agg in ("count", "size", "sum"):
        return policy.min_n
    from .safeframe import _descriptive_k  # lazy: safeframe imports this module
    return _descriptive_k(policy)


def _stop_if_too_sparse(counts, policy):
    """Tiltak 5: refuse a frequency table when more than
    ``max_low_cell_share`` of its cells fall below ``min_n`` — such tables both
    ease indirect identification and carry high relative noise."""
    share = policy.suppression.max_low_cell_share
    if share is None:
        return
    flat = np.asarray(counts).ravel()
    flat = flat[~pd.isna(flat)]
    if flat.size == 0:
        return
    low = int((flat < policy.min_n).sum())
    if low / flat.size > share:
        raise DisclosureError(
            f"table stopped: more than {int(share * 100)}% of its cells are below "
            f"the minimum count of {policy.min_n}. Use coarser categories or a "
            "larger population so cells are better populated.")


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
        """``df.groupby(by)[value].agg(...)`` with paired counts and suppression.

        The *compute* (groupby + winsorize) is pandas-specific; the *release*
        (suppression + count-noise + audit) is backend-neutral (``_release_group_agg``)
        so a polars backend can compute ``(table, counts)`` natively and reuse the
        exact same audited suppressor. See ``polars_api._native_group_agg``."""
        if agg not in _ALLOWED_AGGS:
            raise DisclosureError(
                f"agg '{agg}' is not allowed; choose one of {sorted(_ALLOWED_AGGS)}")
        df = _unwrap(df)
        counts = df.groupby(by, observed=True)[value].size()
        work = _winsorize_col(df, value, self._policy) if agg in _WINSOR_AGGS else df
        grouped = work.groupby(by, observed=True)[value]
        table = counts if agg == "size" else getattr(grouped, agg)()
        return self._release_group_agg(table, counts, agg=agg, by=by, value=value,
                                       min_n=min_n, round=round, backend="pandas")

    def _release_group_agg(self, table, counts, *, agg, by, value,
                           min_n=None, round=None, backend="pandas") -> Released:
        """Backend-neutral release: given an already-computed per-group aggregate
        ``table`` and paired ``counts`` (pandas Series, shared index), suppress
        below ``min_n``, apply count-noise (Tiltak 3), and stamp the audit. This
        is the single audited suppressor for grouped aggregates, whatever backend
        produced the numbers."""
        if protect is None:
            raise DisclosureError("the 'protect' package is required")
        if agg not in _ALLOWED_AGGS:
            raise DisclosureError(
                f"agg '{agg}' is not allowed; choose one of {sorted(_ALLOWED_AGGS)}")
        table = table.reindex(counts.index)     # align (polars group order isn't stable)
        k = max(self._min_n(min_n), _agg_min_n(self._policy, agg))
        noising = bool(self._policy.suppression.count_noise)
        # when noising counts, don't also round them (rounding would swamp the noise)
        rnd = None if (noising and agg in ("count", "size")) else self._round(round)
        safe = protect.suppress(table, counts=counts, min_n=k, round=rnd)
        if noising:
            safe = self._apply_count_noise(safe, table, counts, agg)
        return Released(series_payload(safe, name=f"{agg}({value})"), audit={
            "kind": "table", "verb": "group_agg", "agg": agg, "by": by,
            "value": value, "min_n": k, "groups": int(len(counts)),
            "count_noise": self._policy.suppression.count_noise,
            "cells_suppressed": int((counts < k).sum()), "backend": backend})

    def _apply_count_noise(self, safe, table, counts, agg):
        """Tiltak 3 for a grouped release: noise the counts, and keep sums/means
        consistent with the noised count (mean = sum/count is preserved; a cell
        whose noised count is 0 releases sum 0 and mean NaN)."""
        if agg in ("count", "size"):
            return _noise_count_table(safe, self._policy)
        nc = _noised_counts(counts, self._policy).reindex(safe.index)
        if agg == "sum":
            tc = counts.reindex(safe.index)
            # scale so mean = sum/count is preserved; nc==0 -> 0, suppressed -> NaN
            return safe * (nc / tc)
        if agg == "mean":
            return safe.where(nc > 0, np.nan)         # 0 units behind the mean -> NaN
        return safe                                    # median/std/var: not count-based

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
        # a multi-stat table mixes counts and descriptive stats; use the stricter
        # descriptive floor when any descriptive stat is present.
        k = max([self._min_n(min_n)] + [_agg_min_n(self._policy, s) for s in stats])
        counts = df.groupby(by, observed=True)[value].size()
        # winsorize the value if any requested stat is moment-based (median barely
        # moves under 1% winsorization, so a shared source is acceptable).
        work = _winsorize_col(df, value, self._policy) if any(
            s in _WINSOR_AGGS for s in stats) else df
        grouped = work.groupby(by, observed=True)[value]
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
        df = _unwrap(df)
        counts = df[col].value_counts()
        return self._release_value_counts(counts, col=col, min_n=min_n, round=round,
                                          backend="pandas")

    def _release_value_counts(self, counts, *, col, min_n=None, round=None,
                              backend="pandas") -> Released:
        """Backend-neutral release of a per-value count Series (Tiltak 5 stop rule
        + suppression + count-noise). A polars backend passes counts it computed
        natively; the audited suppression is identical."""
        if protect is None:
            raise DisclosureError("the 'protect' package is required")
        k = self._min_n(min_n)
        _stop_if_too_sparse(counts.to_numpy(), self._policy)
        noising = bool(self._policy.suppression.count_noise)
        safe = protect.suppress(counts, counts=counts, min_n=k,
                                round=None if noising else self._round(round))
        if noising:
            safe = _noise_count_table(safe, self._policy)
        return Released(series_payload(safe, name=f"count({col})"), audit={
            "kind": "table", "verb": "value_counts", "col": col, "min_n": k,
            "count_noise": self._policy.suppression.count_noise,
            "cells_suppressed": int((counts < k).sum()), "backend": backend})

    def crosstab(self, df: pd.DataFrame, row: str, col: str,
                 *, min_n=None, round=None) -> Released:
        """Suppressed frequency cross-tabulation of two columns."""
        df = _unwrap(df)
        tab = pd.crosstab(df[row], df[col])
        return self._release_crosstab(tab, row=row, col=col, min_n=min_n, round=round,
                                      backend="pandas")

    def _release_crosstab(self, tab, *, row, col, min_n=None, round=None,
                          backend="pandas") -> Released:
        """Backend-neutral release of a 2-D count table (row x col DataFrame)."""
        if protect is None:
            raise DisclosureError("the 'protect' package is required")
        k = self._min_n(min_n)
        _stop_if_too_sparse(tab.to_numpy(), self._policy)
        noising = bool(self._policy.suppression.count_noise)
        safe = protect.suppress(tab, counts=tab, min_n=k,
                                round=None if noising else self._round(round))
        if noising:
            safe = _noise_count_table(safe, self._policy)
        return Released(frame_payload(safe), audit={
            "kind": "table", "verb": "crosstab", "row": row, "col": col,
            "min_n": k, "count_noise": self._policy.suppression.count_noise,
            "backend": backend})

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
        _stop_if_too_sparse(counts.to_numpy(), self._policy)
        noising = bool(self._policy.suppression.count_noise)
        pure_count = aggfunc in ("count", "size")
        safe = protect.suppress(tab, counts=counts, min_n=k,
                                round=None if (noising and pure_count) else self._round(round))
        if noising:
            if pure_count:
                safe = _noise_count_table(safe, self._policy)
            elif aggfunc == "sum":
                nc = _noise_count_table(counts, self._policy)
                safe = safe * (nc / counts)                 # preserve mean; 0->0
            elif aggfunc == "mean":
                nc = _noise_count_table(counts, self._policy)
                safe = safe.where(nc > 0, np.nan)
        return Released(frame_payload(safe), audit={
            "kind": "table", "verb": "pivot_table", "aggfunc": aggfunc,
            "index": idx, "columns": cols, "values": values, "min_n": k,
            "count_noise": self._policy.suppression.count_noise,
            "cells_suppressed": int((counts < k).sum().sum()), "backend": "pandas"})
