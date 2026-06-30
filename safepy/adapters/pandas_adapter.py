"""Pandas result adapter — the reference implementation.

The hard truth this adapter encodes: **a raw pandas result has no provenance the
mediator can trust.** A table of means whose values happen to be integers is
indistinguishable from a table of counts; a scalar mean is indistinguishable
from a scalar max. So this adapter default-denies every raw pandas object and
points the user at ``safepy.safe`` (the curated verbs that compute an
aggregate together with its group counts and return a verified ``Released``
value). Those Released values bypass this adapter entirely — the mediator trusts
their attached audit.

This is the design boundary that motivates the phase-2 SafeFrame facade: the
only way to release a pandas result is to have produced it through a verb that
recorded how it was aggregated.

Pandas 3 / Arrow note: under Arrow-backed dtypes and copy-on-write defaults this
adapter is unaffected — it never inspects values to make a release decision, so
the Arrow string dtype and CoW semantics don't change its behaviour.
"""

from __future__ import annotations

import numbers
from typing import Any

import pandas as pd

from ..errors import DisclosureError
from ..policy import Policy
from ..result import SafeResult
from . import base


def _is_scalar_number(x) -> bool:
    return isinstance(x, numbers.Number) and not isinstance(x, bool)


class PandasAdapter:
    name = "pandas"

    def claims(self, result: Any) -> bool:
        return isinstance(result, (pd.Series, pd.DataFrame)) or _is_scalar_number(result)

    def make_safe(self, result: Any, policy: Policy) -> SafeResult:
        if _is_scalar_number(result):
            raise DisclosureError(
                "a bare scalar cannot be released: its provenance can't be "
                "verified (a mean is safe, a max is not, and they look identical "
                "here). Return a grouped table via safepy.safe instead.")
        raise DisclosureError(
            "a raw pandas result cannot be released directly, because its "
            "provenance and group counts are unknown. Produce the result through "
            "safepy.safe (e.g. safe.group_agg, safe.value_counts, "
            "safe.crosstab), which suppresses small cells.")


base.register(PandasAdapter())
