"""The adapter interface.

An adapter is the *only* place that knows about a specific result type
(pandas object, polars object, statsmodels result, plotly/matplotlib figure).
The core (gate, runtime, mediator) stays library-neutral and asks an adapter two
questions:

* ``claims(result)`` — is this mine?
* ``make_safe(result, policy)`` — turn it into a release-checked SafeResult, or
  raise DisclosureError if it cannot be proven safe.

This is the seam that lets safepy fold into m2py later: ``protect`` is the
pandas reference implementation of result-side protection, exactly as the
safestat spec frames it ("protect is the pandas ProtectionAdapter, not the
universal engine"). A polars adapter and a statsmodels adapter slot in beside it
without touching the core.
"""

from __future__ import annotations

from typing import Any, Protocol

from ..policy import Policy
from ..result import SafeResult


class Adapter(Protocol):
    name: str

    def claims(self, result: Any) -> bool: ...

    def make_safe(self, result: Any, policy: Policy) -> SafeResult: ...


_REGISTRY: list[Adapter] = []


def register(adapter: Adapter) -> Adapter:
    _REGISTRY.append(adapter)
    return adapter


def find(result: Any) -> Adapter | None:
    for a in _REGISTRY:
        if a.claims(result):
            return a
    return None
