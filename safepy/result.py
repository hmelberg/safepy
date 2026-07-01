"""Result containers shared across the runtime, the safe-verb helpers, and the
mediator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Released:
    """A value that a *trusted* safe-verb helper has already cleared for release
    (it computed the result together with its group counts and ran
    ``protect.suppress``). The mediator passes these through and trusts the
    attached audit, rather than re-deriving suppression it cannot verify.

    Raw pandas/polars results returned by user code are NOT Released; they go
    through best-effort mediation, which refuses anything it cannot prove safe.
    """
    payload: Any
    audit: dict = field(default_factory=dict)

    @property
    def plot(self):
        """pandas-like plotting on an aggregate: value_counts().plot.bar().
        Refuses anything that isn't an aggregated table."""
        from .charts import PlotAccessor
        return PlotAccessor(self)


@dataclass
class SafeResult:
    """What the API returns. ``payload`` is render-ready and disclosure-checked;
    ``audit`` records what was done (suppressed cells, thresholds, verbs used)."""
    ok: bool
    kind: str                       # "table" | "scalar" | "model" | "plot" | "none" | "error"
    payload: Any = None
    audit: dict = field(default_factory=dict)
    error: dict | None = None
    catalog: list | None = None     # schema of datasets left in the session (no data)
    results: list | None = None     # all released results (the envelope); top-level = last

    def _leaf(self) -> dict:
        return {"ok": self.ok, "kind": self.kind, "payload": self.payload,
                "audit": self.audit, "error": self.error}

    def as_dict(self) -> dict:
        d = self._leaf()
        d["catalog"] = self.catalog
        # serialise each result as a leaf (no nested results/catalog -> no recursion)
        d["results"] = None if self.results is None else [r._leaf() for r in self.results]
        return d
