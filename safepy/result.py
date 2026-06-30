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


@dataclass
class SafeResult:
    """What the API returns. ``payload`` is render-ready and disclosure-checked;
    ``audit`` records what was done (suppressed cells, thresholds, verbs used)."""
    ok: bool
    kind: str                       # "table" | "scalar" | "model" | "plot" | "error"
    payload: Any = None
    audit: dict = field(default_factory=dict)
    error: dict | None = None

    def as_dict(self) -> dict:
        return {"ok": self.ok, "kind": self.kind,
                "payload": self.payload, "audit": self.audit, "error": self.error}
