"""Output mediation: the single point where a computed value becomes (or fails
to become) a releasable result.

Nothing reaches the user except through here. In particular the runtime returns
the *object*, never its repr — so ``df`` as a final expression yields a frame
object that this mediator must clear, not a printed table. There is no path by
which a raw frame is stringified for the user.
"""

from __future__ import annotations

from typing import Any

from . import adapters
from .errors import DisclosureError
from .policy import Policy
from .result import Released, SafeResult


def mediate(result: Any, policy: Policy) -> SafeResult:
    # 1. Already cleared by a trusted safe-verb helper -> trust its audit.
    if isinstance(result, Released):
        return SafeResult(ok=True, kind=result.audit.get("kind", "table"),
                          payload=result.payload, audit=result.audit)

    # 2. Nothing to release.
    if result is None:
        raise DisclosureError("your code did not produce a result to release")

    # 3. Hand to the adapter that claims this type; default-deny if none does.
    adapter = adapters.find(result)
    if adapter is None:
        raise DisclosureError(
            f"results of type '{type(result).__name__}' cannot be released "
            "(no adapter knows how to disclosure-check them)"
        )
    return adapter.make_safe(result, policy)
