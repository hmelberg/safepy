"""The one entry point: ``run(code, sources, level) -> SafeResult``.

This is the synchronous core of what the safestat spec calls ``/run_extended``.
The submit-then-poll wrapper (background task + ``task_id``) is deliberately not
here yet; it wraps this function without changing it.

Pipeline:  policy -> gate -> sandbox -> mediate.  Each stage can only ever
*reduce* what is releasable; there is no path around the mediator.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .ast_gate import validate
from .errors import DisclosureError, SafePythonError, SandboxError, ValidationError
from .mediator import mediate
from .policy import Policy, Profile, ProtectionLevel, resolve_policy
from .result import SafeResult
from .runtime import execute
from .safe import SafeVerbs
from .safeframe import SafeFrame


def _build_namespace(profile: Profile, policy: Policy, sources: dict[str, Any]) -> dict:
    """The single difference between the two security postures.

    OPEN   — real pandas/numpy + the raw frames are in scope.
    STRICT — only the safe-verb library, SafeFrame-wrapped sources, and the
             look-alike `pd`/`np` facades; no real pandas, no raw frame, so
             disclosive capabilities are simply not reachable.
    """
    verbs = SafeVerbs(policy)
    if profile is Profile.STRICT:
        from .namespaces import SafeNp, SafePd
        return {"safe": verbs, "pd": SafePd(), "np": SafeNp(),
                **{name: SafeFrame(df, verbs) for name, df in sources.items()}}
    return {"pd": pd, "np": np, "safe": verbs, **sources}


def run(code: str,
        sources: dict[str, Any],
        level: ProtectionLevel | str = ProtectionLevel.PROTECTED,
        *, profile: Profile | str | None = None) -> SafeResult:
    """Validate, run, and disclosure-check ``code`` against ``sources``.

    ``sources`` maps the names user code may reference (e.g. ``{"df": frame}``)
    to private data objects. ``level`` selects the protection policy; ``profile``
    overrides the executor (OPEN sandbox vs STRICT capability) for that policy,
    which is useful for development and testing.
    """
    policy: Policy = resolve_policy([level])
    active = Profile(profile) if profile is not None else policy.profile

    try:
        namespace = _build_namespace(active, policy, sources)
        allowed_names = frozenset(namespace)
        gate = validate(code, allowed_names=allowed_names)
        if not gate.ok:
            assert gate.error is not None
            return SafeResult(ok=False, kind="error", error=gate.error.as_dict())

        value = execute(code, namespace)

        result = mediate(value, policy)
        result.audit.setdefault("level", policy.level.value)
        result.audit.setdefault("profile", active.value)
        result.audit.setdefault("verbs_used", gate.calls)
        return result

    except ValidationError as exc:
        return SafeResult(ok=False, kind="error", error=exc.as_dict())
    except (DisclosureError, SandboxError) as exc:
        return SafeResult(ok=False, kind="error",
                          error={"kind": type(exc).__name__, "message": str(exc)})
    except SafePythonError as exc:  # pragma: no cover - catch-all, still no data leak
        return SafeResult(ok=False, kind="error",
                          error={"kind": "SafePythonError", "message": str(exc)})
