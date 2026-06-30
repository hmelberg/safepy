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
from .policy import Policy, ProtectionLevel, resolve_policy
from .result import SafeResult
from .runtime import execute
from .safe import SafeVerbs

# Library handles exposed inside the sandbox. ``safe`` is added per-call because
# it is bound to the resolved policy. These names plus the source names are the
# only bare names user code may call (besides ast_gate._SAFE_BUILTINS).
_LIB_HANDLES = {"pd": pd, "np": np}


def run(code: str,
        sources: dict[str, Any],
        level: ProtectionLevel | str = ProtectionLevel.PROTECTED) -> SafeResult:
    """Validate, run, and disclosure-check ``code`` against ``sources``.

    ``sources`` maps the names user code may reference (e.g. ``{"df": frame}``)
    to private data objects. ``level`` selects the protection policy.
    """
    policy: Policy = resolve_policy([level])

    try:
        policy.require_sandbox()  # refuses direct exec for 'sensitive'

        handles = {**_LIB_HANDLES, "safe": SafeVerbs(policy)}
        allowed_names = frozenset(handles) | frozenset(sources)
        gate = validate(code, allowed_names=allowed_names)
        if not gate.ok:
            assert gate.error is not None
            return SafeResult(ok=False, kind="error", error=gate.error.as_dict())

        namespace = {**handles, **sources}
        value = execute(code, namespace)

        result = mediate(value, policy)
        result.audit.setdefault("level", policy.level.value)
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
