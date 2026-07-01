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
        from .formula_api import SafeStats
        return {"safe": verbs, "pd": SafePd(), "np": SafeNp(), "smf": SafeStats(verbs),
                **{name: SafeFrame(df, verbs) for name, df in sources.items()}}
    return {"pd": pd, "np": np, "safe": verbs, **sources}


def run(code: str,
        sources: dict[str, Any],
        level: ProtectionLevel | str = ProtectionLevel.PROTECTED,
        *, profile: Profile | str | None = None,
        render: str = "spec") -> SafeResult:
    """Validate, run, and disclosure-check ``code`` against ``sources``.

    ``sources`` maps the names user code may reference (e.g. ``{"df": frame}``)
    to private data objects. ``level`` selects the protection policy; ``profile``
    overrides the executor (OPEN sandbox vs STRICT capability) for that policy.
    ``render`` picks the transport encoding for chart results:
    ``spec`` (default, JSON) | ``plotly`` | ``png`` | ``html`` | ``ascii``.
    """
    policy: Policy = resolve_policy([level])
    active = Profile(profile) if profile is not None else policy.profile
    catalog = None  # datasets left in the session (populated once execution runs)

    try:
        namespace = _build_namespace(active, policy, sources)
        allowed_names = frozenset(namespace)
        # Whitelisted imports (resolving to safe facades) are allowed only in the
        # STRICT capability profile.
        imports_ok = active is Profile.STRICT
        gate = validate(code, allowed_names=allowed_names, allow_imports=imports_ok)
        if not gate.ok:
            assert gate.error is not None
            return SafeResult(ok=False, kind="error", error=gate.error.as_dict())

        expr_values, ns = execute(code, namespace, allow_imports=imports_ok)
        catalog = _build_catalog(ns, policy)

        # Each top-level bare expression is a potential result. Releasable ones are
        # collected; the last expression is the "primary" (top-level fields), which
        # may be a refusal (backward compatible). Non-releasable intermediates
        # (e.g. cph.fit()) are skipped.
        def _stamp(res):
            res.audit.setdefault("level", policy.level.value)
            res.audit.setdefault("profile", active.value)
            res.audit.setdefault("verbs_used", gate.calls)
            if res.kind == "chart" and render != "spec":
                from .charts import render_chart
                res.payload = render_chart(res.payload, render)
                res.audit["render"] = render
            return res

        results, primary = [], None
        for i, value in enumerate(expr_values):
            is_last = i == len(expr_values) - 1
            try:
                res = _stamp(mediate(value, policy))
            except DisclosureError as exc:
                if is_last:  # keep the last refusal as the primary (ok=False)
                    primary = SafeResult(ok=False, kind="error",
                                         error={"kind": type(exc).__name__, "message": str(exc)})
                continue
            results.append(res)
            if is_last:
                primary = res

        if primary is None:  # no bare expressions (datasets-only) -> catalog only
            primary = SafeResult(ok=True, kind="none")
        primary.results = results
        primary.catalog = catalog
        return primary

    except ValidationError as exc:
        return SafeResult(ok=False, kind="error", error=exc.as_dict(), catalog=catalog)
    except (DisclosureError, SandboxError) as exc:
        return SafeResult(ok=False, kind="error", catalog=catalog,
                          error={"kind": type(exc).__name__, "message": str(exc)})
    except SafePythonError as exc:  # pragma: no cover - catch-all, still no data leak
        return SafeResult(ok=False, kind="error", catalog=catalog,
                          error={"kind": "SafePythonError", "message": str(exc)})


def _build_catalog(ns: dict, policy: Policy) -> list:
    """A schema-only catalog of every SafeFrame bound in the session: names,
    columns, dtypes, and suppressed counts (n_rows / n_missing). Never values."""
    from .safeframe import SafeFrame

    k, rt = policy.min_n, policy.round_to

    def count(n: int):
        n = int(n)
        if n == 0:
            return 0                      # "no missing" is not disclosive
        if n < k:
            return None                   # a small nonzero count is suppressed
        return int(round(n / rt) * rt) if rt else n

    catalog = []
    for name, val in ns.items():
        if name.startswith("_") or not isinstance(val, SafeFrame):
            continue
        d = val._df
        columns = [{"name": str(c), "dtype": str(d[c].dtype),
                    "n_missing": count(d[c].isna().sum())} for c in d.columns]
        catalog.append({"name": name, "n_rows": count(len(d)),
                        "n_columns": len(d.columns), "columns": columns})
    return catalog
