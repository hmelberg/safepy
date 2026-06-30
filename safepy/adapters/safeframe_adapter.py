"""Refuse a bare SafeFrame returned as the final result.

In STRICT mode the shaping verbs (``where``, ``assign``) return a SafeFrame.
If user code ends on one of those instead of an aggregation, there is nothing
to release — and we must not fall back to anything that reveals rows. This
adapter claims SafeFrame and refuses it with guidance, so the failure is a clear
message rather than a leak.
"""

from __future__ import annotations

from typing import Any

from ..errors import DisclosureError
from ..policy import Policy
from ..result import SafeResult
from ..safeframe import SafeFrame
from . import base


class SafeFrameAdapter:
    name = "safeframe"

    def claims(self, result: Any) -> bool:
        return isinstance(result, SafeFrame)

    def make_safe(self, result: Any, policy: Policy) -> SafeResult:
        raise DisclosureError(
            "a SafeFrame is not a releasable result. End on an aggregation "
            "(e.g. .groupby(...).mean(...), .value_counts(...), .ols(...))."
        )


base.register(SafeFrameAdapter())
