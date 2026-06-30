"""safepython — run a familiar subset of Python against private data without ever
revealing individual-level rows.

Posture: a *sandbox* that runs AST-gated user Python directly (the hard,
research path), scoped to public/local data, standalone for now but shaped to
fold into m2py as a Python *frontend* later. Result-side and data-side
protection are delegated to the existing ``protect`` package, not reimplemented.

Public surface:
    run(code, sources, level) -> SafeResult
"""

from .api import run
from .policy import ProtectionLevel
from .result import SafeResult

__all__ = ["run", "ProtectionLevel", "SafeResult"]
