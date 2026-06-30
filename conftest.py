"""Make safepython and the sibling ``protect`` repo importable without install."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_PROTECT = _ROOT.parent / "protect"

for p in (_ROOT, _PROTECT):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
