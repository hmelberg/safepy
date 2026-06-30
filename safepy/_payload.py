"""Render-ready payload conversion. One place, so NaN->None is correct everywhere.

Suppressed cells come back from ``protect.suppress`` as NaN. In a float Series
``.where(cond, None)`` coerces None straight back to NaN, so we must convert at
``tolist`` time, explicitly.
"""

from __future__ import annotations

import pandas as pd


def _clean(v):
    return None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)


def series_payload(s: pd.Series, *, name=None) -> dict:
    return {"type": "series",
            "name": str(name if name is not None else s.name),
            "index": [str(i) for i in s.index.tolist()],
            "values": [_clean(v) for v in s.tolist()]}


def frame_payload(df: pd.DataFrame) -> dict:
    return {"type": "frame",
            "columns": [str(c) for c in df.columns],
            "index": [str(i) for i in df.index.tolist()],
            "data": [[_clean(v) for v in row] for row in df.values.tolist()]}
