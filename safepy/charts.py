"""Plotting for STRICT mode: charts are renderings of already-suppressed data.

The security rule is structural: a chart can only be built from a ``Released``
aggregate (a suppressed table), never from a ``SafeFrame``/``SafeColumn`` of raw
values. So ``PlotAccessor`` lives on ``Released`` (``value_counts().plot.bar()``)
and its ``_make`` refuses anything that isn't an aggregated table. Because the
data is already suppressed, the earlier warning about figures embedding raw
arrays does not apply — the only array at plot time is the suppressed aggregate.

``render_chart`` turns a chart spec into a transport encoding chosen by the API
caller (``spec`` | ``plotly`` | ``png`` | ``html`` | ``ascii``). The spec is the
security boundary; the encoding is a pure function of it.
"""

from __future__ import annotations

from .errors import DisclosureError
from .result import Released

_AGG_KINDS = frozenset({"bar", "barh", "line", "area", "pie", "hist"})


def chart_released(chart_type: str, data_payload: dict, audit: dict) -> Released:
    return Released({"type": "chart", "chart_type": chart_type, "data": data_payload},
                    audit={**audit, "kind": "chart", "chart": chart_type})


class PlotAccessor:
    """``Released.plot`` — pandas-like ``.plot.bar()`` on an aggregate table."""

    def __init__(self, released: Released):
        self._r = released

    def __call__(self, kind: str = "bar", **kw): return self._make(kind)
    def bar(self, **kw): return self._make("bar")
    def barh(self, **kw): return self._make("barh")
    def line(self, **kw): return self._make("line")
    def area(self, **kw): return self._make("area")
    def pie(self, **kw): return self._make("pie")

    # raw-data plot kinds are refused with guidance
    def hist(self, *a, **k):
        raise DisclosureError("hist is a raw-data plot; call .hist() on a column "
                              "(it bins and suppresses), not on an aggregate")
    def box(self, *a, **k):
        raise DisclosureError("box plots draw individual outliers and are not available")
    def scatter(self, *a, **k):
        raise DisclosureError("scatter plots one point per individual and is not available")

    def _make(self, kind: str) -> Released:
        p = self._r.payload
        if not isinstance(p, dict) or p.get("type") not in ("series", "frame"):
            raise DisclosureError("only an aggregated table can be plotted")
        return chart_released(kind, p, self._r.audit)


# ── rendering ────────────────────────────────────────────────────────────────

def render_chart(spec: dict, fmt: str):
    if fmt == "spec":
        return spec
    if fmt == "ascii":
        return {"format": "ascii", "content": _ascii(spec)}
    if fmt == "plotly":
        return {"format": "plotly", "content": _build_fig(spec).to_json()}
    if fmt == "html":
        return {"format": "html",
                "content": _build_fig(spec).to_html(full_html=False, include_plotlyjs="cdn")}
    if fmt == "png":
        return {"format": "png", "content": _png(spec)}
    raise DisclosureError(f"unknown render format: {fmt!r}")


def _zeros(values):
    return [0 if v is None else v for v in values]


def _build_fig(spec: dict):
    import plotly.graph_objects as go

    d, ct = spec["data"], spec["chart_type"]
    if ct == "box" or d.get("type") == "box":
        st = d["stats"]
        kw = {"name": str(d.get("name", "")), "q1": [st["q1"]],
              "median": [st["median"]], "q3": [st["q3"]]}
        if st.get("min") is not None:
            kw["lowerfence"] = [st["min"]]
        if st.get("max") is not None:
            kw["upperfence"] = [st["max"]]
        return go.Figure(go.Box(**kw))
    if d["type"] == "series":
        x, y = d["index"], _zeros(d["values"])
        if ct == "pie":
            fig = go.Figure(go.Pie(labels=x, values=y))
        elif ct == "barh":
            fig = go.Figure(go.Bar(x=y, y=x, orientation="h"))
        elif ct in ("line", "area"):
            fig = go.Figure(go.Scatter(x=x, y=y, fill="tozeroy" if ct == "area" else None))
        else:  # bar, hist
            fig = go.Figure(go.Bar(x=x, y=y))
        fig.update_layout(title=d.get("name", ""))
    else:  # frame -> grouped bars
        fig = go.Figure()
        for j, col in enumerate(d["columns"]):
            fig.add_bar(name=str(col), x=d["index"], y=_zeros([row[j] for row in d["data"]]))
    return fig


def _png(spec: dict) -> str:
    import base64
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d, ct = spec["data"], spec["chart_type"]
    fig, ax = plt.subplots(figsize=(6, 4))
    if ct == "box" or d.get("type") == "box":
        st = d["stats"]
        bx = {"med": st["median"], "q1": st["q1"], "q3": st["q3"],
              "whislo": st["min"] if st["min"] is not None else st["q1"],
              "whishi": st["max"] if st["max"] is not None else st["q3"],
              "fliers": [], "label": str(d.get("name", ""))}
        ax.bxp([bx], showfliers=False)
    elif d["type"] == "series":
        x, y = d["index"], _zeros(d["values"])
        if ct == "line" or ct == "area":
            ax.plot(x, y)
        elif ct == "barh":
            ax.barh(x, y)
        elif ct == "pie":
            ax.pie(y, labels=x)
        else:
            ax.bar(x, y)
        ax.set_title(d.get("name", ""))
        if ct != "pie":
            fig.autofmt_xdate(rotation=45)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _ascii(spec: dict) -> str:
    d = spec["data"]
    if d.get("type") == "box":
        st = d["stats"]
        def c(v): return "(suppressed)" if v is None else v
        return (f"{d.get('name','')}  box (outliers omitted)\n"
                f"  min={c(st['min'])}  q1={c(st['q1'])}  median={c(st['median'])}  "
                f"q3={c(st['q3'])}  max={c(st['max'])}")
    if d["type"] != "series":
        return "(table chart)"
    vals = [v for v in d["values"] if v is not None]
    mx = max(vals) if vals else 1
    lines = []
    for lbl, v in zip(d["index"], d["values"]):
        if v is None:
            bar, tail = "(suppressed)", ""
        else:
            bar = "#" * int(round(24 * v / mx)) if mx else ""
            tail = f" {v}"
        lines.append(f"{str(lbl)[:14]:14} | {bar}{tail}")
    return "\n".join(lines)
