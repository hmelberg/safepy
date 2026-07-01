"""Phase 5: plotting. Charts render already-suppressed aggregates only."""

import numpy as np
import pandas as pd
import pytest

from safepy import run
from safepy.policy import Profile


def _df(n=200, seed=2):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "sex": rng.choice(["F", "M"], n),
        "region": np.where(np.arange(n) < 3, "Z", rng.choice(["A", "B"], n)),
        "age": rng.integers(20, 70, n),
        "salary": rng.integers(30000, 90000, n),
    })


DF = _df()


def _strict(code, **kw):
    return run(code, {"df": DF}, profile=Profile.STRICT, **kw)


# ---- plotting an aggregate works --------------------------------------------

def test_value_counts_plot_bar():
    r = _strict("df['region'].value_counts().plot.bar()")
    assert r.ok and r.kind == "chart"
    assert r.payload["chart_type"] == "bar"
    # underlying data is the already-suppressed table (region Z is None)
    data = dict(zip(r.payload["data"]["index"], r.payload["data"]["values"]))
    assert data["Z"] is None


def test_groupby_mean_plot_barh():
    r = _strict("df.groupby('sex')['salary'].mean().plot.barh()")
    assert r.ok and r.payload["chart_type"] == "barh"


def test_plot_call_default_kind():
    r = _strict("df['region'].value_counts().plot()")
    assert r.ok and r.kind == "chart"


# ---- histogram is redirected to a suppressed binned frequency ---------------

def test_column_hist_is_binned_and_suppressed():
    r = _strict("df['age'].hist(bins=5)")
    assert r.ok and r.kind == "chart" and r.payload["chart_type"] == "hist"
    assert r.audit["verb"] == "hist"


def test_plot_hist_accessor():
    r = _strict("df['salary'].plot.hist()")
    assert r.ok and r.kind == "chart"


# ---- raw-data plotting stays blocked ----------------------------------------

@pytest.mark.parametrize("code", [
    "df.plot()",                       # whole-frame raw plot: no such method
    "df['salary'].plot.line()",        # raw column line
    "df['salary'].plot.scatter()",     # raw scatter
    "df.groupby('sex')['salary'].mean().plot.scatter()",  # scatter refused even on agg
    "df['salary'].mean().plot.bar()",  # a scalar can't be plotted
])
def test_raw_plotting_blocked(code):
    assert _strict(code).ok is False


# ---- render formats ---------------------------------------------------------

@pytest.mark.parametrize("fmt,check", [
    ("spec",   lambda p: p["type"] == "chart"),
    ("ascii",  lambda p: p["format"] == "ascii" and isinstance(p["content"], str)),
    ("plotly", lambda p: p["format"] == "plotly" and "data" in p["content"]),
    ("png",    lambda p: p["format"] == "png" and p["content"].startswith("data:image/png")),
    ("html",   lambda p: p["format"] == "html" and "<" in p["content"]),
])
def test_render_formats(fmt, check):
    r = _strict("df['region'].value_counts().plot.bar()", render=fmt)
    assert r.ok and check(r.payload)
