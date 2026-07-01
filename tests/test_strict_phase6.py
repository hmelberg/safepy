"""Phase 6: order statistics (min/max/quantile/describe/boxplot) under the rule
that a value is releasable iff >= min_n observations lie at or beyond it."""

import numpy as np
import pandas as pd

from safepy import run
from safepy.policy import Profile


def _df():
    # 'uniq' has all-distinct values (unique extremes -> suppressed raw min/max);
    # 'rounded' has many ties at each value (shared extremes -> released);
    # 'flag' is boolean (0/1 shared by many).
    n = 200
    rng = np.random.default_rng(4)
    return pd.DataFrame({
        "uniq": np.arange(n) * 1.0 + rng.random(n),
        "rounded": rng.integers(0, 5, n) * 10,       # values in {0,10,20,30,40}
        "flag": (rng.random(n) < 0.4).astype(int),
        "grp": rng.choice(["A", "B"], n),
    })


DF = _df()


def _strict(code, **kw):
    return run(code, {"df": DF}, profile=Profile.STRICT, **kw)


# ---- extremes: shared -> released, unique -> suppressed ----------------------

def test_shared_max_released():
    r = _strict("df['rounded'].max()")
    assert r.ok and r.kind == "scalar"
    assert r.payload["value"] == 40 and r.payload["n"] >= 5


def test_unique_max_suppressed():
    r = _strict("df['uniq'].max()")
    assert r.ok and r.kind == "scalar"
    assert r.payload["value"] is None            # unique extreme -> blanked
    assert r.audit["suppressed"] is True


def test_boolean_min_max_released():
    assert _strict("df['flag'].max()").payload["value"] == 1
    assert _strict("df['flag'].min()").payload["value"] == 0


# ---- winsorize releases a shared bound --------------------------------------

def test_winsorized_max_released():
    r = _strict("df['uniq'].max(winsorize=0.1)")
    assert r.ok and r.payload["value"] is not None
    assert r.audit["winsorized"] == 0.1 and r.payload["n"] >= 5


# ---- median / quartiles are released (interior), extremes checked -----------

def test_describe_blanks_only_extremes_for_continuous():
    r = _strict("df['uniq'].describe()")
    assert r.ok
    s = r.payload["stats"]
    assert s["mean"] is not None and s["50%"] is not None      # median released
    assert s["25%"] is not None and s["75%"] is not None       # quartiles released
    assert s["min"] is None and s["max"] is None               # unique extremes blanked


def test_quantile_interior_released_tail_suppressed():
    assert _strict("df['uniq'].quantile(0.5)").payload["value"] is not None   # median-ish
    assert _strict("df['uniq'].quantile(0.999)").payload["value"] is None     # extreme tail


# ---- boxplot: 5-number summary, no outliers, renders --------------------------

def test_boxplot_omits_outliers_and_renders():
    r = _strict("df['rounded'].plot.box()", render="ascii")
    assert r.ok and r.kind == "chart"
    assert "outliers omitted" in r.payload["content"]


def test_boxplot_plotly_render():
    r = _strict("df['rounded'].boxplot()", render="plotly")
    assert r.ok and r.payload["format"] == "plotly"


# ---- categorical extremes are refused ---------------------------------------

def test_categorical_max_refused():
    r = _strict("df['grp'].max()")
    assert r.ok is False and "numeric" in r.error["message"]
