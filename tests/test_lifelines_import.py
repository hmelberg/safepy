"""Idiomatic lifelines usage: `from lifelines import CoxPHFitter` etc.

The imported names resolve to safe facades; only aggregates/curves are released,
predictions are private SafeColumns, and non-whitelisted imports are refused.
"""

import numpy as np
import pandas as pd
import pytest

from safepy import run, ProtectionLevel
from safepy.policy import Profile

pytest.importorskip("lifelines")


def _df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age": rng.integers(30, 80, n), "sex": rng.choice(["F", "M"], n),
        "dur": (rng.exponential(20, n) + 1).round(1),
        "died": (rng.random(n) < 0.5).astype(int)})


DF = _df()


def _strict(code):
    return run(code, {"df": DF}, profile=Profile.STRICT)


def test_cox_idiomatic_summary():
    code = ("from lifelines import CoxPHFitter\n"
            "CoxPHFitter().fit(df, duration_col='dur', event_col='died').summary")
    r = _strict(code)
    assert r.ok and r.payload["family"] == "cox"


def test_cox_partial_hazard_is_private_column():
    code = ("from lifelines import CoxPHFitter\n"
            "cph = CoxPHFitter()\n"
            "cph.fit(df, duration_col='dur', event_col='died')\n"
            "cph.predict_partial_hazard().mean()")
    r = _strict(code)
    assert r.ok and r.kind == "scalar"


def test_cox_partial_hazard_raw_refused():
    code = ("from lifelines import CoxPHFitter\n"
            "CoxPHFitter().fit(df, duration_col='dur', event_col='died').predict_partial_hazard()")
    r = _strict(code)
    assert r.ok is False and "intermediate" in r.error["message"]


def test_kaplan_meier_idiomatic():
    code = ("from lifelines import KaplanMeierFitter\n"
            "KaplanMeierFitter().fit(df['dur'], df['died']).median_survival_time_")
    r = _strict(code)
    assert r.ok and r.kind == "scalar"


def test_km_plot():
    code = ("from lifelines import KaplanMeierFitter\n"
            "KaplanMeierFitter().fit(df['dur'], df['died']).plot()")
    r = _strict(code)
    assert r.ok and r.kind == "chart"


def test_weibull_aft_idiomatic():
    code = ("from lifelines import WeibullAFTFitter\n"
            "WeibullAFTFitter().fit(df, duration_col='dur', event_col='died').summary")
    assert _strict(code).ok


def test_dangling_fitter_refused():
    code = ("from lifelines import CoxPHFitter\n"
            "CoxPHFitter().fit(df, duration_col='dur', event_col='died')")
    r = _strict(code)
    assert r.ok is False and "intermediate" in r.error["message"]


# ---- import safety ----------------------------------------------------------

def test_unavailable_lifelines_class_refused():
    code = "from lifelines import AalenJohansenFitter\nAalenJohansenFitter()"
    assert _strict(code).ok is False


@pytest.mark.parametrize("code", [
    "import os\nos.getcwd()",
    "from subprocess import run as r\nr('x')",
    "import sys\nsys",
])
def test_forbidden_imports_blocked(code):
    assert _strict(code).ok is False


def test_imports_disallowed_in_open_profile():
    code = ("from lifelines import CoxPHFitter\n"
            "CoxPHFitter().fit(df, duration_col='dur', event_col='died').summary")
    r = run(code, {"df": DF}, ProtectionLevel.PUBLIC)  # OPEN profile
    assert r.ok is False
