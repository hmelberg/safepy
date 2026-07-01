"""Survival analysis (lifelines) in STRICT mode: cox, kaplan_meier, logrank.

Asserts aggregate summaries/tests are released, per-subject outputs are
unreachable, and small groups / curve tails are suppressed.
"""

import numpy as np
import pandas as pd
import pytest

from safepy import run
from safepy.policy import Profile

pytest.importorskip("lifelines")


def _df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.integers(30, 80, n)
    sex = rng.choice(["F", "M"], n)
    dur = rng.exponential(20, n).round(1) + 1
    died = (rng.random(n) < 0.5).astype(int)
    return pd.DataFrame({"age": age, "sex": sex, "dur": dur, "died": died})


DF = _df()


def _strict(code):
    return run(code, {"df": DF}, profile=Profile.STRICT)


def test_cox_releases_hazard_ratios():
    r = _strict("df.cox(duration='dur', event='died', x=['age', 'sex'])")
    assert r.ok and r.payload["family"] == "cox"
    assert all("hazard_ratio" in t for t in r.payload["terms"])


def test_kaplan_meier_curve_and_median():
    r = _strict("df.kaplan_meier(duration='dur', event='died', by='sex')")
    assert r.ok and r.payload["method"] == "kaplan_meier"
    for curve in r.payload["curves"].values():
        assert len(curve["time"]) == len(curve["survival"]) > 0
        assert "median" in curve


def test_logrank_test():
    r = _strict("df.logrank(duration='dur', event='died', by='sex')")
    assert r.ok and r.payload["test"] == "logrank"
    assert r.payload["statistic"] is not None and r.payload["p_value"] is not None
    assert set(r.payload["groups"]) == {"F", "M"}


def test_per_subject_outputs_unreachable():
    # there is no verb returning predictions/residuals, and lifelines classes
    # are not in scope
    assert _strict("CoxPHFitter().fit(df, 'dur', 'died').predict_partial_hazard(df)").ok is False


def test_logrank_needs_two_groups_of_min_n():
    tiny = pd.DataFrame({"dur": [1.0, 2.0, 3.0], "died": [1, 0, 1],
                         "g": ["a", "a", "b"]})
    r = run("df.logrank(duration='dur', event='died', by='g')",
            {"df": tiny}, profile=Profile.STRICT)
    assert r.ok is False


def test_weibull_aft():
    r = _strict("df.weibull_aft(duration='dur', event='died', x=['age', 'sex'])")
    assert r.ok and r.payload["family"] == "weibull_aft"
    assert any("age" in t["term"] for t in r.payload["terms"])


def test_lognormal_aft():
    r = _strict("df.lognormal_aft(duration='dur', event='died', x=['age'])")
    assert r.ok and r.payload["family"] == "lognormal_aft"


def test_rmst_by_group():
    r = _strict("df.rmst(duration='dur', event='died', t=20, by='sex')")
    assert r.ok and r.payload["type"] == "rmst"
    assert set(r.payload["values"]) == {"F", "M"}
