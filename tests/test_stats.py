"""Regression (statsmodels) and survival (lifelines) safe verbs.

Asserts: aggregate summaries are released; per-observation outputs and raw
formula strings are not reachable; coefficients/curves backed by fewer than
min_n individuals are suppressed.
"""

import numpy as np
import pandas as pd
import pytest

from safepy import run, ProtectionLevel
from safepy.policy import Profile

sm = pytest.importorskip("statsmodels")


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 70, n)
    sex = rng.choice(["F", "M"], n)
    # a tiny category level (3 people) to exercise coefficient suppression
    region = np.where(np.arange(n) < 3, "Z", rng.choice(["A", "B"], n))
    salary = 20000 + 800 * age + 5000 * (sex == "M") + rng.normal(0, 3000, n)
    died = (rng.random(n) < 0.3).astype(int)
    dur = rng.integers(1, 100, n)
    return pd.DataFrame({"age": age, "sex": sex, "region": region,
                         "salary": salary, "died": died, "dur": dur})


DF = _data()


def _run(code, level=ProtectionLevel.PROTECTED):
    return run(code, {"df": DF}, level)


# ---- OLS / GLM via safe verbs ----------------------------------------------

def test_ols_releases_coefficients():
    r = _run("safe.ols(df, y='salary', x=['age', 'sex'])")
    assert r.ok and r.payload["type"] == "regression"
    terms = {t["term"] for t in r.payload["terms"]}
    assert any("age" in t for t in terms)
    # every released row is a coefficient, never a row of data
    for t in r.payload["terms"]:
        assert set(t) >= {"term", "coef", "ci_low", "ci_high", "pvalue"}


def test_logit_runs():
    r = _run("safe.logit(df, y='died', x=['age'])")
    assert r.ok and r.payload["family"] == "logit"


def test_small_category_coefficient_suppressed():
    # region has a level 'Z' with only 3 members (< min_n=5); its dummy
    # coefficient must be blanked.
    r = _run("safe.ols(df, y='salary', x=['region'])")
    assert r.ok
    zrows = [t for t in r.payload["terms"] if "Z" in t["term"]]
    assert zrows and all(t["coef"] is None for t in zrows)
    assert any("Z" in s for s in r.audit["terms_suppressed"])


def test_raw_formula_string_is_refused():
    # a free-text formula would be patsy-eval'd; safe verbs never accept one.
    r = _run("safe.ols(df, y='salary ~ age', x=['age'])")
    assert r.ok is False
    assert "identifier" in r.error["message"] or "invalid column" in r.error["message"]


def test_per_observation_outputs_unreachable():
    # there is no verb that returns predictions/residuals; the smf module is not
    # in scope either.
    assert _run("smf.ols('salary ~ age', data=df).fit().predict()").ok is False


def test_ols_works_in_strict_profile():
    r = run("df.ols(y='salary', x=['age', 'sex'])", {"df": DF}, profile=Profile.STRICT)
    assert r.ok and r.payload["type"] == "regression"


# ---- survival via lifelines -------------------------------------------------

lifelines = pytest.importorskip("lifelines")


def test_kaplan_meier_releases_curve():
    r = _run("safe.kaplan_meier(df, duration='dur', event='died')")
    assert r.ok and r.payload["method"] == "kaplan_meier"
    curve = r.payload["curves"]["all"]
    assert len(curve["time"]) == len(curve["survival"]) > 0


def test_cox_releases_hazard_ratios():
    r = _run("safe.cox(df, duration='dur', event='died', x=['age', 'sex'])")
    assert r.ok and r.payload["family"] == "cox"
    assert all("hazard_ratio" in t for t in r.payload["terms"])


def test_cox_works_in_strict_profile():
    r = run("df.cox(duration='dur', event='died', x=['age'])",
            {"df": DF}, profile=Profile.STRICT)
    assert r.ok
