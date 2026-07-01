"""Fixed-effects / IV / DiD via pyfixest (df.feols, df.iv), plus the
already-supported causal patterns (DiD via interactions)."""

import numpy as np
import pandas as pd
import pytest

from safepy import run
from safepy.policy import Profile

pytest.importorskip("pyfixest")


def _panel(n=800, seed=0):
    rng = np.random.default_rng(seed)
    firm = rng.integers(0, 25, n)
    year = rng.integers(2000, 2006, n)
    x = rng.normal(0, 1, n)
    z = rng.normal(0, 1, n)
    endog = 0.6 * z + rng.normal(0, 1, n)
    y = 2 * x + 1.5 * endog + 0.1 * firm + 0.2 * (year - 2000) + rng.normal(0, 1, n)
    return pd.DataFrame({"y": y, "x": x, "z": z, "endog": endog,
                         "firm": firm.astype(str), "year": year.astype(str)})


DF = _panel()


def _strict(code):
    return run(code, {"df": DF}, profile=Profile.STRICT)


def test_feols_two_way_fixed_effects():
    r = _strict("df.feols(y='y', x=['x'], fe=['firm', 'year'])")
    assert r.ok and r.payload["family"] == "feols"
    terms = {t["term"]: t["coef"] for t in r.payload["terms"]}
    assert "x" in terms and terms["x"] is not None
    # fixed effects are absorbed, never reported
    assert not any("firm" in t["term"] for t in r.payload["terms"])
    assert r.payload["fixed_effects"] == ["firm", "year"]


def test_feols_clustered_se():
    r = _strict("df.feols(y='y', x=['x'], fe=['firm'], cluster='firm')")
    assert r.ok and r.audit["backend"] == "pyfixest"
    assert r.payload["cluster"] == "firm"
    for t in r.payload["terms"]:
        assert t["se"] is not None


def test_iv_two_stage():
    r = _strict("df.iv(y='y', x=['x'], endog='endog', instruments=['z'], fe=['firm'])")
    assert r.ok and r.payload["family"] == "iv"
    terms = {t["term"] for t in r.payload["terms"]}
    assert "endog" in terms and "x" in terms


def test_did_via_interaction_still_works():
    # difference-in-differences needs no new library: a treat*post interaction
    rng = np.random.default_rng(1)
    n = 600
    treat = rng.integers(0, 2, n)
    post = rng.integers(0, 2, n)
    y = 3 + 1.5 * treat + 0.8 * post + 2.0 * treat * post + rng.normal(0, 1, n)
    d = pd.DataFrame({"y": y, "treat": treat, "post": post})
    r = run("smf.ols('y ~ treat*post', data=df).fit().summary()",
            {"df": d}, profile=Profile.STRICT)
    assert r.ok
    did = next(t["coef"] for t in r.payload["terms"] if t["term"] == "treat:post")
    assert 1.5 < did < 2.5  # recovers the true DiD effect ~2.0
