"""Average treatment effect via DoWhy propensity methods (df.ate).

Only the aggregate effect + CI are released; matched pairs / propensity scores
are never exposed; binary treatment with each arm >= min_n is required.
"""

import numpy as np
import pandas as pd
import pytest

from safepy import run
from safepy.policy import Profile

pytest.importorskip("dowhy")


def _df(n=800, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    ps = 1 / (1 + np.exp(-(0.5 * x1 + 0.5 * x2)))
    t = (rng.random(n) < ps).astype(int)
    y = 2.0 * t + x1 + x2 + rng.normal(0, 1, n)   # true ATE = 2.0
    return pd.DataFrame({"T": t, "Y": y, "x1": x1, "x2": x2})


DF = _df()


def _strict(code):
    return run(code, {"df": DF}, profile=Profile.STRICT)


def test_ate_weighting_recovers_effect():
    r = _strict("df.ate(outcome='Y', treatment='T', confounders=['x1', 'x2'], method='weighting')")
    assert r.ok and r.payload["estimand"] == "ate"
    assert 1.5 < r.payload["effect"] < 2.5           # recovers ~2.0
    assert r.payload["ci_low"] is not None and r.payload["ci_high"] is not None
    assert set(r.payload["groups"]) == {"0", "1"}


def test_ate_matching():
    r = _strict("df.ate(outcome='Y', treatment='T', confounders=['x1', 'x2'], method='matching')")
    assert r.ok and 1.3 < r.payload["effect"] < 2.7


def test_ate_stratification():
    r = _strict("df.ate(outcome='Y', treatment='T', confounders=['x1'], method='stratification')")
    assert r.ok


def test_non_binary_treatment_refused():
    d = DF.copy()
    d["T"] = np.arange(len(d)) % 3        # 3 levels
    r = run("df.ate(outcome='Y', treatment='T', confounders=['x1'])",
            {"df": d}, profile=Profile.STRICT)
    assert r.ok is False and "binary" in r.error["message"]


def test_small_arm_refused():
    d = DF.copy()
    d.loc[d.index[3:], "T"] = 0           # only 3 treated units
    r = run("df.ate(outcome='Y', treatment='T', confounders=['x1'])",
            {"df": d}, profile=Profile.STRICT)
    assert r.ok is False and "min_n" in r.error["message"]


def test_unknown_column_refused():
    r = _strict("df.ate(outcome='Y', treatment='T', confounders=['nope'])")
    assert r.ok is False
