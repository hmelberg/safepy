"""Follow-ups to df.ate: propensity scores (private column) and refutation."""

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
    y = 2.0 * t + x1 + x2 + rng.normal(0, 1, n)
    return pd.DataFrame({"T": t, "Y": y, "x1": x1, "x2": x2})


DF = _df()


def _strict(code):
    return run(code, {"df": DF}, profile=Profile.STRICT)


# ---- propensity scores are a private column ---------------------------------

def test_propensity_mean_is_a_valid_probability():
    r = _strict("df.propensity(treatment='T', confounders=['x1', 'x2']).mean()")
    assert r.ok and r.kind == "scalar"
    assert 0.0 <= r.payload["value"] <= 1.0


def test_propensity_histogram():
    r = _strict("df.propensity(treatment='T', confounders=['x1', 'x2']).hist(bins=5)")
    assert r.ok and r.kind == "chart"


def test_propensity_overlap_by_group_via_assign():
    # common-support: mean propensity by treatment arm
    r = _strict("df.assign(ps=df.propensity(treatment='T', confounders=['x1','x2']))"
                ".groupby('T')['ps'].mean()")
    assert r.ok and r.kind == "table"


def test_raw_propensity_column_refused():
    r = _strict("df.propensity(treatment='T', confounders=['x1', 'x2'])")
    assert r.ok is False and "intermediate" in r.error["message"]


def test_propensity_values_never_revealed():
    r = _strict("df.propensity(treatment='T', confounders=['x1']).values")
    assert r.ok is False


# ---- refutation (aggregate robustness check) --------------------------------

def test_refute_surfaces_original_and_refuted_effects():
    r = _strict("df.refute_ate(outcome='Y', treatment='T', confounders=['x1','x2'], refuter='placebo')")
    assert r.ok and r.payload["type"] == "causal_refutation"
    # the wrapper surfaces both the original estimate (~2.0) and the refuted one
    assert 1.5 < r.payload["estimated_effect"] < 2.5
    assert r.payload["new_effect"] is not None
