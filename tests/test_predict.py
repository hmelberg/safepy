"""Predictions/residuals are private columns (SafeColumn): aggregate/plot them,
never see individual values."""

import numpy as np
import pandas as pd
import pytest

from safepy import run
from safepy.policy import Profile

pytest.importorskip("statsmodels")


def _df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 70, n)
    y = 2 * age + rng.normal(0, 5, n)
    return pd.DataFrame({"age": age, "y": y, "sex": rng.choice(["F", "M"], n)})


DF = _df()


def _strict(code):
    return run(code, {"df": DF}, profile=Profile.STRICT)


def test_predict_can_be_aggregated():
    r = _strict('smf.ols("y ~ age", data=df).fit().predict().mean()')
    assert r.ok and r.kind == "scalar"


def test_predict_can_be_histogrammed():
    r = _strict('smf.ols("y ~ age", data=df).fit().predict().hist(bins=5)')
    assert r.ok and r.kind == "chart"


def test_residuals_are_a_safe_column():
    r = _strict('smf.ols("y ~ age", data=df).fit().resid.mean()')
    assert r.ok and r.kind == "scalar"


def test_raw_prediction_is_refused():
    # a bare predicted column is a dangling SafeColumn -> refused (never revealed)
    r = _strict('smf.ols("y ~ age", data=df).fit().predict()')
    assert r.ok is False and "intermediate" in r.error["message"]


def test_prediction_values_never_revealed():
    # the individual-level accessors don't exist on the predicted column
    for tail in ("predict().values", "predict().tolist()", "predict()[0]"):
        r = _strict(f'smf.ols("y ~ age", data=df).fit().{tail}')
        assert r.ok is False
