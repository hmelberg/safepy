"""Phase 4: coverage wins found by the sweep (tools/coverage_sweep.py)."""

import numpy as np
import pandas as pd
import pytest

from safepy import run
from safepy.policy import Profile

pytest.importorskip("statsmodels")


def _df(n=200, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x": rng.normal(10, 3, n), "y": rng.normal(5, 2, n), "z": rng.normal(0, 1, n),
        "a": rng.integers(0, 10, n), "c": rng.integers(0, 5, n),
        "g": [f"g{i}" for i in rng.integers(0, 8, n)],
        "age": rng.integers(20, 70, n), "sex": rng.choice(["F", "M"], n),
        "income": rng.integers(20000, 90000, n),
    })


DF = _df()


def _strict(code):
    return run(code, {"df": DF}, profile=Profile.STRICT)


@pytest.mark.parametrize("code", [
    "df.assign(bnd=pd.cut(df['age'], bins=[0,30,np.inf], labels=[1,2])).groupby('bnd')['income'].mean()",
    "df.assign(xn=pd.to_numeric(df['x'])).groupby('g')['xn'].mean()",
    "df[['x','y']].corr()",
    "df.rename(columns={'a':'aa'}).groupby('g')['x'].mean()",
    "df.fillna(0).groupby('g')['x'].mean()",
    "df.dropna(subset=['income']).groupby('g')['x'].mean()",
    "df.drop(columns=['c']).groupby('g')['x'].mean()",
    "smf.ols('y ~ age*sex', data=df).fit().summary()",
])
def test_phase4_idioms_supported(code):
    assert _strict(code).ok is True


def test_corr_too_few_rows_refused():
    tiny = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    r = run("df.corr()", {"df": tiny}, profile=Profile.STRICT)
    assert r.ok is False  # fewer than min_n rows
