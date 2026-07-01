"""Phase A: hypothesis tests, grouped describe, marginal effects.

Test statistics/p-values are aggregates -> released once every contributing
group has >= min_n observations."""

import numpy as np
import pandas as pd
import pytest

from safepy import run
from safepy.policy import Profile


def _df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    sex = rng.choice(["F", "M"], n)
    region = np.where(np.arange(n) < 3, "Z", rng.choice(["A", "B"], n))  # Z tiny
    age = rng.integers(20, 70, n)
    salary = 30000 + 400 * age + 3000 * (sex == "M") + rng.normal(0, 4000, n)
    died = (rng.random(n) < 0.3).astype(int)
    return pd.DataFrame({"sex": sex, "region": region, "age": age,
                         "salary": salary, "died": died})


DF = _df()


def _strict(code):
    return run(code, {"df": DF}, profile=Profile.STRICT)


# ---- hypothesis tests -------------------------------------------------------

def test_two_sample_ttest():
    r = _strict("df.ttest(value='salary', by='sex')")
    assert r.ok and r.kind == "test"
    assert r.payload["test"].startswith("two-sample t")
    assert r.payload["p_value"] is not None and set(r.payload["groups"]) == {"F", "M"}


def test_one_sample_ttest():
    r = _strict("df.ttest(value='salary', mu=50000)")
    assert r.ok and r.payload["test"] == "one-sample t"


def test_anova():
    r = _strict("df.anova(value='salary', by='sex')")
    assert r.ok and r.payload["test"] == "one-way ANOVA"
    assert set(r.payload["df"]) == {"between", "within"}


def test_chisq():
    r = _strict("df.chisq(row='sex', col='died')")
    assert r.ok and r.payload["test"] == "chi-square" and r.payload["df"] is not None


def test_corr_test():
    r = _strict("df.corr_test(x='age', y='salary', method='pearson')")
    assert r.ok and "correlation" in r.payload["test"]


def test_mannwhitney():
    r = _strict("df.mannwhitney(value='salary', by='sex')")
    assert r.ok and r.payload["test"] == "Mann-Whitney U"


def test_ttest_refuses_more_than_two_groups():
    r = _strict("df.ttest(value='salary', by='region')")   # 3 groups
    assert r.ok is False


def test_test_refuses_small_group():
    # a 2-group split where one arm < min_n
    d = DF.copy()
    d["grp"] = np.where(np.arange(len(d)) < 3, "x", "y")   # x has 3 rows
    r = run("df.ttest(value='salary', by='grp')", {"df": d}, profile=Profile.STRICT)
    assert r.ok is False


def test_chisq_never_releases_the_table():
    r = _strict("df.chisq(row='sex', col='died')")
    assert "data" not in r.payload and "index" not in r.payload  # only the statistic


# ---- grouped describe -------------------------------------------------------

def test_grouped_describe():
    r = _strict("df.groupby('sex')['salary'].describe()")
    assert r.ok and r.kind == "table"
    assert r.payload["columns"][:3] == ["count", "mean", "std"]
    assert set(r.payload["index"]) == {"F", "M"}


def test_grouped_describe_suppresses_small_group():
    r = _strict("df.groupby('region')['salary'].describe()")
    assert r.ok
    rows = dict(zip(r.payload["index"], r.payload["data"]))
    assert all(v is None for v in rows["Z"])   # region Z (n<min_n) fully suppressed


# ---- marginal effects -------------------------------------------------------

def test_marginal_effects_logit():
    r = _strict("smf.logit('died ~ age + C(sex)', data=df).fit().margeff()")
    assert r.ok and r.payload["type"] == "marginal_effects"
    assert any("age" in t["term"] for t in r.payload["terms"])


def test_margeff_unavailable_for_ols():
    r = _strict("smf.ols('salary ~ age', data=df).fit().margeff()")
    assert r.ok is False
