"""Multiple-results envelope: each top-level bare expression is a potential
result; releasable ones are collected in .results, the last is the primary."""

import numpy as np
import pandas as pd

from safepy import run
from safepy.policy import Profile


def _df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "sex": rng.choice(["F", "M"], n),
        "region": rng.choice(["A", "B"], n),
        "age": rng.integers(20, 70, n),
        "salary": rng.integers(30000, 90000, n).astype(float),
    })


DF = _df()


def _strict(code):
    return run(code, {"df": DF}, profile=Profile.STRICT)


def test_multiple_results_collected():
    code = ("df.groupby('sex')['salary'].mean()\n"
            "df['region'].value_counts()\n"
            "df['salary'].mean()")
    r = _strict(code)
    assert r.ok and len(r.results) == 3
    assert r.results[0].payload["name"] == "mean(salary)"
    assert r.results[2].kind == "scalar"
    # top-level fields mirror the last (primary) result
    assert r.payload == r.results[-1].payload


def test_backward_compat_single_result():
    r = _strict("df.groupby('sex')['salary'].mean()")
    assert r.ok and r.kind == "table"
    assert len(r.results) == 1 and r.results[0].payload == r.payload


def test_non_releasable_intermediates_are_skipped():
    # a bare column expression is an intermediate -> skipped, not a result
    code = "df['salary']\ndf.groupby('sex')['salary'].mean()"
    r = _strict(code)
    assert r.ok and len(r.results) == 1     # only the mean


def test_last_refusal_is_primary_error():
    r = _strict("df.groupby('sex')['salary'].mean()\ndf['salary']")
    assert r.ok is False                    # last expr (a column) refused
    assert len(r.results) == 1              # the mean still collected


def test_datasets_only_script_is_ok_with_catalog():
    code = "new_df = df.summarise('sex', pay=('salary','mean'))"
    r = _strict(code)
    assert r.ok and r.kind == "none"
    assert r.results == []
    assert any(d["name"] == "new_df" for d in r.catalog)


def test_fit_then_summary_idiom_still_works():
    code = ("m = smf.ols('salary ~ age', data=df).fit()\n"
            "m.summary()")
    r = _strict(code)
    assert r.ok and len(r.results) == 1 and r.results[0].payload["type"] == "regression"


def test_as_dict_serialises_results_without_recursion():
    code = "df['salary'].mean()\ndf['age'].mean()"
    d = _strict(code).as_dict()
    assert isinstance(d["results"], list) and len(d["results"]) == 2
    assert set(d["results"][0]) == {"ok", "kind", "payload", "audit", "error"}
    assert "catalog" in d
