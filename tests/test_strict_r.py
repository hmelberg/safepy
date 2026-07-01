"""The R dialect (STRICT) — first slice.

R is a *different language*, so it does not ride the Python AST gate. Instead a
restricted dplyr/base-R surface is **parsed and translated** (never executed) to
the same backend-neutral release core (SafeVerbs) the pandas/polars dialects use.
The load-bearing property is equivalence: an R pipeline produces the same
suppressed ``Released`` as the pandas equivalent.
"""

import pytest

from safepy import run
from safepy.policy import Profile
from tests.fixtures import salaries

PDF = salaries()          # pid, name, sex, region, salary; region 'Z' has n=2


def _pandas(code):
    return run(code, {"df": PDF}, profile=Profile.STRICT)


def _r(code):
    return run(code, {"df": PDF}, profile=Profile.STRICT, dialect="r")


def _as_dict(payload):
    return dict(zip(payload["index"], payload["values"]))


def test_group_by_summarise_matches_pandas():
    p = _pandas("df.groupby('sex')['salary'].mean()")
    r = _r("df |> group_by(sex) |> summarise(m = mean(salary))")
    assert r.ok and r.kind == "table"
    assert _as_dict(r.payload) == _as_dict(p.payload)


def test_magrittr_pipe_also_works():
    r = _r("df %>% group_by(sex) %>% summarise(m = mean(salary))")
    assert r.ok and set(_as_dict(r.payload)) == {"F", "M"}


@pytest.mark.parametrize("rfn,pyfn", [
    ("mean", "mean"), ("sum", "sum"), ("median", "median"),
    ("sd", "std"), ("var", "var"),
])
def test_summarise_aggregations_match_pandas(rfn, pyfn):
    p = _pandas(f"df.groupby('sex')['salary'].{pyfn}()")
    r = _r(f"df |> group_by(sex) |> summarise(m = {rfn}(salary))")
    assert r.ok and _as_dict(r.payload) == _as_dict(p.payload)


def test_filter_then_group_summarise_matches_pandas():
    p = _pandas("df[df['salary'] >= 40000].groupby('sex')['salary'].mean()")
    r = _r("df |> filter(salary >= 40000) |> group_by(sex) |> summarise(m = mean(salary))")
    assert r.ok and _as_dict(r.payload) == _as_dict(p.payload)


def test_count_matches_pandas_and_suppresses():
    p = _pandas("df['region'].value_counts()")
    r = _r("df |> count(region)")
    assert r.ok and _as_dict(r.payload) == _as_dict(p.payload)
    assert _as_dict(r.payload)["Z"] is None            # region Z (n=2) suppressed


def test_grouped_small_group_suppressed():
    r = _r("df |> group_by(region) |> summarise(m = mean(salary))")
    assert r.ok and _as_dict(r.payload)["Z"] is None


# ---- red team: disclosive / unknown / code-execution R must be refused -------

@pytest.mark.parametrize("code", [
    "df |> group_by(sex) |> summarise(m = max(salary))",     # extreme -> refused
    "df |> group_by(sex) |> summarise(m = min(salary))",
    "df |> group_by(sex) |> summarise(m = quantile(salary))",
    "df |> head()",                                          # unknown/disclosive verb
    "df |> slice(1)",
    "df |> pull(salary)",                                    # column extraction
    "df |> arrange(salary)",                                 # no releasable summary
    "df |> group_by(nope) |> summarise(m = mean(salary))",   # unknown column
    "df |> summarise(m = mean(salary))",                     # no group_by
    "df |> group_by(sex)",                                   # no terminal summary
    "system('ls')",                                          # code execution attempt
    "df$salary",                                             # raw column access
    "nope |> count(region)",                                 # unknown source
])
def test_disclosive_or_unknown_r_refused(code):
    assert _r(code).ok is False
