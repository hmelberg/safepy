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


# ---- expression parser: mutate + compound filter ----------------------------

def test_mutate_arithmetic_then_summarise_matches_pandas():
    p = _pandas("df.assign(k=df['salary'] / 1000).groupby('sex')['k'].mean()")
    r = _r("df |> mutate(k = salary / 1000) |> group_by(sex) |> summarise(m = mean(k))")
    assert r.ok and _as_dict(r.payload) == _as_dict(p.payload)


def test_mutate_log_uses_np_and_matches_pandas():
    p = _pandas("df.assign(l=np.log(df['salary'])).groupby('sex')['l'].mean()")
    r = _r("df |> mutate(l = log(salary)) |> group_by(sex) |> summarise(m = mean(l))")
    assert r.ok and _as_dict(r.payload) == _as_dict(p.payload)


def test_compound_filter_matches_pandas():
    p = _pandas("df[(df['salary'] >= 40000) & (df['sex'] == 'F')]['region'].value_counts()")
    r = _r("df |> filter(salary >= 40000 & sex == 'F') |> count(region)")
    assert r.ok and _as_dict(r.payload) == _as_dict(p.payload)


def test_filter_in_operator_matches_pandas():
    p = _pandas("df[df['region'].isin(['A', 'B'])]['region'].value_counts()")
    r = _r("df |> filter(region %in% c('A', 'B')) |> count(region)")
    assert r.ok and _as_dict(r.payload) == _as_dict(p.payload)


def test_mutate_ifelse_bucket():
    r = _r("df |> mutate(band = ifelse(salary >= 50000, 'hi', 'lo')) |> count(band)")
    assert r.ok and set(_as_dict(r.payload)) == {"hi", "lo"}


# ---- base-R modelling reaches the shared facade verbs (the "extras") ---------

def test_lm_matches_pandas_ols():
    p = _pandas("df.ols(y='salary', x=['pid'])")
    r = _r("lm(salary ~ pid, data = df)")
    assert r.ok and r.payload == p.payload


def test_lm_intercept_only_term_ignored():
    r = _r("lm(salary ~ pid + 1, data = df)")
    assert r.ok and r.kind == "regression"


def test_glm_non_family_refused():
    assert _r("glm(salary ~ pid, family = gamma, data = df)").ok is False


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
    # expression-level escapes: unknown/disclosive functions must be refused
    "df |> mutate(x = max(salary)) |> group_by(sex) |> summarise(m = mean(x))",
    "df |> mutate(x = system('ls')) |> count(sex)",
    "df |> mutate(x = quantile(salary)) |> count(sex)",
    "df |> filter(sort(salary) > 0) |> count(sex)",
    "df |> mutate(x = nope(salary)) |> count(sex)",          # unknown function
    "df |> mutate(x = salary[1]) |> count(sex)",             # positional -> parse error
])
def test_disclosive_or_unknown_r_refused(code):
    assert _r(code).ok is False
