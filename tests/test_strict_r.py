"""The R dialect (STRICT) — first slice.

R is a *different language*, so it does not ride the Python AST gate. Instead a
restricted dplyr/base-R surface is **parsed and translated** (never executed) to
the same backend-neutral release core (SafeVerbs) the pandas/polars dialects use.
The load-bearing property is equivalence: an R pipeline produces the same
suppressed ``Released`` as the pandas equivalent.
"""

import pandas as pd
import pytest

from safepy import run
from safepy.policy import Profile
from tests.fixtures import salaries

PDF = salaries()          # pid, name, sex, region, salary; region 'Z' has n=2
REG = pd.DataFrame({"region": ["A", "B", "Z"], "budget": [100, 200, 300]})


def _pandas2(code):
    return run(code, {"df": PDF, "reg": REG}, profile=Profile.STRICT)


def _r2(code):
    return run(code, {"df": PDF, "reg": REG}, profile=Profile.STRICT, dialect="r")


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


# ---- tidyverse verb breadth: select / rename / arrange / distinct / multi ----

def test_select_then_summarise_matches_pandas():
    p = _pandas("df[['sex', 'salary']].groupby('sex')['salary'].mean()")
    r = _r("df |> select(sex, salary) |> group_by(sex) |> summarise(m = mean(salary))")
    assert r.ok and _as_dict(r.payload) == _as_dict(p.payload)


def test_select_negative_drops_column():
    # dropping 'name' still lets us aggregate salary
    r = _r("df |> select(-name) |> group_by(sex) |> summarise(m = mean(salary))")
    assert r.ok and set(_as_dict(r.payload)) == {"F", "M"}


def test_rename_then_summarise_matches_pandas():
    p = _pandas("df.rename(columns={'salary': 'pay'}).groupby('sex')['pay'].mean()")
    r = _r("df |> rename(pay = salary) |> group_by(sex) |> summarise(m = mean(pay))")
    assert r.ok and _as_dict(r.payload) == _as_dict(p.payload)


def test_arrange_desc_is_shaping_only():
    p = _pandas("df.groupby('sex')['salary'].mean()")
    r = _r("df |> arrange(desc(salary)) |> group_by(sex) |> summarise(m = mean(salary))")
    assert r.ok and _as_dict(r.payload) == _as_dict(p.payload)


def test_distinct_then_summarise():
    r = _r("df |> distinct() |> group_by(sex) |> summarise(m = mean(salary))")
    assert r.ok and set(_as_dict(r.payload)) == {"F", "M"}


def test_multi_stat_summarise():
    r = _r("df |> group_by(sex) |> summarise(m = mean(salary), s = sd(salary))")
    assert r.ok and r.payload["type"] == "frame"
    assert set(r.payload["columns"]) == {"m", "s"}
    assert set(r.payload["index"]) == {"F", "M"}


# ---- multi-statement scripts -------------------------------------------------

def test_multi_statement_intermediate_frame():
    p = _pandas("df[df['salary'] >= 40000].groupby('sex')['salary'].mean()")
    r = _r("x <- df |> filter(salary >= 40000)\n"
           "x |> group_by(sex) |> summarise(m = mean(salary))")
    assert r.ok and _as_dict(r.payload) == _as_dict(p.payload)


def test_multi_statement_semicolons():
    r = _r("x <- df |> mutate(k = salary / 1000); "
           "x |> group_by(sex) |> summarise(m = mean(k))")
    assert r.ok and set(_as_dict(r.payload)) == {"F", "M"}


def test_multiline_pipeline_trailing_pipe():
    r = _r("df |>\n  group_by(sex) |>\n  summarise(m = mean(salary))")
    assert r.ok and set(_as_dict(r.payload)) == {"F", "M"}


def test_comments_are_ignored():
    r = _r("# compute mean salary by sex\n"
           "df |> group_by(sex) |> summarise(m = mean(salary))  # done")
    assert r.ok and set(_as_dict(r.payload)) == {"F", "M"}


def test_intermediate_feeds_base_r_data():
    r = _r("sub <- df |> filter(region %in% c('A', 'B'))\n"
           "aggregate(salary ~ sex, data = sub, FUN = mean)")
    assert r.ok and set(_as_dict(r.payload)) == {"F", "M"}


def test_dangling_final_frame_refused():
    # last statement is a bare frame -> not releasable
    assert _r("x <- df |> filter(salary >= 40000)\nx").ok is False


@pytest.mark.parametrize("code", [
    "x <- df |> group_by(sex) |> summarise(m = mean(salary))\nx |> count(sex)",  # pipe from a result
    "y |> group_by(sex) |> summarise(m = mean(salary))",                          # unknown name
    "aggregate(salary ~ sex, data = nope, FUN = mean)",                           # unknown data name
])
def test_multi_statement_bad_refs_refused(code):
    assert _r(code).ok is False


# ---- joins -------------------------------------------------------------------

def test_left_join_then_summarise_matches_pandas():
    p = _pandas2("df.merge(reg, on='region', how='left').groupby('sex')['budget'].mean()")
    r = _r2("df |> left_join(reg, by = 'region') |> group_by(sex) |> summarise(m = mean(budget))")
    assert r.ok and _as_dict(r.payload) == _as_dict(p.payload)


def test_inner_join_runs_and_suppresses():
    r = _r2("df |> inner_join(reg, by = 'region') |> group_by(region) "
            "|> summarise(m = mean(budget))")
    assert r.ok and _as_dict(r.payload)["Z"] is None      # region Z (n=2) suppressed


@pytest.mark.parametrize("code", [
    "df |> left_join(nope, by = 'region') |> count(sex)",     # unknown frame
    "df |> left_join(reg, by = 'nope') |> count(sex)",        # unknown join key
])
def test_join_bad_args_refused(code):
    assert _r2(code).ok is False


# ---- case_when ---------------------------------------------------------------

def test_case_when_bucket():
    r = _r("df |> mutate(band = case_when(salary >= 50000 ~ 'hi', TRUE ~ 'lo')) "
           "|> count(band)")
    assert r.ok and set(_as_dict(r.payload)) == {"hi", "lo"}


def test_case_when_three_way_first_match_wins():
    r = _r("df |> mutate(b = case_when(salary >= 60000 ~ 'hi', "
           "salary >= 40000 ~ 'mid', TRUE ~ 'lo')) |> count(b)")
    assert r.ok and set(_as_dict(r.payload)) <= {"hi", "mid", "lo"}


# ---- base R: aggregate / table / mean(df$x) / assignment ---------------------

def test_aggregate_matches_pandas():
    p = _pandas("df.groupby('sex')['salary'].mean()")
    r = _r("aggregate(salary ~ sex, data = df, FUN = mean)")
    assert r.ok and _as_dict(r.payload) == _as_dict(p.payload)


def test_table_one_col_matches_value_counts():
    p = _pandas("df['region'].value_counts()")
    r = _r("table(df$region)")
    assert r.ok and _as_dict(r.payload) == _as_dict(p.payload)
    assert _as_dict(r.payload)["Z"] is None


def test_table_two_cols_is_crosstab():
    r = _r("table(df$sex, df$region)")
    assert r.ok and r.payload["type"] == "frame"


def test_base_mean_of_column_matches_pandas():
    p = _pandas("df['salary'].mean()")
    r = _r("mean(df$salary)")
    assert r.ok and r.kind == "scalar" and r.payload["value"] == p.payload["value"]


def test_leading_assignment_is_stripped():
    r = _r("result <- df |> group_by(sex) |> summarise(m = mean(salary))")
    assert r.ok and set(_as_dict(r.payload)) == {"F", "M"}


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
    "df |> select(sex, salary)",                             # dangling frame, no summary
    "df |> arrange(salary)",                                 # dangling frame, no summary
    "df |> select(nope) |> count(nope)",                     # unknown column
    "df |> rename(x = nope) |> count(sex)",                  # unknown column
    "aggregate(salary ~ sex, data = df, FUN = max)",         # disclosive FUN
    "summary(df)",                                           # unsupported base-R fn
    "df[df$salary > 40000, ]",                               # base-R row subset (dangling)
    "df |> mutate(x = case_when(salary >= 5 ~ max(salary), TRUE ~ 0)) |> count(sex)",
])
def test_disclosive_or_unknown_r_refused(code):
    assert _r(code).ok is False
