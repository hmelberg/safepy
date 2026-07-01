"""M1-b: the polars dialect in STRICT mode.

The polars *surface* (``df.filter().group_by().agg(pl.col(...).mean())``) over the
pandas suppression *backend*. The load-bearing property is **equivalence**: a
polars script must produce the same suppressed ``Released`` as the equivalent
pandas-dialect script — the security core is reused, only the surface differs.
"""

import datetime as _dt

import polars as pl
import pytest

from safepy import run
from safepy.policy import Profile
from tests.fixtures import salaries

PDF = salaries()                 # pandas: pid, name, sex, region, salary; region 'Z' n=2
PL_DF = pl.from_pandas(PDF)      # same data, polars surface

# a dated frame for the .dt accessor: two well-populated years (30 rows each)
DT_DF = pl.DataFrame({"d": [_dt.date(2020, 1, 1)] * 30 + [_dt.date(2021, 1, 1)] * 30,
                      "x": list(range(60))})


def _pandas(code):
    return run(code, {"df": PDF}, profile=Profile.STRICT)


def _polars(code, df=PL_DF):
    return run(code, {"df": df}, profile=Profile.STRICT, dialect="polars")


def _as_dict(payload):
    return dict(zip(payload["index"], payload["values"]))


# ---- the first end-to-end slice ---------------------------------------------

def test_group_by_agg_mean_matches_pandas():
    pandas = _pandas("df.groupby('sex')['salary'].mean()")
    polars = _polars("import polars as pl\ndf.group_by('sex').agg(pl.col('salary').mean())")
    assert polars.ok and polars.kind == "table"
    assert polars.payload["name"] == "mean(salary)"
    assert _as_dict(polars.payload) == _as_dict(pandas.payload)


def test_group_by_len_counts_and_suppresses_small_group():
    # polars-idiomatic count of rows per group; region 'Z' (n=2) must suppress.
    r = _polars("df.group_by('region').len()")
    assert r.ok and r.kind == "table"
    assert _as_dict(r.payload)["Z"] is None


def test_count_agg_matches_pandas():
    pandas = _pandas("df.groupby('region')['salary'].count()")
    polars = _polars("import polars as pl\ndf.group_by('region').agg(pl.col('salary').count())")
    assert polars.ok
    assert _as_dict(polars.payload) == _as_dict(pandas.payload)


def test_filter_then_group_agg_matches_pandas():
    pandas = _pandas("df[df['salary'] >= 40000].groupby('sex')['salary'].mean()")
    polars = _polars(
        "import polars as pl\n"
        "df.filter(pl.col('salary') >= 40000).group_by('sex').agg(pl.col('salary').mean())")
    assert polars.ok
    assert _as_dict(polars.payload) == _as_dict(pandas.payload)


def test_filter_to_small_group_suppresses():
    r = _polars(
        "import polars as pl\n"
        "df.filter(pl.col('region') == 'Z').group_by('sex').agg(pl.col('salary').mean())")
    assert r.ok
    # every surviving group is drawn from region Z (n=2) -> all suppressed
    assert all(v is None for v in r.payload["values"])


# ---- select / with_columns --------------------------------------------------

def test_with_columns_derived_then_agg_matches_pandas():
    pandas = _pandas("df.assign('k', 'salary / 1000').groupby('sex')['k'].mean()")
    polars = _polars(
        "import polars as pl\n"
        "df.with_columns((pl.col('salary') / 1000).alias('k'))"
        ".group_by('sex').agg(pl.col('k').mean())")
    assert polars.ok
    assert _as_dict(polars.payload) == _as_dict(pandas.payload)


def test_with_columns_named_kwarg_form():
    r = _polars(
        "import polars as pl\n"
        "df.with_columns(k=pl.col('salary') / 1000).group_by('sex').agg(pl.col('k').mean())")
    assert r.ok and r.payload["name"] == "mean(k)"


def test_select_reducer_returns_scalar_matches_pandas():
    pandas = _pandas("df['salary'].mean()")
    polars = _polars("import polars as pl\ndf.select(pl.col('salary').mean())")
    assert polars.ok and polars.kind == "scalar"
    assert polars.payload["stat"] == "mean"
    assert polars.payload["value"] == pandas.payload["value"]


def test_select_columns_then_agg_matches_pandas():
    pandas = _pandas("df[['sex', 'salary']].groupby('sex')['salary'].mean()")
    polars = _polars(
        "import polars as pl\n"
        "df.select('sex', 'salary').group_by('sex').agg(pl.col('salary').mean())")
    assert polars.ok
    assert _as_dict(polars.payload) == _as_dict(pandas.payload)


def test_select_dangling_frame_refused():
    # a plain column selection is an intermediate, not releasable
    r = _polars("df.select('sex', 'salary')")
    assert r.ok is False and "intermediate" in r.error["message"]


# ---- .str / .dt accessors and when/then/otherwise ---------------------------

def test_str_lowercase_derive_and_group():
    r = _polars("import polars as pl\n"
                "df.with_columns(pl.col('sex').str.to_lowercase().alias('s')).group_by('s').len()")
    assert r.ok and set(_as_dict(r.payload)) == {"f", "m"}


def test_str_len_reducer_matches_pandas():
    pandas = _pandas("df['name'].str.len().mean()")
    polars = _polars("import polars as pl\ndf.select(pl.col('name').str.len_chars().mean())")
    assert polars.ok and polars.kind == "scalar"
    assert polars.payload["value"] == pandas.payload["value"]


def test_str_derived_reducer_grouped():
    # inline reducer over a derived (str) expression, grouped
    r = _polars("import polars as pl\n"
                "df.group_by('sex').agg(pl.col('name').str.len_chars().mean())")
    assert r.ok and r.kind == "table"
    assert set(r.payload["index"]) == {"F", "M"}


def test_dt_year_derive_and_group():
    r = _polars("import polars as pl\n"
                "df.with_columns(pl.col('d').dt.year().alias('y')).group_by('y').len()",
                df=DT_DF)
    assert r.ok and set(_as_dict(r.payload)) == {"2020", "2021"}  # group labels stringified


def test_when_then_otherwise_bucket():
    r = _polars("import polars as pl\n"
                "df.with_columns(pl.when(pl.col('salary') >= 50000).then(pl.lit('hi'))"
                ".otherwise(pl.lit('lo')).alias('band')).group_by('band').len()")
    assert r.ok and set(_as_dict(r.payload)) == {"hi", "lo"}


# ---- the facade is the boundary: intermediates and disclosive verbs refused --

@pytest.mark.parametrize("code", [
    "import polars as pl\ndf.filter(pl.col('salary') >= 0)",   # dangling SafePolarsFrame
    "import polars as pl\ndf.group_by('sex')",                 # dangling groupby
    "import polars as pl\npl.col('salary')",                   # dangling SafeExpr
])
def test_dangling_intermediates_refused(code):
    r = _polars(code)
    assert r.ok is False and "intermediate" in r.error["message"]


@pytest.mark.parametrize("code", [
    "df.head()",                                               # gate: denied
    "df.get_column('salary')",                                 # gate: denied
    "df.to_pandas()",                                          # gate: denied (raw export)
    "df.sort('salary')",                                       # gate: denied (value-ordered)
    "import polars as pl\ndf.select(pl.col('salary').max())",  # SafeExpr: no max()
    "import polars as pl\ndf.group_by('sex').agg(pl.col('salary').top_k(3))",  # no top_k
])
def test_disclosive_verbs_refused(code):
    assert _polars(code).ok is False
