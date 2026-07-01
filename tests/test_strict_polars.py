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


@pytest.mark.parametrize("stat", ["mean", "sum", "std", "var", "median", "count"])
def test_group_agg_matches_pandas_across_stats(stat):
    pandas = _pandas(f"df.groupby('sex')['salary'].{stat}()")
    polars = _polars("import polars as pl\n"
                     f"df.group_by('sex').agg(pl.col('salary').{stat}())")
    assert polars.ok
    assert _as_dict(polars.payload) == _as_dict(pandas.payload)


def test_group_agg_uses_native_polars_backend():
    # M2: the reduction is computed in polars, not via whole-frame to_pandas
    r = _polars("import polars as pl\ndf.group_by('sex').agg(pl.col('salary').mean())")
    assert r.ok and r.audit.get("backend") == "polars"


def test_group_by_len_uses_native_polars_backend():
    r = _polars("df.group_by('region').len()")
    assert r.ok and r.audit.get("backend") == "polars"


def test_group_agg_count_matches_pandas_microdata_tier():
    # microdata tier turns on count-noise (deterministic, label-keyed) + rounding;
    # native polars must produce the identical noised/suppressed counts as pandas.
    p = run("df.groupby('region')['salary'].count()", {"df": PDF},
            profile=Profile.STRICT, suppression="microdata")
    q = run("import polars as pl\ndf.group_by('region').agg(pl.col('salary').count())",
            {"df": PL_DF}, profile=Profile.STRICT, dialect="polars", suppression="microdata")
    assert q.ok and _as_dict(q.payload) == _as_dict(p.payload)
    assert _as_dict(q.payload)["Z"] is None          # Z (n=2) suppressed in both


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


def test_multi_select_aggregation():
    r = _polars("import polars as pl\n"
                "df.select(pl.col('salary').mean().alias('avg'), pl.col('salary').std().alias('sd'))")
    assert r.ok and r.kind == "table"
    d = dict(zip(r.payload["index"], r.payload["values"]))
    assert set(d) == {"avg", "sd"} and d["avg"] is not None


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


# ---- compound (multi-reducer) agg -------------------------------------------

def test_compound_agg_multiple_stats():
    r = _polars("import polars as pl\n"
                "df.group_by('sex').agg(pl.col('salary').mean(), pl.col('salary').std())")
    assert r.ok and r.payload["type"] == "frame"
    assert set(r.payload["columns"]) == {"mean(salary)", "std(salary)"}
    assert set(r.payload["index"]) == {"F", "M"}


def test_compound_agg_suppresses_small_group():
    r = _polars("import polars as pl\n"
                "df.group_by('region').agg(pl.col('salary').mean(), pl.col('salary').count())")
    assert r.ok and r.payload["type"] == "frame"
    zi = r.payload["index"].index("Z")
    assert all(cell is None for cell in r.payload["data"][zi])   # whole Z row suppressed


def test_compound_agg_mixed_columns_with_alias():
    r = _polars("import polars as pl\n"
                "df.group_by('sex').agg(pl.col('salary').mean().alias('avg'),"
                " pl.col('pid').count())")
    assert r.ok and r.payload["type"] == "frame"
    assert "avg" in r.payload["columns"]


# ---- lazy frames ------------------------------------------------------------

def test_lazyframe_source_group_agg_matches_eager():
    code = ("import polars as pl\n"
            "df.filter(pl.col('salary') >= 40000).group_by('sex').agg(pl.col('salary').mean())")
    eager = _polars(code)
    lazy = run(code, {"df": PL_DF.lazy()}, profile=Profile.STRICT, dialect="polars")
    assert lazy.ok and _as_dict(lazy.payload) == _as_dict(eager.payload)


def test_lazyframe_source_in_catalog():
    r = run("import polars as pl\ndf.group_by('sex').agg(pl.col('salary').mean())",
            {"df": PL_DF.lazy()}, profile=Profile.STRICT, dialect="polars")
    cat = {c["name"]: c for c in (r.catalog or [])}
    assert "df" in cat and cat["df"]["n_rows"] is not None and cat["df"]["n_columns"] == 5


def test_lazyframe_delegated_verb():
    r = run("df.corr()", {"df": PL_DF.lazy()}, profile=Profile.STRICT, dialect="polars")
    assert r.ok


# ---- schema catalog ---------------------------------------------------------

def test_catalog_includes_polars_source():
    r = _polars("import polars as pl\ndf.group_by('sex').agg(pl.col('salary').mean())")
    cat = {c["name"]: c for c in (r.catalog or [])}
    assert "df" in cat
    df_cat = cat["df"]
    assert df_cat["n_rows"] is not None            # 50 rows, above min_n
    assert df_cat["n_columns"] == 5
    cols = {c["name"]: c for c in df_cat["columns"]}
    assert {"pid", "name", "sex", "region", "salary"} == set(cols)
    assert cols["salary"]["n_missing"] == 0        # no missing -> reported as 0


# ---- delegated model / stat verbs (mirror the pandas SafeFrame surface) ------

def test_ols_matches_pandas():
    pandas = _pandas("df.ols(y='salary', x=['pid'])")
    polars = _polars("df.ols(y='salary', x=['pid'])")
    assert polars.ok and polars.payload == pandas.payload


def test_corr_matches_pandas():
    pandas = _pandas("df.corr()")
    polars = _polars("df.corr()")
    assert polars.ok and polars.payload == pandas.payload


def test_describe_matches_pandas():
    pandas = _pandas("df.describe()")
    polars = _polars("df.describe()")
    assert polars.ok and polars.payload == pandas.payload


def test_frame_mean_reducer_delegates():
    r = _polars("df.mean()")
    assert r.ok and r.payload["type"] == "series"
    assert "salary" in r.payload["index"]


def test_plot_on_polars_aggregate():
    r = _polars("import polars as pl\n"
                "df.group_by('sex').agg(pl.col('salary').mean()).plot.bar()")
    assert r.ok and r.kind == "chart"


def test_non_whitelisted_frame_method_still_refused():
    # a non-terminal / unknown method is not delegated -> refused
    assert _polars("df.assign(x=1)").ok is False


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
