"""Second batch of pandas-shaped verbs (compute-private principle):
column transforms (fillna/where/mask/ffill/bfill/interpolate/cumprod),
shape reducers (sem/skew/kurt), and frame-level reducers (mean/sum/median/
std/var/count/nunique, cov, describe). Transforms return private objects;
reducers are release paths and are suppressed on their own contributing count."""

import numpy as np
import pandas as pd

from safepy import run
from safepy.policy import Profile


def _df(n=300, seed=1):
    rng = np.random.default_rng(seed)
    salary = rng.integers(30000, 90000, n).astype(float)
    salary[rng.integers(0, n, 15)] = np.nan          # some missing
    return pd.DataFrame({
        "unit": rng.choice(["a", "b", "c"], n),
        "year": rng.integers(2010, 2016, n),
        "sex": rng.choice(["M", "F"], n),
        "salary": salary,
        "bonus": rng.integers(0, 10000, n).astype(float),
    })


DF = _df()


def _strict(code):
    return run(code, {"df": DF}, profile=Profile.STRICT)


# ---- column transforms (private) --------------------------------------------

def test_column_fillna():
    r = _strict("df.assign(s=df['salary'].fillna(0)).groupby('sex')['s'].mean()")
    assert r.ok


def test_column_where_and_mask():
    assert _strict("df.assign(c=df['salary'].where(df['salary'] > 50000, 0))"
                   ".groupby('sex')['c'].mean()").ok
    assert _strict("df.assign(c=df['salary'].mask(df['salary'] > 50000, 0))"
                   ".groupby('sex')['c'].mean()").ok


def test_where_needs_column_condition():
    r = _strict("df['salary'].where(True, 0)")
    assert r.ok is False and "boolean column" in r.error["message"]


def test_ffill_bfill_interpolate():
    assert _strict("df.sort_values('year').assign(f=df.sort_values('year')['salary'].ffill())"
                   ".groupby('sex')['f'].mean()").ok
    assert _strict("df.assign(b=df['salary'].bfill()).groupby('sex')['b'].mean()").ok
    assert _strict("df.assign(i=df['salary'].interpolate()).groupby('sex')['i'].mean()").ok


def test_interpolate_rejects_callable():
    r = _strict("df['salary'].interpolate(len)")
    assert r.ok is False


def test_cumprod():
    r = _strict("df.assign(c=(df['salary'] / 90000).cumprod()).groupby('sex')['c'].mean()")
    assert r.ok


# ---- shape / precision reducers (release path, suppressed) ------------------

def test_sem_skew_kurt():
    for stat in ("sem", "skew", "kurt"):
        r = _strict(f"df['salary'].{stat}()")
        assert r.ok and r.kind == "scalar"
        assert r.payload["value"] is not None      # 300 rows >> min_n


def test_skew_not_coarsened_by_round_to():
    # STRICT round_to=10 would crush a dimensionless skew to 0; it must not apply
    r = _strict("df['salary'].skew()")
    assert r.ok and abs(r.payload["value"]) < 5 and r.payload["value"] != 0.0


def test_reducer_suppressed_when_few_rows():
    r = _strict("df[df['unit'] == 'zzz']['salary'].skew()")
    assert r.ok and r.payload["value"] is None


# ---- frame-level reducers ----------------------------------------------------

def test_frame_mean_series():
    r = _strict("df.mean()")
    assert r.ok and r.kind == "table"
    # numeric columns only (unit/sex dropped)
    assert set(r.payload["index"]) == {"year", "salary", "bonus"}
    assert all(v is not None for v in r.payload["values"])


def test_frame_sum_median_std_var():
    for stat in ("sum", "median", "std", "var"):
        assert _strict(f"df.{stat}()").ok


def test_frame_count_includes_all_columns():
    r = _strict("df.count()")
    assert r.ok
    assert set(r.payload["index"]) == {"unit", "year", "sex", "salary", "bonus"}


def test_frame_nunique_not_rounded():
    r = _strict("df.nunique()")
    assert r.ok
    vals = dict(zip(r.payload["index"], r.payload["values"]))
    assert vals["unit"] == 3 and vals["sex"] == 2      # exact, not rounded to 10/0


def test_frame_mean_suppresses_small_column():
    # a column that is nearly all-NaN should suppress while others release
    code = ("df.assign(rare=df['salary'].where(df['unit'] == 'a').where(df['year'] == 2099))"
            ".mean()")
    r = _strict(code)
    assert r.ok
    vals = dict(zip(r.payload["index"], r.payload["values"]))
    assert vals["rare"] is None and vals["salary"] is not None


# ---- cov / describe ----------------------------------------------------------

def test_frame_cov():
    r = _strict("df.cov()")
    assert r.ok and r.kind == "table"


def test_frame_describe_table():
    r = _strict("df.describe()")
    assert r.ok and r.kind == "table"
    assert set(r.payload["index"]) == {"count", "mean", "std", "min", "25%", "50%", "75%", "max"}


# ---- frame transforms (private) ---------------------------------------------

def test_frame_astype_round_clip():
    assert _strict("df.dropna().astype({'salary': 'int'}).groupby('sex')['salary'].mean()").ok
    assert _strict("df.round(0).groupby('sex')['salary'].mean()").ok
    assert _strict("df.clip(lower=0, upper=80000).groupby('sex')['salary'].mean()").ok


def test_frame_select_dtypes_and_filter():
    assert _strict("df.select_dtypes('number').mean()").ok
    r = _strict("df.filter(like='sal').mean()")
    assert r.ok and r.payload["index"] == ["salary"]
