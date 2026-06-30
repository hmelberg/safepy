"""Phase 2: attribute column access + pd/np look-alike namespaces + natural assign."""

import numpy as np
import pandas as pd
import pytest

from safepy import run
from safepy.policy import Profile


def _df():
    rng = np.random.default_rng(3)
    n = 120
    return pd.DataFrame({
        "sex": rng.choice(["F", "M"], n),
        "region": np.where(np.arange(n) < 3, "Z", rng.choice(["A", "B"], n)),
        "age": rng.integers(20, 70, n),
        "salary": rng.integers(30000, 90000, n),
        "hire": pd.date_range("2010-01-01", periods=n, freq="20D").astype(str),
    })


DF = _df()


def _strict(code):
    return run(code, {"df": DF}, profile=Profile.STRICT)


# ---- attribute column access ------------------------------------------------

def test_attribute_column_reducer():
    r = _strict("df.salary.mean()")
    assert r.ok and r.kind == "scalar"


def test_attribute_groupby_chain():
    # the optional sugar: df.groupby('sex').salary.mean()
    r = _strict("df.groupby('sex').salary.mean()")
    assert r.ok and r.kind == "table"


def test_methods_win_over_columns():
    # a real method is reached even though attribute access exists
    r = _strict("df.groupby('sex')['salary'].mean()")
    assert r.ok


def test_private_attribute_blocked():
    assert _strict("df._df").ok is False        # gate blocks _-attributes


# ---- np look-alike ----------------------------------------------------------

def test_np_log_in_assign():
    r = _strict("df.assign(logsal=np.log(df['salary'])).groupby('sex')['logsal'].mean()")
    assert r.ok and r.kind == "table"


def test_np_where_derived_indicator():
    r = _strict("df.assign(high=np.where(df['salary'] > 50000, 1, 0)).groupby('sex')['high'].mean()")
    assert r.ok


def test_np_unknown_func_blocked():
    # np.median isn't in the SafeNp whitelist -> clear "not available" message.
    # (np.array is caught even earlier, by the gate's deny-list.)
    r = _strict("np.median(df['salary'])")
    assert r.ok is False and "not available" in r.error["message"]


# ---- pd look-alike ----------------------------------------------------------

def test_pd_crosstab():
    r = _strict("pd.crosstab(df['sex'], df['region'])")
    assert r.ok and r.payload["type"] == "frame"


def test_pd_cut_then_groupby():
    r = _strict("df.assign(band=pd.cut(df['age'], [0, 30, 50, 100])).groupby('band')['salary'].mean()")
    assert r.ok and r.kind == "table"


def test_pd_to_datetime_dt_year():
    r = _strict("df.assign(yr=pd.to_datetime(df['hire']).dt.year).groupby('sex')['yr'].mean()")
    assert r.ok


def test_pd_unknown_helper_blocked():
    r = _strict("pd.read_csv('x.csv')")
    assert r.ok is False and "not available" in r.error["message"]


# ---- natural assign with SafeColumn ----------------------------------------

def test_natural_kwargs_assign():
    r = _strict("df.assign(k=df['salary'] / 1000).groupby('sex')['k'].mean()")
    assert r.ok


def test_legacy_string_assign_still_works():
    r = _strict("df.assign('k', 'salary / 1000').groupby('sex')['k'].mean()")
    assert r.ok
