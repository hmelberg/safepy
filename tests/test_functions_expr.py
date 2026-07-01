"""Phase 3/4: the assign() string-expression compiler (microdata-style) matching
the SafeColumn surface, plus date arithmetic (timedelta -> .dt.days)."""

import numpy as np
import pandas as pd

from safepy import run
from safepy.policy import Profile


def _df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    start = pd.date_range("2015-01-01", periods=n, freq="5D")
    return pd.DataFrame({
        "muni": rng.choice(["0301 Oslo", "1103 Stavanger", "4601 Bergen"], n),
        "sex": rng.choice(["F", "M"], n),
        "salary": rng.integers(30000, 90000, n).astype(float),
        "start": start.astype(str),
        "end": (start + pd.to_timedelta(rng.integers(30, 400, n), unit="D")).astype(str),
    })


DF = _df()


def _strict(code):
    return run(code, {"df": DF}, profile=Profile.STRICT)


# ---- expanded assign() expression compiler ----------------------------------

def test_expr_math_funcs():
    r = _strict("df.assign('logsal', 'log(salary)').groupby('sex')['logsal'].mean()")
    assert r.ok


def test_expr_binary_and_round():
    assert _strict("df.assign('c', 'minimum(salary, 50000)').groupby('sex')['c'].mean()").ok
    assert _strict("df.assign('r', 'round(salary, -3)').value_counts('r')").ok


def test_expr_where_with_comparison():
    r = _strict("df.assign('hi', 'where(salary >= 60000, 1, 0)').groupby('sex')['hi'].mean()")
    assert r.ok


def test_expr_boolean_combination():
    r = _strict("df.assign('flag', 'where((salary > 40000) & (sex == \"M\"), 1, 0)')"
                ".groupby('sex')['flag'].mean()")
    assert r.ok


def test_expr_string_substr():
    r = _strict("df.assign('county', 'substr(muni, 0, 2)').value_counts('county')")
    assert r.ok
    assert set(r.payload["index"]) == {"03", "11", "46"}


def test_expr_string_upper_and_concat():
    assert _strict("df.assign('u', 'upper(sex)').value_counts('u')").ok
    assert _strict("df.assign('c', 'concat(sex, muni)').value_counts('c')").ok


def test_expr_unknown_func_refused():
    r = _strict("df.assign('x', 'wombat(salary)').groupby('sex')['x'].mean()")
    assert r.ok is False and "not available" in r.error["message"]


def test_expr_attribute_access_refused():
    r = _strict("df.assign('x', 'salary.values').groupby('sex')['x'].mean()")
    assert r.ok is False


# ---- date arithmetic (pandas path) ------------------------------------------

def test_date_difference_in_days():
    code = ("df.assign(dur=(pd.to_datetime(df['end']) - pd.to_datetime(df['start'])).dt.days)"
            ".groupby('sex')['dur'].mean()")
    r = _strict(code)
    assert r.ok and r.kind == "table"
