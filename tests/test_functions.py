"""Element-wise functions: .str (incl. substr), expanded numpy, expanded .dt.
These transform column -> column (private), so they are safe by construction;
disclosure control still happens at aggregation/release."""

import numpy as np
import pandas as pd

from safepy import run
from safepy.policy import Profile


def _df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "grp": rng.choice(["Apple", "Banana"], n),
        "sex": rng.choice(["F", "M"], n),
        "salary": rng.integers(30000, 90000, n).astype(float),
        "hire": pd.date_range("2015-01-01", periods=n, freq="7D").astype(str),
    })


DF = _df()


def _strict(code):
    return run(code, {"df": DF}, profile=Profile.STRICT)


# ---- .str (string functions) ------------------------------------------------

def test_str_substr_then_value_counts():
    r = _strict("df.assign(initial=df['grp'].str.substr(0, 1)).value_counts('initial')")
    assert r.ok
    vals = dict(zip(r.payload["index"], r.payload["values"]))
    assert set(vals) == {"A", "B"}          # first letters of Apple/Banana


def test_str_upper_and_filter():
    r = _strict("df[df['grp'].str.startswith('A')]['salary'].mean()")
    assert r.ok and r.kind == "scalar"


def test_str_len_reducer():
    r = _strict("df.assign(L=df['grp'].str.len()).groupby('grp')['L'].mean()")
    assert r.ok


def test_str_cat_two_columns():
    r = _strict("df.assign(c=df['grp'].str.cat(df['sex'], sep='-')).value_counts('c')")
    assert r.ok
    assert any("Apple-" in k for k in r.payload["index"])


def test_str_replace_and_zfill():
    r = _strict("df.assign(z=df['grp'].str.replace('a', 'X').str.upper()).value_counts('z')")
    assert r.ok


def test_str_cat_without_column_refused():
    # joining all rows into one string would disclose -> not allowed
    r = _strict("df['grp'].str.cat(sep=',')")
    assert r.ok is False


def test_str_on_numeric_refused():
    r = _strict("df['salary'].str.upper()")
    assert r.ok is False and ".str requires a text column" in r.error["message"]


def test_unknown_str_method_refused():
    r = _strict("df['grp'].str.wombat()")
    assert r.ok is False and "not available" in r.error["message"]


def test_str_get_blocked_by_gate():
    r = _strict("df['grp'].str.get(0)")   # 'get' is on the gate deny-list
    assert r.ok is False


# ---- expanded numpy ---------------------------------------------------------

def test_np_unary_expansion():
    r = _strict("df.assign(s=np.sin(df['salary'])).groupby('sex')['s'].mean()")
    assert r.ok


def test_np_round_negative_decimals():
    r = _strict("df.assign(r=np.round(df['salary'], -3)).value_counts('r')")
    assert r.ok


def test_np_binary_minimum_and_power():
    r = _strict("df.assign(m=np.minimum(df['salary'], 50000)).groupby('sex')['m'].mean()")
    assert r.ok
    r2 = _strict("df.assign(p=np.power(df['salary'] / 90000, 2)).groupby('sex')['p'].mean()")
    assert r2.ok


def test_np_unknown_still_refused():
    r = _strict("df.assign(x=np.wombat(df['salary'])).groupby('sex')['x'].mean()")
    assert r.ok is False


# ---- expanded .dt -----------------------------------------------------------

def test_dt_weekday_and_month_end():
    r = _strict("df.assign(wd=pd.to_datetime(df['hire']).dt.weekday).groupby('sex')['wd'].mean()")
    assert r.ok
    r2 = _strict("df.assign(me=pd.to_datetime(df['hire']).dt.is_month_end).groupby('sex')['me'].mean()")
    assert r2.ok
