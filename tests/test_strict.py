"""STRICT profile: the SafeFrame capability facade.

Run with profile='strict' so the source is wrapped in a SafeFrame and pandas is
not in scope. These tests assert both that legitimate analysis works through the
facade and that the disclosive capabilities are simply absent.
"""

import pytest

from safepython import run
from safepython.policy import Profile
from tests.fixtures import salaries

DF = salaries()


def _strict(code):
    return run(code, {"df": DF}, profile=Profile.STRICT)


# ---- the facade supports legitimate analysis --------------------------------

def test_groupby_mean_chains():
    r = _strict("df.groupby('sex').mean('salary')")
    assert r.ok and r.kind == "table"
    assert r.audit["profile"] == "strict"


def test_value_counts_suppresses_small_cell():
    r = _strict("df.value_counts('region')")
    assert r.ok
    values = dict(zip(r.payload["index"], r.payload["values"]))
    assert values["Z"] is None          # region Z (n=2) < min_n


def test_where_then_aggregate():
    r = _strict("df.where('salary', '>=', 0).value_counts('sex')")
    assert r.ok


def test_assign_derives_column_then_groups():
    r = _strict("df.assign('k', 'salary / 1000').groupby('sex').mean('k')")
    assert r.ok


# ---- disclosive capabilities don't exist in the facade ----------------------

@pytest.mark.parametrize("code", [
    "df.head()",                 # method not on SafeFrame
    "df.values",                 # ditto
    "df['salary']",              # no __getitem__
    "df.iloc[0]",                # not on SafeFrame
    "df.groupby('sex').max('salary')",   # SafeGroupBy has no max
    "df.assign('x', 'salary.bit_length')",   # expr compiler rejects attribute access
    "df.where('region', '==', 'Z')",     # returns a SafeFrame -> not releasable as final result
])
def test_disclosive_paths_refused(code):
    r = _strict(code)
    assert r.ok is False


def test_pandas_not_in_scope():
    # under STRICT there is no `pd` / `np` handle at all
    assert _strict("pd.DataFrame()").ok is False
    assert _strict("np.array([1,2])").ok is False
