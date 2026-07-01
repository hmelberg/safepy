"""Phase 1: traditional pandas chaining shapes in STRICT mode.

Asserts (a) familiar pandas idioms run and release suppressed aggregates, and
(b) the SafeColumn intermediate never reveals values — the linchpin invariant.
"""

import pytest

from safepy import run
from safepy.policy import Profile
from tests.fixtures import salaries

DF = salaries()  # columns: pid, name, sex, region, salary; region 'Z' has n=2


def _strict(code):
    return run(code, {"df": DF}, profile=Profile.STRICT)


# ---- traditional pandas shapes that should WORK -----------------------------

def test_whole_column_mean_returns_scalar():
    r = _strict("df['salary'].mean()")
    assert r.ok and r.kind == "scalar"
    assert r.payload["stat"] == "mean" and r.payload["value"] is not None
    assert r.payload["n"] == 50


def test_traditional_groupby_chain():
    # the exact shape the user asked for
    r = _strict("df.groupby('sex')['salary'].mean()")
    assert r.ok and r.kind == "table"
    assert r.payload["name"] == "mean(salary)"
    assert set(r.payload["index"]) == {"F", "M"}


def test_boolean_mask_filter_then_aggregate():
    r = _strict("df[df['salary'] >= 0]['region'].value_counts()")
    assert r.ok
    values = dict(zip(r.payload["index"], r.payload["values"]))
    assert values["Z"] is None          # region Z (n=2) < min_n -> suppressed


def test_column_value_counts():
    r = _strict("df['region'].value_counts()")
    assert r.ok
    assert dict(zip(r.payload["index"], r.payload["values"]))["Z"] is None


def test_assign_then_traditional_groupby_chain():
    r = _strict("df.assign('k', 'salary / 1000').groupby('sex')['k'].mean()")
    assert r.ok and r.kind == "table"


def test_filtered_small_group_scalar_is_suppressed():
    # filter to region Z (2 people), then mean -> scalar must be suppressed
    r = _strict("df[df['region'] == 'Z']['salary'].mean()")
    assert r.ok and r.kind == "scalar"
    assert r.payload["value"] is None and r.payload["n"] is None


def test_derived_column_arithmetic():
    r = _strict("df[(df['salary'] > 0) & (df['sex'] == 'F')]['salary'].mean()")
    assert r.ok and r.kind == "scalar"


# ---- the SafeColumn must never reveal values (red team) ----------------------

@pytest.mark.parametrize("code", [
    "df['salary']",                 # dangling column -> refused by mediator
    "df['salary'].values",          # gate: denied attribute
    "df['salary'].to_numpy()",      # gate: denied
    "df['salary'].tolist()",        # gate: denied
    "df['salary'].iloc[0]",         # gate: denied
    "df['salary'][0]",              # gate: positional indexing
    "df['salary'].head()",          # gate: denied
    "df['salary'].tolist()",        # gate: denied
    "len(df['salary'])",            # SafeColumn.__len__ refuses
    "[x for x in df['salary']]",    # gate: comprehension not allowed
])
def test_column_never_reveals_values(code):
    assert _strict(code).ok is False


@pytest.mark.parametrize("code", [
    "df[df['salary'] > 0]",                 # dangling SafeFrame
    "df.groupby('sex')['salary']",          # dangling grouped column
    "df['region']",                         # dangling SafeColumn
])
def test_dangling_intermediates_refused(code):
    r = _strict(code)
    assert r.ok is False and "intermediate" in r.error["message"]
