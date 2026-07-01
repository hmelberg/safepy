"""More pandas-shaped verbs: replace/recode, sort_values, drop_duplicates,
per-row derived columns (shift/diff/cumsum/pct_change), nunique, and
pivot_table. The reshape/recode/order verbs return private objects (no new
release path); pivot_table and nunique are release paths and are suppressed.
The raw `pivot`/`stack`/`melt` reshapes are refused by the gate."""

import numpy as np
import pandas as pd

from safepy import run
from safepy.policy import Profile


def _df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    unit = rng.choice(["a", "b", "c"], n)
    year = rng.integers(2010, 2016, n)
    return pd.DataFrame({
        "unit": unit,
        "year": year,
        "sex": rng.choice(["M", "F"], n),
        "region": rng.choice(["North", "South"], n),
        "salary": rng.integers(30000, 90000, n).astype(float),
    })


DF = _df()


def _strict(code):
    return run(code, {"df": DF}, profile=Profile.STRICT)


# ---- replace / recode --------------------------------------------------------

def test_column_replace_mapping():
    r = _strict("df.assign(s=df['sex'].replace({'M': 'Male', 'F': 'Female'}))"
                ".value_counts('s')")
    assert r.ok
    assert set(r.payload["index"]) == {"Male", "Female"}


def test_column_replace_scalar_pair():
    r = _strict("df.assign(s=df['sex'].replace('M', 'Male')).value_counts('s')")
    assert r.ok
    assert "Male" in set(r.payload["index"])


def test_frame_replace_nested_mapping():
    r = _strict("df.replace({'region': {'North': 'N', 'South': 'S'}})"
                ".value_counts('region')")
    assert r.ok
    assert set(r.payload["index"]) == {"N", "S"}


def test_replace_rejects_callable():
    r = _strict("df['sex'].replace(str)")
    assert r.ok is False
    # 'str' is a whitelisted builtin name, so it reaches the verb and is rejected
    assert "function" in r.error["message"].lower() or "replace" in r.error["message"]


# ---- sort_values + per-row derived columns (panel/time-series) ---------------

def test_sort_then_diff_and_cumsum():
    code = ("df.sort_values(['unit', 'year'])"
            ".assign(d=df.sort_values(['unit','year'])['salary'].diff())"
            ".groupby('unit')['d'].mean()")
    r = _strict(code)
    assert r.ok


def test_shift_produces_private_column():
    r = _strict("df.assign(prev=df['salary'].shift(1)).groupby('sex')['prev'].mean()")
    assert r.ok and r.kind == "table"


def test_cumsum_and_pct_change():
    assert _strict("df.assign(c=df['salary'].cumsum()).groupby('sex')['c'].mean()").ok
    assert _strict("df.assign(p=df['salary'].pct_change()).groupby('sex')['p'].mean()").ok


def test_cumsum_on_text_refused():
    r = _strict("df['sex'].cumsum()")
    assert r.ok is False and "numeric" in r.error["message"]


# ---- drop_duplicates ---------------------------------------------------------

def test_drop_duplicates_then_count():
    r = _strict("df.drop_duplicates(subset=['unit', 'year'])['salary'].count()")
    assert r.ok and r.kind == "scalar"


# ---- nunique -----------------------------------------------------------------

def test_nunique_released():
    r = _strict("df['unit'].nunique()")
    assert r.ok and r.kind == "scalar"
    assert r.payload["value"] == 3


def test_nunique_suppressed_when_few_rows():
    r = _strict("df[df['unit'] == 'zzz']['unit'].nunique()")   # empty -> < min_n
    assert r.ok
    assert r.payload["value"] is None


# ---- pivot_table -------------------------------------------------------------

def test_pivot_table_mean():
    r = _strict("df.pivot_table(values='salary', index='region', columns='sex', aggfunc='mean')")
    assert r.ok and r.kind == "table"


def test_pivot_table_count_no_columns():
    r = _strict("df.pivot_table(values='salary', index='region', aggfunc='count')")
    assert r.ok and r.kind == "table"


def test_pivot_table_bad_aggfunc_refused():
    r = _strict("df.pivot_table(values='salary', index='region', aggfunc='max')")
    assert r.ok is False and "aggfunc" in r.error["message"]


def test_pivot_table_suppresses_small_cells():
    # each unit x year cell holds ~16 rows; raising min_n above that forces
    # suppression, proving the cell-count pairing works for a mean aggfunc.
    r = _strict("df.pivot_table(values='salary', index='unit', columns='year', "
                "aggfunc='mean', min_n=25)")
    assert r.ok
    flat = [v for row in r.payload["data"] for v in row]
    assert any(v is None for v in flat)


# ---- refusals: raw reshape blocked by the gate -------------------------------

def test_pivot_refused_by_gate():
    r = _strict("df.pivot(index='unit', columns='year', values='salary')")
    assert r.ok is False


def test_stack_refused_by_gate():
    r = _strict("df.stack()")
    assert r.ok is False


def test_melt_refused_by_gate():
    r = _strict("df.melt()")
    assert r.ok is False
