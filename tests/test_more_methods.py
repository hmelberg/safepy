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


# ---- reshapes: now allowed as private-object producers (compute-private) -----

def test_melt_then_aggregate():
    # melt is a private reshape; the melted 'value' column still exits only via a
    # suppressed aggregation.
    r = _strict("df.melt(id_vars=['sex'], value_vars=['salary'])"
                ".groupby('sex')['value'].mean()")
    assert r.ok and r.kind == "table"


def test_pivot_private_then_reduce_suppressed():
    # a raw pivot places individual salaries into (unit x sex) cells -- fine while
    # private. Reducing a pivoted column has only n=3 (units) contributing rows,
    # so the reducer's own n<min_n check suppresses the value. This is the whole
    # compute-private argument in one test: the reshape is safe because the EXIT
    # is guarded, not the operation.
    r = _strict("df.drop_duplicates(subset=['unit', 'sex'])"
                ".pivot(index='unit', columns='sex', values='salary')"
                "['M'].mean()")
    assert r.ok and r.kind == "scalar"
    assert r.payload["value"] is None      # n=3 units < min_n=5 -> suppressed


def test_explode_is_allowed():
    # no list columns here, so explode is a structural no-op, but it must run
    r = _strict("df.explode('salary').groupby('sex')['salary'].mean()")
    assert r.ok


# ---- guarded callable verbs: map / agg / transform / rank --------------------

def test_column_map_dict():
    r = _strict("df.assign(s=df['sex'].map({'M': 'Male', 'F': 'Female'})).value_counts('s')")
    assert r.ok and set(r.payload["index"]) == {"Male", "Female"}


def test_map_rejects_callable_name():
    # 'str' is a whitelisted builtin, so it reaches the facade guard
    r = _strict("df['sex'].map(str)")
    assert r.ok is False and "dict" in r.error["message"]


def test_map_lambda_blocked_by_gate():
    r = _strict("df['salary'].map(lambda x: x)")   # lambda is not an allowed node
    assert r.ok is False


def test_groupby_agg_multi_stats():
    r = _strict("df.groupby('sex')['salary'].agg(['mean', 'std'])")
    assert r.ok and r.kind == "table"
    assert set(r.payload["columns"]) == {"mean", "std"}


def test_groupby_agg_single_string():
    r = _strict("df.groupby('sex')['salary'].agg('mean')")
    assert r.ok and r.kind == "table"


def test_groupby_agg_rejects_callable():
    r = _strict("df.groupby('sex')['salary'].agg(len)")   # len is a whitelisted builtin
    assert r.ok is False and "not a function" in r.error["message"]


def test_groupby_transform_broadcast():
    # within-group demeaning: value minus its group mean, then aggregate
    code = ("df.assign(dev=df['salary'] - df.groupby('sex')['salary'].transform('mean'))"
            ".groupby('sex')['dev'].mean()")
    r = _strict(code)
    assert r.ok


def test_transform_rejects_callable():
    r = _strict("df.groupby('sex')['salary'].transform(abs)")
    assert r.ok is False


def test_rank_denied_by_gate():
    # rank + filter + sum is an order-statistic differencing primitive, so rank
    # stays gate-denied until a multi-query audit layer exists.
    r = _strict("df.assign(rk=df['salary'].rank()).groupby('sex')['rk'].mean()")
    assert r.ok is False


def test_rank_differencing_attack_blocked():
    # the concrete attack the denial prevents: sum over rank<=k vs rank<=k-1
    r = _strict("df.assign(rk=df['salary'].rank())[df['rk'] <= 6]['salary'].sum()")
    assert r.ok is False
