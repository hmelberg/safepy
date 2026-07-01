"""Multiple datasets: merge + summarise (derived frames) and the schema catalog
returned for every SafeFrame left in the session (names/dtypes/missing, no data)."""

import numpy as np
import pandas as pd

from safepy import run
from safepy.policy import Profile


def _df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "id": np.arange(n),
        "sex": rng.choice(["F", "M"], n),
        "region": np.where(np.arange(n) < 2, "Z", rng.choice(["A", "B"], n)),  # Z tiny
        "salary": rng.integers(30000, 90000, n).astype(float),
    })


DF = _df()
OTHER = pd.DataFrame({"id": np.arange(200), "dept": np.resize(["x", "y"], 200)})


def _strict(code, sources=None):
    return run(code, sources or {"df": DF}, profile=Profile.STRICT)


# ---- merge + summarise ------------------------------------------------------

def test_merge_then_aggregate():
    r = _strict("df.merge(other, on='id').groupby('dept')['salary'].mean()",
                {"df": DF, "other": OTHER})
    assert r.ok and r.kind == "table"


def test_merge_unknown_key_refused():
    r = _strict("df.merge(other, on='nope').groupby('dept')['salary'].mean()",
                {"df": DF, "other": OTHER})
    assert r.ok is False and "join key" in r.error["message"]


def test_summarise_builds_a_dataset_refused_raw_but_catalogued():
    code = "s = df.summarise('sex', pay=('salary','mean'), n=('salary','count'))\ns"
    r = _strict(code)
    assert r.ok is False and "intermediate" in r.error["message"]   # dangling frame
    scat = next(d for d in r.catalog if d["name"] == "s")           # ...but catalogued
    assert {"sex", "pay", "n"} <= {c["name"] for c in scat["columns"]}


def test_summarise_bad_func_refused():
    r = _strict("df.summarise('sex', m=('salary','max')).groupby('sex')['m'].mean()")
    assert r.ok is False and "not allowed" in r.error["message"]


# ---- the catalog ------------------------------------------------------------

def test_catalog_lists_sources_and_derived():
    code = "new_df = df.summarise('sex', pay=('salary','mean'))\ndf['salary'].mean()"
    r = _strict(code)
    assert r.ok and r.kind == "scalar"
    names = {d["name"] for d in r.catalog}
    assert {"df", "new_df"} <= names


def test_catalog_has_schema_not_values():
    r = _strict("df['salary'].mean()")
    dfcat = next(d for d in r.catalog if d["name"] == "df")
    assert dfcat["n_rows"] == 200 and dfcat["n_columns"] == 4
    sal = next(c for c in dfcat["columns"] if c["name"] == "salary")
    assert sal["dtype"].startswith("float") and sal["n_missing"] == 0
    # only schema keys — no data/values/min/max/unique
    for c in dfcat["columns"]:
        assert set(c) == {"name", "dtype", "n_missing"}


def test_catalog_suppresses_small_derived_frame():
    code = "tiny = df.where('region', '==', 'Z')\ndf['salary'].mean()"
    r = _strict(code)
    tiny = next(d for d in r.catalog if d["name"] == "tiny")
    assert tiny["n_rows"] is None            # 2 rows < min_n -> suppressed


def test_catalog_present_even_on_refused_result():
    r = _strict("df")   # final expr is a SafeFrame -> refused, catalog still returned
    assert r.ok is False
    assert any(d["name"] == "df" for d in r.catalog)


def test_missing_counts_suppressed_when_small():
    d = DF.copy()
    d.loc[d.index[:3], "salary"] = np.nan     # 3 missing (< min_n)
    r = run("df['salary'].count()", {"df": d}, profile=Profile.STRICT)
    sal = next(c for c in r.catalog[0]["columns"] if c["name"] == "salary")
    assert sal["n_missing"] is None           # small nonzero missing count suppressed
