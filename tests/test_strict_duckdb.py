"""The DuckDB (SQL) dialect — STRICT.

Architecture (gated execution, unlike R's pure translation):

1. **Static gate on the parsed AST** (``json_serialize_sql`` — parse-only, the
   user SQL is never executed before validation): exactly one SELECT statement;
   the *outer* select list may contain only GROUP BY keys and whitelisted
   aggregates; every function anywhere in the tree must be on a whitelist
   (default-deny — kills min/max/quantile/string_agg even inside subqueries).
2. **Locked execution**: ``enable_external_access=false`` (+ locked config), so
   COPY/read_csv/ATTACH/INSTALL/httpfs are dead; only registered private frames
   are visible.
3. **Release through the shared core**: the outer select is rewritten (JSON
   surgery) to carry a paired count per aggregate, and each aggregate column is
   released via ``SafeVerbs._release_group_agg`` — the same audited suppressor
   as the pandas/polars dialects.

The load-bearing property is equivalence: a SQL aggregation produces the same
suppressed ``Released`` as the pandas equivalent.
"""

import pytest

from safepy import run
from safepy.policy import Profile
from tests.fixtures import salaries

PDF = salaries()          # pid, name, sex, region, salary; region 'Z' has n=2


def _pandas(code):
    return run(code, {"df": PDF}, profile=Profile.STRICT)


def _sql(code, **kw):
    return run(code, {"df": PDF}, profile=Profile.STRICT, dialect="duckdb", **kw)


def _as_dict(payload):
    return dict(zip(payload["index"], payload["values"]))


# ---- the core slice: grouped aggregation ------------------------------------

def test_group_avg_matches_pandas():
    p = _pandas("df.groupby('sex')['salary'].mean()")
    q = _sql("SELECT sex, avg(salary) FROM df GROUP BY sex")
    assert q.ok and q.kind == "table"
    assert _as_dict(q.payload) == _as_dict(p.payload)


@pytest.mark.parametrize("sqlfn,pyfn", [
    ("avg", "mean"), ("sum", "sum"), ("median", "median"),
    ("stddev", "std"), ("var_samp", "var"), ("count", "count"),
])
def test_group_aggs_match_pandas(sqlfn, pyfn):
    p = _pandas(f"df.groupby('sex')['salary'].{pyfn}()")
    q = _sql(f"SELECT sex, {sqlfn}(salary) FROM df GROUP BY sex")
    assert q.ok, q.error
    assert _as_dict(q.payload) == _as_dict(p.payload)


def test_small_group_suppressed():
    q = _sql("SELECT region, avg(salary) FROM df GROUP BY region")
    assert q.ok and _as_dict(q.payload)["Z"] is None          # region Z (n=2)


def test_where_filter_matches_pandas():
    p = _pandas("df[df['salary'] >= 40000].groupby('sex')['salary'].count()")
    q = _sql("SELECT sex, count(salary) FROM df WHERE salary >= 40000 GROUP BY sex")
    assert q.ok and _as_dict(q.payload) == _as_dict(p.payload)


def test_count_star_group():
    p = _pandas("df.groupby('region')['salary'].count()")
    q = _sql("SELECT region, count(*) FROM df GROUP BY region")
    assert q.ok
    assert _as_dict(q.payload)["Z"] is None


def test_multi_aggregate_select():
    q = _sql("SELECT sex, avg(salary) AS m, stddev(salary) AS s FROM df GROUP BY sex")
    assert q.ok and q.payload["type"] == "frame"
    assert set(q.payload["columns"]) == {"m", "s"}


def test_whole_table_aggregate_scalar():
    p = _pandas("df['salary'].mean()")
    q = _sql("SELECT avg(salary) FROM df")
    assert q.ok
    # a single aggregate with no GROUP BY releases one suppressed value
    vals = [v for v in (q.payload.get("values") or [q.payload.get("value")]) if v is not None]
    assert vals and vals[0] == p.payload["value"]


def test_microdata_tier_counts_match_pandas():
    p = run("df.groupby('region')['salary'].count()", {"df": PDF},
            profile=Profile.STRICT, suppression="microdata")
    q = run("SELECT region, count(salary) FROM df GROUP BY region", {"df": PDF},
            profile=Profile.STRICT, dialect="duckdb", suppression="microdata")
    assert q.ok and _as_dict(q.payload) == _as_dict(p.payload)


def test_subquery_shaping_then_aggregate():
    # inner shaping is a private intermediate; the outer aggregate is suppressed
    q = _sql("SELECT sex, avg(k) FROM "
             "(SELECT sex, salary / 1000 AS k FROM df WHERE salary > 0) t "
             "GROUP BY sex")
    assert q.ok and set(_as_dict(q.payload)) == {"F", "M"}


def test_cte_shaping_then_aggregate():
    q = _sql("WITH t AS (SELECT sex, salary FROM df WHERE salary >= 40000) "
             "SELECT sex, avg(salary) FROM t GROUP BY sex")
    assert q.ok and set(_as_dict(q.payload)) == {"F", "M"}


def test_isolating_subquery_is_suppressed_not_leaked():
    # classic attack: shape down to 1 row, then "aggregate" — the paired count
    # is 1, so the cell is suppressed.
    q = _sql("SELECT sex, avg(salary) FROM "
             "(SELECT sex, salary FROM df ORDER BY salary DESC LIMIT 1) t "
             "GROUP BY sex")
    assert q.ok is False or all(v is None for v in q.payload["values"])


# ---- v2 polish ---------------------------------------------------------------

def test_group_key_missing_from_select_is_auto_added():
    # legal SQL: the key appears only in GROUP BY; we auto-include it in the output
    q = _sql("SELECT avg(salary) FROM df GROUP BY sex")
    assert q.ok and set(_as_dict(q.payload)) == {"F", "M"}


def test_count_distinct_matches_pandas_nunique():
    q = _sql("SELECT sex, count(DISTINCT region) FROM df GROUP BY sex",
             suppression="light")               # light: no value rounding
    p = PDF.groupby("sex")["region"].nunique()
    assert q.ok
    assert _as_dict(q.payload) == {k: int(v) for k, v in p.items()}


def test_group_by_expression_matches_pandas():
    p = _pandas("df.assign(r=df['region'].str.slice(0, 1))"
                ".groupby('r')['salary'].mean()")
    q = _sql("SELECT substr(region, 1, 1) AS r, avg(salary) FROM df "
             "GROUP BY substr(region, 1, 1)")
    assert q.ok
    assert _as_dict(q.payload) == _as_dict(p.payload)


def test_order_by_group_key_allowed():
    q = _sql("SELECT sex, avg(salary) FROM df GROUP BY sex ORDER BY sex")
    assert q.ok


def test_catalog_lists_registered_frames():
    q = _sql("SELECT sex, avg(salary) FROM df GROUP BY sex")
    cat = {c["name"]: c for c in (q.catalog or [])}
    assert "df" in cat and cat["df"]["n_columns"] == 5


@pytest.mark.parametrize("code", [
    # ORDER BY an aggregate (or its alias/position) is an unprotected oracle on
    # the exact, unrounded values — the *order* leaks beyond the rounded release.
    "SELECT sex, avg(salary) FROM df GROUP BY sex ORDER BY avg(salary)",
    "SELECT sex, avg(salary) AS m FROM df GROUP BY sex ORDER BY m",
    "SELECT sex, avg(salary) FROM df GROUP BY sex ORDER BY 2",
    # HAVING filters on exact aggregate values -> row presence is an oracle
    # (binary search recovers unrounded means / unnoised counts).
    "SELECT sex, avg(salary) FROM df GROUP BY sex HAVING avg(salary) > 50000",
    "SELECT region, count(*) FROM df GROUP BY region HAVING count(*) = 2",
    # arithmetic on aggregates: scaling defeats value-rounding (avg*1e6 then
    # divide mentally), and +0 would strip Tiltak-3 count noise from sums/counts.
    "SELECT sex, avg(salary) / 1000 FROM df GROUP BY sex",
    "SELECT sex, sum(salary) + 0 FROM df GROUP BY sex",
    # DISTINCT on non-count aggregates stays denied
    "SELECT sex, sum(DISTINCT salary) FROM df GROUP BY sex",
])
def test_oracle_channels_refused(code):
    r = _sql(code)
    assert r.ok is False, f"expected refusal: {code!r}"


# ---- red team: everything else is refused ------------------------------------

@pytest.mark.parametrize("code", [
    "SELECT * FROM df",                                     # row dump
    "SELECT salary FROM df",                                # bare column
    "SELECT salary FROM df LIMIT 1",                        # single row
    "SELECT DISTINCT region FROM df",                       # distinct value dump
    "SELECT sex, salary FROM df GROUP BY sex, salary",      # keys-only = DISTINCT dump (no aggregate)
    "SELECT max(salary) FROM df",                           # extreme
    "SELECT sex, min(salary) FROM df GROUP BY sex",         # extreme
    "SELECT quantile_cont(salary, 0.99) FROM df",           # order stat
    "SELECT string_agg(name, ',') FROM df",                 # aggregate-all-rows
    "SELECT list(salary) FROM df",                          # listification
    "SELECT first(salary) FROM df",                         # row identity
    "SELECT sex, avg(salary), (SELECT max(salary) FROM df) FROM df GROUP BY sex",  # scalar subquery leak
    "SELECT sex, row_number() OVER () FROM df",             # window = per-row
    "INSERT INTO df VALUES (1,'x','F','A',1)",              # not a SELECT
    "UPDATE df SET salary = 0",
    "DROP TABLE df",
    "CREATE TABLE x AS SELECT * FROM df",
    "COPY df TO '/tmp/leak.csv'",
    "PRAGMA database_list",
    "SET enable_external_access=true",
    "SELECT avg(salary) FROM df; SELECT * FROM df",         # multiple statements
    "SELECT * FROM read_csv('/etc/passwd')",                # file read
    "SELECT avg(salary) FROM nope GROUP BY sex",            # unknown table
])
def test_disclosive_sql_refused(code):
    r = _sql(code)
    assert r.ok is False, f"expected refusal: {code!r} -> {r.payload!r}"
