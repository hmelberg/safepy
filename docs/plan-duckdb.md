# Plan: DuckDB (SQL) support in STRICT mode

**Decision: gated execution, not translation.** Unlike R (a general-purpose
language, translated and never executed), SQL is declarative and DuckDB runs
in-process in Python — which is what Anvil gives us. The dialect executes the
user's SQL, but only after a static AST gate, inside a locked engine, and the
result exits only through the shared audited suppressor. Three layers:

1. **Static gate on the parsed AST.** `json_serialize_sql` parses the SQL
   *without executing it* (the SQL string is passed as a bound parameter, so the
   parse call itself cannot be injected). The gate enforces:
   - exactly **one statement**, and it must be a `SELECT` (INSERT/UPDATE/CREATE/
     COPY/PRAGMA/SET… are refused at parse);
   - **default-deny on functions everywhere in the tree** — a whitelist of
     scalar/element-wise functions plus the safe aggregates. `min`/`max`/
     `quantile_*`/`first`/`last`/`string_agg`/`list`/`read_csv`… are refused even
     inside subqueries (consistent with the pandas/polars posture: extremes are
     not even allowed as private intermediates);
   - **no window functions**, no `USING SAMPLE`, no `HAVING`/`QUALIFY` (v1);
   - the **outer select list is the release boundary**: every item must be a
     GROUP BY key (plain column) or a whitelisted aggregate
     (`avg`/`sum`/`count`/`count(*)`/`median`/`stddev`/`var_samp`), and at least
     one aggregate is required (a keys-only select is a DISTINCT value dump).

2. **Locked execution.** `SET enable_external_access=false` kills COPY/ATTACH/
   INSTALL/httpfs/read_csv at the engine level (verified); `lock_configuration`
   freezes settings. Only the registered private frames are visible. When the
   policy winsorizes (Tiltak 2), numeric columns are winsorized **at
   registration** — every moment aggregate the SQL computes is capped, matching
   the pandas dialect's global-quantile winsorize byte-for-byte on unfiltered
   queries (median/count are unaffected by tail caps). Known deviation: a
   WHERE-filtered moment stat sees full-table caps rather than subset caps
   (slightly more conservative than pandas).

3. **Release through the shared core.** The outer select is rewritten by JSON
   surgery (each aggregate paired with a `count` over the same argument;
   `count(*)` pairs with itself) and deserialized back to SQL via
   `json_deserialize_sql`; one execution returns values + paired counts. Each
   aggregate column is released via `SafeVerbs._release_group_agg` — the same
   suppressor as pandas/polars (min_n floors per stat, count noise, rounding).
   Multi-aggregate selects combine like the polars compound `agg`.

**Why inner shaping is free:** subqueries/CTEs/joins/LIMIT are private
intermediates. Shape down to one row (`FROM (... ORDER BY salary DESC LIMIT 1)`)
and the paired count is 1 → the cell is suppressed. Same argument as polars
`filter`/`with_columns`.

## Status

First slice implemented (`safepy/duckdb_api.py`, `dialect="duckdb"`):
grouped aggregates (single + multi), whole-table aggregates, WHERE, subqueries,
CTEs, joins between registered frames; equivalence-tested against the pandas
dialect incl. the microdata count-noise tier; red-team suite in
`tests/test_strict_duckdb.py`.

## Not yet / later

- `HAVING` (walk it with the same whitelist), `count(DISTINCT x)` (→ nunique
  floor), grouping by expressions (`GROUP BY substr(region,1,1)`).
- Window functions inside subqueries (polars allows `.over()`; SQL v1 denies all
  windows — revisit with a per-type whitelist that still denies `min`/`max`
  windows).
- Order statistics under the shared order-stat rule (`quantile_cont` with the
  min-support check) — needs a dedicated release path, like `SafeColumn.quantile`.
- Resource limits (memory cap / statement timeout) as robustness hardening.
- The catalog for SQL sessions (schema of registered frames).
