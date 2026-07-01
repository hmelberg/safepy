# Plan: Polars support in STRICT mode

**Decision: integrate as a second facade *dialect* over the shared security core.
Do NOT build a separate library.** The disclosure-control core is not
pandas-specific; duplicating it would mean two copies of the suppression logic to
audit and keep in sync — unacceptable for a security tool.

## Status (2026-07-01): Milestone 1-b, first slice landed

The build is sequenced **surface first, backend second** (see "Build order"
below). The first slice is implemented and tested:

- `run(code, sources, profile=STRICT, dialect="polars")` selects the polars
  surface; sources are wrapped in `SafePolarsFrame`.
- End-to-end: `df.filter(pl.col(...) <cmp> v).group_by(...).agg(pl.col(c).<reducer>())`
  for the safe reducers (`mean/sum/count/median/std/var`), plus
  `group_by(...).len()`. Real polars evaluates the shaping; the shaped frame is
  `.to_pandas()`-ed at the terminal verb and routed through the **existing
  `SafeVerbs`**, so suppression is byte-identical to the pandas dialect.
- New code: `safepy/polars_api.py` (`SafePl`/`SafeExpr`/`SafePolarsFrame`/
  `SafePolarsGroupBy`). Gate whitelists `import polars` and denylists the polars
  raw-export / value-ordered names. Dangling polars intermediates are refused.
- Tests: `tests/test_strict_polars.py` — pandas-equivalence + boundary refusals.

**Landed since:**
- `select` / `with_columns` / `SafeExpr.alias` — column selection and derived
  columns (→ private frame), whole-frame `select(reducer)` → suppressed scalar.
- `.str` / `.dt` accessors on `SafeExpr` (element-wise, whitelisted, → derived
  private expressions) and `pl.when(...).then(...).otherwise(...)`.
- Reducers over *derived* expressions (e.g. `pl.col('name').str.len_chars().mean()`)
  route through a materialized value column into the shared suppression path, so
  `_col` is no longer required to be a plain column.
- Compound grouped `agg` (multiple reducers → a suppressed frame, aligned on the
  shared group index; alias honored as the column name) and multi-aggregation
  whole-frame `select` (→ a series of suppressed scalars).
- Polars branch in `_build_catalog` — polars sources appear in the schema catalog
  (schema/null_count/height introspection, suppressed counts).

**Milestone 2 — native polars compute (grouped aggregates): DONE.**
- `safe.py` split into a pandas *compute* step and a backend-neutral *release*
  step (`SafeVerbs._release_group_agg`) — the single audited suppressor
  (protect.suppress + count-noise + audit) for grouped aggregates, whatever
  backend produced the `(table, counts)`.
- `polars_api._native_group_agg` computes the per-group aggregate + row counts
  **in polars** (winsorization included, via linear-interpolation quantiles that
  match `protect.winsorize` byte-for-byte — verified numerically and by
  equivalence tests across mean/sum/std/var/median/count, and at the microdata
  tier with count-noise). Only the small per-group result crosses to pandas; the
  private per-row frame stays in polars. Audit records `backend="polars"`.
- Whole-frame scalar `select(reducer)` converts only the single needed column
  (not the whole frame) and reuses the shared `SafeColumn` reducer.

**Breadth: model / stat verbs + charting + lazy.**
- The terminal (Released-returning) verbs of the pandas `SafeFrame` are delegated
  from `SafePolarsFrame` via an explicit whitelist (`_DELEGATED_VERBS`): `ols`,
  `logit`, `poisson`, `cox`, `kaplan_meier`, `logrank`, the AFT models, `rmst`,
  `feols`, `iv`, `ate`, `refute_ate`, `synthetic_control`, `corr`, `cov`,
  `describe`, the hypothesis tests, `value_counts`/`crosstab`/`pivot_table`, and
  the per-column frame reducers. Each converts the frame to pandas once and calls
  the already-audited verb. Intermediate-returning verbs (`assign`/`where`/
  `groupby`/`merge`/…) and `propensity` (returns a private column) are excluded —
  the facade stays the boundary.
- `.plot` works on any polars aggregate for free (`Released.plot`), e.g.
  `df.group_by('sex').agg(pl.col('salary').mean()).plot.bar()`.
- **Lazy frames:** a `pl.LazyFrame` source is supported — shaping stays lazy, and
  the frame is `.collect()`-ed only at the conversion boundary (native aggregate,
  scalar, delegated verb, catalog). Schema introspection uses `collect_schema()`.

**Not yet:** native compute for `value_counts`/`crosstab`/`pivot_table` (they
delegate to pandas today); native whole-column scalar reducers (one-column pandas
conversion via `SafeColumn`); null group-key parity (polars keeps null-key groups;
pandas `groupby(observed=True)` drops them); query-plan introspection off the
LazyFrame plan (the audit-friendly win the lazy seam now enables).

## The two axes (keep them separate)

- **Backend** — what computes the private intermediate data. Today: pandas.
- **Surface** — what syntax the user writes. Today: pandas-shaped `SafeFrame`.

"Polars users" = people who want the polars *surface*. That's the goal here.
Polars-as-backend (speed/arrow) is a separate, optional win.

## What is shared (reuse unchanged) vs dialect-specific

**Shared security core — do not duplicate:**
- `ast_gate.py` (needs polars-aware *additions*, not a fork)
- `policy.py` — `Policy`/`Suppression`/preset tiers/all Tiltak thresholds
- `protect` wiring, `_stop_if_too_sparse`, winsorization, count noise, order-stat rule
- `result.py` (`Released`, payload dicts are backend-neutral), the mediator, `api.run`
- `runtime.py` exec model (namespace just holds a `SafePolarsFrame` instead)

**Dialect-specific (new):**
- `SafePolarsFrame` / `SafePolarsColumn` / `SafeExpr` facade
- the verb *compute* (group_by/agg/filter in polars)

## Build order (surface first, backend second)

- **Milestone 1-b — polars surface, pandas backend.** Real polars evaluates the
  shaping; convert the shaped (still-private) frame to pandas at the *terminal*
  verb and reuse the existing `SafeVerbs`. Zero changes to the security core;
  suppression proven byte-identical by pandas-equivalence tests.
- **Milestone 2 — native polars compute.** Swap the reduction into polars behind
  the identical facade, after factoring the suppressor out of `safe.py` into a
  backend-neutral core. Refactor against the green equivalence suite.

**Correction to the original "small-aggregate conversion" idea below:** the
release path in `safe.py` does *not* accept a precomputed aggregate — it needs
the **raw column** (it recomputes paired group counts, winsorizes the column for
Tiltak 2, and computes order statistics over the full column). So "convert only
the small result" cannot reach Tiltak parity. M1-b converts the shaped *frame*
(still private — same trust boundary pandas already sits on) and lets `SafeVerbs`
do the counts/winsorize/suppress. "Private data never crosses" is a backend/perf
aspiration for M2, not a security property; pandas already holds private data.

## Architecture

1. **`protect` boundary.** *(M2 aspiration; M1-b converts the shaped frame — see
   the correction above.)* Keep private per-row data in polars; compute the
   aggregate natively; convert the *small* result to pandas (`.to_pandas()`,
   arrow zero-copy) for `protect.suppress` + the Tiltak measures; return a
   `Released` (already neutral).
2. **`SafePolarsFrame` mirrors polars idioms:** `df.filter(...)`,
   `df.select(...)`, `df.with_columns(...)`, `df.group_by('g').agg(...)`.
3. **`SafeExpr` wraps `pl.col(...)` expressions** with a whitelist mirroring the
   `SafeColumn` surface (arithmetic, comparisons, `.str`, `.dt`, safe reducers).
   Polars' native expression system replaces the need for our `assign()` string
   compiler (`_compile_expr`) — wrap real expressions instead of a mini-DSL.
4. **Gate additions:** whitelist `polars` import + `pl.col`/`pl.lit`/`pl.when`;
   deny the polars value-ordered / row-identity expression methods (see below).

## Security: the deny-list ports directly

The compute-private and indirect-disclosure principles are unchanged; only the
method *names* differ. Deny these polars expressions/methods (value-ordered
subset selection or row identity — same class as pandas `nlargest`/`rank`/`head`):

`head`, `tail`, `slice`, `gather`, `get`, `top_k`, `bottom_k`, `arg_max`,
`arg_min`, `arg_sort`, `sort` (+ head), `first`, `last`, `item`, `sample`,
`search_sorted`, `to_list`/`to_numpy`/`to_series` (raw export), `rank`.

Allow (private per-row → released only via suppressed aggregate): arithmetic,
comparisons, `when/then/otherwise`, `.str.*`, `.dt.*`, `fill_null`, `clip`,
`.over()` (window = group-broadcast, like `transform`), and the safe reducers
(`mean/sum/count/median/std/var`) routed through the suppression verbs.

## Open decisions (settle early in the build)

- **Eager vs lazy.** Start **eager** (simpler to reason about). Lazy `LazyFrame`
  gives an inspectable query plan (audit-friendly) but adds complexity — revisit.
- **Dialect selection.** A `dialect="polars"` arg on `run()`, a distinct profile,
  or auto-detect from the source type? Recommend an explicit arg.
- **Optional dependency.** `safepy[polars]`, lazy-imported in a `polars_api`
  module, so pandas-only users don't pull polars.
- **Catalog/schema** needs a polars schema-introspection path (names/dtypes/
  n_missing/n_rows) — mirror `_build_catalog`.
- **protect native-polars path** later (optimization) vs always-convert (simpler,
  start here).

## Effort

Bounded and mostly additive: the security core is reused. Work = (a) a backend
seam in the verb layer, (b) the polars facade + `SafeExpr` whitelist, (c) gate
additions, (d) polars catalog/schema. Most of safepy (policy, Tiltak, protect,
mediator, result, runtime) is untouched.

## To seed the build chat

- Point it at this doc and `DESIGN.md`.
- Auto-loaded memories already carry the principles (compute-private,
  indirect-disclosure, suppression tiers). Reaffirm: **STRICT focus; reuse
  `protect`; the security core must not be duplicated; deny value-ordered /
  row-identity polars expressions.**
- Interpreter: `C:\ProgramData\anaconda3\python.exe`; confirm `polars` is
  installed (`pip install polars`) before starting.
- First milestone: `SafePolarsFrame` + `SafeExpr` covering
  `df.filter().group_by().agg(mean/count)` end-to-end through the existing
  suppression + Tiltak path, with a handful of tests mirroring `test_datasets`.
