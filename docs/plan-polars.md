# Plan: Polars support in STRICT mode

**Decision: integrate as a second facade *dialect* over the shared security core.
Do NOT build a separate library.** The disclosure-control core is not
pandas-specific; duplicating it would mean two copies of the suppression logic to
audit and keep in sync — unacceptable for a security tool.

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

## Architecture

1. **`protect` boundary via small-aggregate conversion.** Keep private per-row
   data in polars; compute the aggregate natively; convert the *small* result to
   pandas (`.to_pandas()`, arrow zero-copy) for `protect.suppress` + the Tiltak
   measures; return a `Released` (already neutral). Private data never crosses.
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
