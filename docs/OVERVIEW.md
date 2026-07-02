# safepy — project overview (conversation seed)

> A concise, current snapshot of the project's aims and architecture. Point a new
> conversation at this file first, then at `DESIGN.md` (deep design) and the
> per-dialect `docs/plan-*.md`.

## The aim

Let an analyst submit a **script** to a server that holds **private,
individual-level data**, run it, and return **aggregate results only** — never a
row, a scalar extreme, or anything that identifies an individual (directly or
indirectly). Inspired by **microdata.no**: submit a script, get aggregates back.
Unlike microdata.no's bespoke DSL, the input is a **restricted subset of real
analysis languages**, so analysts write familiar code.

An institution keeps secret data on a secure host (Anvil, or another secure
place); users analyse it in the language of their choice via an API call that
returns results + metadata, but never individual-level information.

## Deployment context

- **Backend:** Anvil (hosted = Python-only: no R binary, no subprocess, no Node).
  So anything that must run there is pure Python.
- **Frontend:** `m2py` (`index.html`).
- Eventually folds into `microdata-api` (the Anvil server; `run()` is the
  synchronous core of a future `/run_extended` endpoint).
- Disclosure control itself is the separate **`protect`** package; safepy owns
  the *language frontends* + the trusted release path.

## The core idea: many dialects, one audited release core

Every dialect produces the same kind of result — a **suppressed `Released`
aggregate** — through **one shared, audited suppressor** (`SafeVerbs._release_*`
+ `protect.suppress`). No dialect reimplements disclosure control; adding a
language is a *frontend*, not a second security engine.

Two postures share one engine (selected by `Profile`):
- **STRICT** (the focus) — *safe by construction*: the sandbox namespace holds
  only capability-facade objects; disclosive operations don't exist. Audit
  surface is the small facade method list.
- **OPEN** — *probably safe*: real pandas + a gate deny-list + the mediator.

## Dialects (all STRICT, all through the shared core)

| Dialect | How it works | Status |
|---|---|---|
| **pandas** | `SafeFrame`/`SafeColumn` capability facade (mirrors real pandas call shapes) | mature; models, plots, causal/survival verbs |
| **polars** | polars facade + **native polars compute**; suppression byte-identical to pandas | mature; group/agg, value_counts, crosstab, pivot_table, frame reducers, lazy frames, delegated model/plot verbs |
| **R** | **translated, never executed** — a restricted dplyr/base-R surface parsed (hand-rolled parser + `r_expr`) and mapped to the facade | broad: dplyr verbs, `mutate`/`case_when`, joins, pivots, multi-statement scripts, `lm`/`glm`/`feols`/`iv`/`coxph`/`survfit`/`ate`, `ggplot`/`hist`/`boxplot` |
| **DuckDB (SQL)** | **gated execution** in a locked engine (`enable_external_access=false`), AST-gated, released via the shared suppressor with paired counts | grouped/whole-table aggregates, WHERE, subqueries, CTEs, joins, GROUP BY expressions, `count(DISTINCT)`; oracle channels (ORDER BY aggregate, HAVING, arithmetic-on-aggregate) refused |

Selected via `run(code, sources, profile=STRICT, dialect="pandas"|"polars"|"r"|"duckdb")`.

## Security model (what "safe" means here)

- **Aggregates only.** A raw result has no provenance the mediator can trust, so
  only values produced by the trusted verbs (which know the contributing counts)
  are released.
- **Suppression + Tiltak measures:** min-cell `min_n`, descriptive-population
  floors, winsorization (Tiltak 2), cell-key count noise (Tiltak 3), rounding,
  sparse-table stop (Tiltak 5), edit-size guard (Tiltak 6), percentile
  coarsening (Tiltak 8). Preset tiers: `light` / `standard` / `microdata`.
- **Extremes/order stats** released only under the order-stat rule
  (≥ `min_n` observations at/beyond the value); categorical extremes refused.
- **Models:** per-coefficient / at-risk suppression; user formula strings never
  reach an evaluator (validated + rebuilt from column names).
- **Charts:** a chart renders an already-suppressed aggregate, never raw data;
  raw-data plots (scatter/line/`geom_point`) refused.
- **Never execute untrusted code** except the gated, sandboxed DuckDB path
  (locked config, no file/network, result forced through the suppressor).

## Where to look

- `DESIGN.md` — deep design (threat model, the provenance boundary, profiles).
- `docs/plan-polars.md`, `docs/plan-r.md`, `docs/plan-duckdb.md` — per-dialect
  architecture, status, and roadmap.
- `docs/security-indirect-disclosure.md`, `docs/suppression-measures.md` — the
  disclosure-control rules.
- `safepy/` — `api.py` (the `run` entry), `safe.py` (release core), `safeframe.py`
  (pandas facade), `polars_api.py`, `r_api.py` + `r_expr.py`, `duckdb_api.py`.
- `tests/` — including `tests/attacks/` (the executable threat model).

## Status (2026-07-02)

Runnable; **580 tests passing** across the four dialects. Work is pushed to the
`dev` branch. Highest-value next steps: wire the dialects into the Anvil
`/run_extended` endpoint (the product payoff), a cross-dialect red-team sweep, or
per-dialect long-tail polish.
