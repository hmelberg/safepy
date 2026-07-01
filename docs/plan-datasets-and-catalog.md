# Plan: multiple datasets, derived frames, merge, and a schema catalog

**Status:** planned, not implemented. Raised alongside Phase A. This is an
evolution of the execution model (from "one script → one result" to "a session
with several named datasets → results + a catalog").

## What we want

1. **Multiple input datasets** in one run (server has access to several).
2. **Derived datasets** created in the script: `new_df = df.aggregate(by=...)`,
   filters, joins — usable further and mergeable.
3. **Merge/join** across datasets: `merged = df.merge(other, on='id')`.
4. **A catalog** returned to the front end: the *names* of all datasets and their
   *variables* with light metadata (dtype, # missing, # rows) — **never the data
   or any values**.

## Model change: SafeSession

Today `run(code, sources, ...)` mediates the *final expression* into one
`SafeResult`. New model:

- `sources` already accepts several frames → each becomes a `SafeFrame` (works
  today). Merge is then just a `SafeFrame` method.
- **Derived frames are private `SafeFrame`s**, not `Released`. A groupby result
  kept as an intermediate is private data (fine until *released*); suppression
  applies only at release. So we distinguish:
  - **dataset-producing** verbs → return a `SafeFrame` (`where`, `assign`,
    `merge`, a new `aggregate`/`summarise`);
  - **terminal reducers** → return `Released` (`.mean()`, tests, models, …).
- After exec, `run()` inspects the namespace for `SafeFrame` bindings (assigned
  names) and returns **both** the final released result(s) **and** the catalog.

## New verbs

- `df.merge(other, on=, how='inner'|'left'|...)` → `SafeFrame`. `other` is another
  `SafeFrame` in scope. Merge itself discloses nothing (join keys are columns;
  downstream aggregation is suppressed as usual). Validate keys exist.
- `df.aggregate(by=, name=(col, func), ...)` (pandas named-agg shape) → a new
  `SafeFrame` of group statistics, so `new_df = df.aggregate(...)` is a reusable
  dataset that can be merged/re-aggregated. (The terminal
  `groupby(...)[col].mean()` still returns a released table.)

## The catalog (schema + suppressed counts only)

For each dataset in the session, return:

```json
{"name": "df",
 "n_rows": <rounded/suppressed>,
 "columns": [{"name": "salary", "dtype": "float64", "n_missing": <rounded>},
             {"name": "sex",    "dtype": "category", "n_missing": 0}]}
```

**Disclosure rules for the catalog:**
- **Safe metadata:** dataset names, column names, dtypes.
- **Counts (`n_rows`, `n_missing`)** are frequencies → round (policy `round_to`)
  and/or suppress if `< min_n`.
- **Never** include: values, examples, min/max, distinct values, top categories,
  or `n_unique` verbatim (a unique-per-row column is disclosive). If a
  "uniqueness" hint is wanted, expose only a bucketed/boolean flag, not the count.

## Ties to other threads

- This is the same "multiple outputs" question as the parked **loops** work — a
  session returns several things (results + catalog). Settle the output envelope
  here and loops become easier.
- The catalog mirrors microdata.no's variable browser (name/type/missing), which
  is the familiar UX we're targeting.

## Suggested phasing

1. **Merge + `aggregate`** (dataset-producing verbs; single-run, no session
   change yet). Small, high value.
2. **SafeSession + catalog** — `run()` returns `{results, catalog}`; collect
   `SafeFrame` bindings post-exec; build the suppressed schema catalog.
3. Fold in the multiple-results output envelope (shared with loops).
