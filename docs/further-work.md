# Further work (parked threads)

Consolidated list of deferred work. Each has enough of a note to pick up later.

## Analysis methods

- **Phase B — weighted statistics** *(parked on request)*. Add `weights=` to group
  aggregations (`groupby[col].mean(weights='w')`) and to `ols` (→ WLS).
  **Safety:** enforce `min_n` on the **unweighted** row count of each group — a
  single row that "represents 1000 people" is still one individual. Weighted
  aggregates are otherwise safe. Important for register/survey data.
- **Phase B — quantile regression** *(parked)*. `df.quantreg(y=, x=[], q=0.5)`
  (statsmodels `QuantReg`), mirrors `ols`; coefficient table with per-term
  suppression.
- **Mixed / multilevel models** — `df.mixedlm(y=, x=[], groups=)` (statsmodels
  `MixedLM`). Fixed-effects summary is aggregate/safe; random-effect BLUPs stay
  private, small groups guarded.
- **Competing risks / KM confidence bands** — extend `kaplan_meier` with CIs
  (mask like the curve); `df.cumulative_incidence(...)` (lifelines
  `AalenJohansenFitter`).
- **Synthetic control — phase 2** — placebo/permutation inference (p-value) and
  the Aug/Robust/Penalized variants pysyncon already provides. See
  [deferred-synthetic-control.md](deferred-synthetic-control.md).
- **Interrupted time series / richer quasi-experiments** — CausalPy (heavy/PyMC;
  OLS backend first). Donor-weight / counterfactual disclosure caveats apply.

## Engine / platform

- **Loops** — allow `for` over non-data iterables. The multiple-results **output
  envelope now exists** (so multi-output is solved); remaining pieces: a safe
  `.groups()` accessor (iterating group labels can leak rare categories) and a
  server iteration/time cap. **The real risk is multi-query differencing** →
  rests on the audit layer, so loops are the one feature not secure-by-
  construction. Decide explicitly before enabling.
- **Security hardening / audit** — adversarial audit + expanded red-team of the
  `SafeColumn`/`SafeFrame` non-revelation invariant (the linchpin).
- **Secondary suppression — remaining Tiltak** — the configurable tier system and
  measures 1/2/3/5/6/7/8 are wired (see [suppression-measures.md](suppression-measures.md)).
  Still to do: **Tiltak 9** intercept hiding on low k-anonymity (`protect.risk`),
  **Tiltak 10** micro-aggregation + smoothing before percentiles (`protect` has
  `noise(method='group_mean')` / `_noise_group_mean` for the micro-agg step), and
  **Tiltak 4** hexbin scatter (needs a scatter surface first — note `protect.suppress`
  already has a plot/hexbin path). Also wire `protect.suppress`'s `dominance` / `p_percent`
  / `secondary` (complementary) suppression into released tables — all already in
  `_suppress_table`, just not passed through yet.
- **Multi-query audit / budget layer** — track queries across a session to
  mitigate differencing/averaging attacks (prerequisite for loops).
- **Server integration** — wire `run()` + the `render` pipeline + `catalog` into
  `microdata-api`'s `/run_extended` (submit/poll). `run()` is the synchronous
  core today.
- **Resource limits** — execution timeouts (DoWhy/pysyncon can be slow), memory
  caps.

## Reflections to act on (from the suppression-measures review)

- **Measure triage — not all earn their keep equally.** Tiltak 6 (edit
  isolation) and 3 (noise) address *distinct* attack vectors; 5/8 are
  defense-in-depth that overlap `min_n`; **7 is essentially a higher `min_n`
  dial** and 1 is a policy/legal choice. The biggest hole (multi-query
  differencing) is untouched by *all* of them. Frame the **tiers**, not the
  measure count, as the security story — the audit layer is still the linchpin.
- **Consolidate the measure helpers + add a coverage invariant.** The checks are
  inlined per verb across four files (winsor logic ×2, sparse-check ×4). Risk:
  a new reducer silently skips a control. Move the compute-time helpers into one
  `measures` module and add a test asserting every descriptive reducer routes
  through the floor+winsor helper — enforce coverage rather than remember it.
- **Winsorization inconsistency:** applied *globally* in frame/group reducers but
  *per-group* in `group_describe`. Same statistic, two code paths, slightly
  different numbers — reconcile.
- **Caching — defer to server/session integration, don't add now.** For a
  one-shot `run()` it saves almost nothing (dominant cost is the pandas op +
  model fit). At session/server scale, cache **per-dataset invariants** keyed by
  dataset identity (datasets are read-only → trivial invalidation): group-size
  tables (basis for suppression + sparse-check + descriptive floor at once),
  winsorized columns `(column, limits)`, risk reports `(dataset, quasi_ids)`,
  catalog schema. A **precomputed 1-way frequency map** is cheap and can power a
  front-end "this table will be mostly suppressed" hint; higher-order crosstabs
  are combinatorial → compute on demand and cache the result. **Caveat:** a
  sub-`min_n` cache is itself disclosive — server-side only, and cache hit/miss
  timing is a (small) side channel.

## Docs

- **README / API reference** — a user-facing tour of the whole safe surface
  (pandas-shaped verbs, tests, models, plotting, functions, datasets/catalog).
