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
  `SafeColumn`/`SafeFrame` non-revelation invariant (the linchpin); wire
  `protect`'s dominance / p% / secondary suppression into the verbs (only
  `min_n` + rounding are wired today).
- **Multi-query audit / budget layer** — track queries across a session to
  mitigate differencing/averaging attacks (prerequisite for loops).
- **Server integration** — wire `run()` + the `render` pipeline + `catalog` into
  `microdata-api`'s `/run_extended` (submit/poll). `run()` is the synchronous
  core today.
- **Resource limits** — execution timeouts (DoWhy/pysyncon can be slow), memory
  caps.

## Docs

- **README / API reference** — a user-facing tour of the whole safe surface
  (pandas-shaped verbs, tests, models, plotting, functions, datasets/catalog).
