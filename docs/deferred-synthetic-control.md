# Deferred: synthetic control (and interrupted time series)

**Status:** noted, not implemented. Revisit when panel/aggregate-unit data is a
priority. CausalPy is deferred because it pulls **PyMC** (heavy) and is
formula-based; lighter options exist (below).

## The disclosure concern is specific to SC: donor weights

Synthetic control builds a counterfactual for a treated unit as a **weighted
average of control ("donor") units**. Unlike the estimators we already wrap,
SC's risk is *not* mainly the point estimate — it's the **weights**:

- A weight of ~1 on a single donor means "that unit's trajectory *is* the
  counterfactual" → a near-complete disclosure of that unit.
- If donor units are **individuals**, releasing weights (or a concentrated
  counterfactual) is identifying. If donor units are **aggregates**
  (regions/firms/countries — SC's usual setting), it's fine.

So the safe design is the same regardless of library:

1. **Never release raw donor weights.** Release only the aggregate
   **treatment-effect path** (gap = treated − synthetic) and pre-period fit
   (RMSPE), plus placebo/permutation **inference** (all aggregate).
2. **Guard against weight concentration.** Refuse (or coarsen) if the top donor
   weight exceeds a threshold, or report only an "effective number of donors"
   (e.g. inverse Herfindahl) to show the synthetic isn't one unit.
3. **Require units to be aggregates of ≥ min_n individuals** (or restrict SC to
   pre-aggregated panel data). This is the cleanest guard in a microdata context.
4. **Per-period gaps** get the same treatment as survival curves — release only
   where the underlying counts are ≥ min_n.

Because the whole risk is the weights, **owning a thin implementation is
attractive** — the SC weight problem is a small constrained least squares
(w ≥ 0, Σw = 1 minimizing pre-period fit), solvable with `scipy`/`cvxpy`. A
library just computes the weights we then have to guard anyway.

## Lighter library options (vs CausalPy/PyMC)

| Option | Weight | Notes |
|---|---|---|
| **pysyncon** | light (numpy/pandas/scipy) | Dedicated SC: classic (Abadie), augmented, robust, penalized. Best "drop-in" light library. |
| **roll our own** | lightest (scipy or cvxpy) | SC weights = a small QP. Full control of what's exposed; least code surface to audit. Best fit for our disclosure needs. |
| **synthdid / pysynthdid** | light–moderate | Synthetic *difference-in-differences* (Arkhangelsky et al.) — often preferred to classic SC now. |
| **scpi_pkg** (`scpi`) | moderate (adds cvxpy, plotnine) | Authoritative SC **with prediction intervals/inference** (Cattaneo et al.). Heavier but principled inference. |
| **SparseSC** (Microsoft) | heavier | SC at scale; sklearn-based. Overkill here. |
| **CausalPy (OLS backend)** | moderate | Avoids PyMC by using its scikit-learn/OLS path; still formula-based. |

## Recommendation when we implement

Prefer **pysyncon** *or* a **thin in-house SC** (scipy/cvxpy), wrapped as a
curated verb like `df.synthetic_control(unit=, time=, outcome=, treated_unit=,
treatment_time=, predictors=[...])` returning only:

- the **effect path** (gap per period, suppressed where counts are thin),
- **pre-period RMSPE** and an **effective-number-of-donors** figure,
- optional **placebo inference** (permutation p-value),

with donor weights **never** returned and a **weight-concentration / min_n unit**
guard. Treat it as a *curated verb* (like `df.ate`), not a library-mimicking
facade — same reasoning as DoWhy.
