# Secondary suppression measures & aggressiveness tiers

safepy's primary control is `min_n` cell suppression at every release path. On top
of that sits a configurable set of **secondary measures** modelled on
microdata.no's *Tiltak*, wired to the `protect` package (never reimplemented).

## Configuring aggressiveness

All measures live on one `Suppression` config (`safepy/policy.py`). Named preset
tiers bundle them; any single knob can be overridden per run.

| Tier | Intended level | What it turns on |
|---|---|---|
| `off` | public / OPEN sandbox | `min_n=1` only |
| `light` | development | `min_n=5`, no secondary measures |
| `standard` | protected (default) | winsorization, sig-fig percentiles, descriptive floor (10), edit floor (10), sparse-table stop, intercept k-anon |
| `microdata` | sensitive | everything in `standard` + the 1000-person population floor (noise & micro-aggregation land in later batches) |

`resolve_policy` maps protection level → tier by default. Override with a preset
name or a `Suppression` instance:

```python
run(code, {"df": df}, level="protected")                     # -> standard tier
run(code, {"df": df}, suppression="light")                   # lighter
run(code, {"df": df}, suppression=Suppression(min_n=5, winsorize=(0.01, 0.99)))
```

## The measures (status)

| # | Tiltak | Knob | Status |
|---|---|---|---|
| 1 | Minimum population | `min_population` | ✅ descriptive stats suppressed below the floor (microdata: 1000) |
| 2 | Winsorization (2%) | `winsorize` | ✅ mean/std/sum/hist/box + describe min/max via `protect.winsorize`; **not** medians/quartiles, **not** regression |
| 3 | Noise on counts | `count_noise` | ⏳ batch 2 (`protect.noise`) |
| 4 | Hexbin scatter | — | ⏳ needs a scatter surface first |
| 5 | Stop sparse tables | `max_low_cell_share` | ✅ value_counts/crosstab/pivot_table refused when > share of cells < `min_n` |
| 6 | No edit affecting < N units | `min_edit_units` | ✅ replace/map/where/mask/fillna/clip refuse changing `[1,N)` or `(n-N,n)` rows (all/none is fine) |
| 7 | No descriptive stats for pop < N | `min_descriptive_n` | ✅ mean/std/percentile suppressed below floor; counts/sums exempt |
| 8 | Percentiles to 3 sig figs | `percentile_sig_figs` | ✅ median/quartiles/min/max coarsened; mean/std unaffected |
| 9 | Hide intercept if k-anon < 5 | `intercept_k_anon` | ⏳ batch 3 (`protect.risk`) |
| 10 | Micro-aggregate percentiles | `microaggregate` | ⏳ batch 3 |

## Design notes

- **Where each measure hooks:** floors and precision live in the SafeColumn/SafeFrame
  reducers and `_order_stat`; the edit floor wraps the recode verbs; the sparse-table
  stop guards the count-table verbs; winsorization is applied to the *source series*
  before moment stats and to the order-stat `winsorize` bound for extremes.
- **Counts vs descriptive stats:** counts/sums/frequencies use the plain `min_n`;
  mean/std/percentiles use the higher descriptive floor (`max(min_n,
  min_descriptive_n, min_population)`). This mirrors Tiltak 7's explicit exemption.
- **Regression is never winsorized** (Tiltak 2) — model estimates are not personal
  data and run on the raw values.
- **Error messages never state the affected count** (e.g. the edit-floor refusal),
  since that count is itself disclosive.
- **Differencing still open:** these are cell/table/edit-level controls. The
  multi-query differencing gap (see [security-indirect-disclosure.md](security-indirect-disclosure.md))
  is unaffected and still awaits the audit/budget layer.
