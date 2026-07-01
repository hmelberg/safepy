# safepy — design

## Goal

Let an analyst write a *familiar subset of Python* (pandas now; polars,
statsmodels, lifelines, matplotlib, plotly later) against private,
individual-level data and get back **aggregate results only** — never a row, a
scalar extreme, or any view of an individual. Like microdata.no, but the input
language is restricted Python rather than a bespoke DSL.

## Posture (decided)

- **Sandbox, not translator.** The server runs the user's AST-gated Python
  directly. This is the path the m2py *safestat* spec (2026-06-29) deliberately
  rejected for sensitive data, so safepy is scoped to **public/local** data
  and treated as a research track. `protected` keeps the sandbox enabled as a
  deliberate research configuration; `sensitive` forbids it (use the
  translate-to-artifact frontend there).
- **Standalone now, fold into m2py later.** Built against a thin adapter
  interface so it slots in beside `py2m`/`r2m` as a Python *frontend*.
- **Reuse, don't reimplement.** All data-side and result-side disclosure
  control is the existing [`protect`](../protect) package. safepy only owns
  the *language frontend*: the gate, the sandbox, output mediation, and the
  curated safe verbs.

## Threat model

The adversary is the analyst submitting code. Three classes of attack:

1. **Direct row disclosure** — `df.head()`, `print(df)`, `df.iloc[0]`,
   `df.values`, `df.to_csv()`, `df['x'].tolist()`, positional indexing.
2. **Statistical disclosure via "aggregates"** — `max`/`min`/`describe` (the
   tails *are* individual values), `nlargest(1)`, `idxmax`, `groupby(unique_id)`,
   `groupby(...).first()`, a mean over a singleton group, a frequency cell of 1.
3. **Code-execution escapes** — `eval`/`exec`/`__import__`/`open`,
   `getattr`/dunder walks (`().__class__.__bases__[0].__subclasses__()`),
   callables into `apply`/`map`/`pipe`, the `query`/`eval` mini-language,
   exceptions that embed data values, f-strings that build data-bearing strings.

## Defense in depth (four layers)

```
code ──▶ [1 AST gate] ──▶ [2 restricted runtime] ──▶ value
                                                       │
                                              [3 output mediator]
                                                       │
                                         [4 protect.suppress] ──▶ SafeResult
```

1. **AST gate** (`ast_gate.py`) — default-deny on node *types*; structural rule
   (simple assignments + one final expression, no loops/defs/lambdas/
   comprehensions/imports); dunder ban; bare-call allow-list; method deny-list
   for row-dumps, extremes, and callable-taking verbs. Kills classes 1 and 3,
   and the easy part of 2. Mirrors `m2py/m2py_runtime/exprcompile.py`'s
   node-by-node discipline.
2. **Restricted runtime** (`runtime.py`) — stripped `__builtins__`, library
   handles bound explicitly, exceptions sanitised so no data value escapes in a
   message. Defence-in-depth, not the primary guard.
3. **Output mediator** (`mediator.py`) — the *only* exit. The runtime returns
   the result **object**, never its repr, so a bare `df` yields an object the
   mediator must clear, not a printed table.
4. **`protect.suppress`** — min-cell, (n,k) dominance, p%-rule, rounding,
   secondary suppression. Already built; safepy calls it.

## The load-bearing boundary: provenance

The mediator **cannot decide disclosure from a raw result's values.** A table of
means that happen to be integers is indistinguishable from a table of counts; a
scalar mean is indistinguishable from a scalar max. (We learned this the honest
way — an early value-sniffing heuristic released a means table.) Therefore:

- **Raw pandas results are default-denied.** Compute freely for intermediate
  steps, but the *released* value must carry provenance.
- **`safepy.safe` verbs are the trusted release path.** Each computes the
  aggregate *together with its group counts*, runs `protect.suppress`, and
  returns a `Released` value the mediator trusts. `safe` is **policy-bound**:
  `min_n` defaults to the policy floor and callers may only make it stricter.

This boundary is the seed of the **SafeFrame facade** below.

## The two profiles (both built, one engine)

The permissive sandbox can be made *very good* but never *provably complete*,
because its trusted surface is all of pandas/numpy/statsmodels. So safepy
ships two postures that share one engine (gate, runtime, mediator, policy,
`protect`) and differ only in **what is put in the sandbox namespace**, selected
by `Profile` (which follows `ProtectionLevel`):

| | **OPEN** (`policy.Profile.OPEN`) | **STRICT** (`policy.Profile.STRICT`) |
|---|---|---|
| Namespace | `pd`, `np`, raw frame, `safe` | `safe` + `SafeFrame`-wrapped sources only |
| Defense | enumeration: gate denylist + mediator provenance | construction: disclosive capabilities aren't reachable |
| Audit surface | all of pandas (open-ended) | the `SafeFrame` method list (small, closed) |
| Property | "probably safe" | "safe by construction" |
| For | public / local | protected / sensitive |

In STRICT mode `df.head()` is not blocked — it **doesn't exist**, because
`SafeFrame` has no such method. The facade mirrors pandas' **real call shapes**
(`safeframe.py`): `df['salary']` and `df[['a','b']]` (selection), `df[mask]`
boolean filtering, `df.groupby('sex')['salary'].mean()` (the traditional grouped
shape), whole-column reducers `df['salary'].mean()` (→ a suppressed scalar),
`df['region'].value_counts()`, `assign`/`where`, plus the regression/survival
verbs. Every terminal reducer returns a suppressed `Released` aggregate.

The linchpin is **`SafeColumn`**: it carries a column through comparisons (→
mask), arithmetic (→ derived column), `isin`/`isna`/`between`, a small `.dt`
accessor, and the safe reducers — while exposing **no** value: no `__repr__`/
`__iter__`/`values`/`tolist`/`max`/`min`/`quantile`/`__getitem__`/scalar
coercion. Reducers are sound because *we* own them and therefore know the
contributing count, so the same `min_n` rule protects whole-column, filtered,
and grouped aggregates alike (`df[df['region']=='Z']['salary'].mean()` over 2
people is suppressed). The raw objects live in `_df`/`_s`, unreachable from user
code (the gate blocks `_`-attributes). A dangling intermediate (`SafeFrame`,
`SafeColumn`, grouped object) returned as the final result is refused by the
mediator (`adapters/safeframe_adapter.py`).

The same `safe.*` verbs power both profiles (they unwrap a SafeFrame or take a
raw frame), so analysis code is largely portable between them.

The look-alike `pd`/`np` facades (`namespaces.py`) take and return `SafeColumn`s
(or a suppressed `Released` table) and borrow their policy from the column passed
in. They implement only a whitelist (`np.log/exp/sqrt/abs/where`,
`pd.crosstab/cut/to_datetime`); any other attribute raises a clear "not
available" error. So `np.log(df['wage'])` works but `np.array(...)`/`pd.read_csv`
do not.

## Regression & survival (statsmodels + lifelines)

`stats.py` adds `ols`/`logit`/`poisson` (statsmodels) and `cox`/`kaplan_meier`
(lifelines), reachable as `safe.ols(...)` and as `SafeFrame.ols(...)`. Two
disclosure dangers shaped the design:

- **No *raw* user formula strings reach patsy.** A patsy formula is `eval`-ed,
  and the AST gate can't see inside a string literal. Two safe entry points:
  the keyword API (`safe.ols(y=, x=[])`) builds the formula from validated
  names; and the `smf` facade (`smf.ols("y ~ age + C(sex)", data=df)`) accepts a
  formula string but **parses it with our own whitelisted grammar**
  (`formula.py`) and reconstructs a canonical formula from validated tokens —
  patsy only ever sees names it can't turn into code.
- **Per-coefficient suppression.** A dummy for a categorical level with few
  members leaks those individuals. After fitting, we compute the support behind
  every term and blank any coefficient/CI/p-value with support `< min_n` (Cox
  drops sub-`min_n` dummy levels before fitting). Kaplan-Meier drops the tail
  where the at-risk set falls below `min_n` (that tail is individual event
  times). Only aggregate summaries are returned; `.predict`/`.resid`/per-subject
  curves are never exposed because no verb returns them.

## ProtectionLevel

One ordered level per source (`public < protected < sensitive`), resolved
most-restrictive-wins into a single `Policy` (`policy.py`) that drives `min_n`,
rounding, logging, auth, and whether the sandbox is permitted. Same shape as the
safestat spec, so it collapses onto m2py's `resolve_policy` when integrated.

## Relationship to the existing repos

| Repo | Role | safepy uses it as |
|---|---|---|
| `protect` | SDC verbs (`suppress`, `protect`, `risk`, `profile`) | the disclosure-control engine — called, never reimplemented |
| `m2py` | microdata emulator, `m2py_runtime` (pandas+polars ops), `py2m`/`r2m` frontends | the host it folds into; `safepy` becomes the `language="python"` frontend |
| `microdata-api` | Anvil server, tiered validator, `/run_extended` submit-poll | the deployment target; `run()` is the synchronous core of a future `/run_extended` |

## Backend notes

- **polars** — the adapter interface is backend-neutral. Polars is *lazy*, so the
  query plan is introspectable before execution and `.collect()` is the single
  materialisation point — a natural fit; it likely becomes the reference backend
  for a richer Tier-2. Deny-verbs there: `head`, `tail`, `glimpse`, `row`,
  `rows`, `item`, `get_column`, `to_numpy`, `sample`, `write_*`.
- **pandas 3 / Arrow** — copy-on-write becomes the default, which makes our
  "never mutate input" contract the engine default and `suppress`'s `.copy()`
  cheap. The mediator never inspects values to decide release, so the Arrow
  string dtype doesn't change its behaviour.

## Roadmap

1. **(done)** OPEN vertical slice: gate + runtime + mediator + pandas safe verbs
   + policy + red-team suite (`tests/attacks/`).
2. **(done)** STRICT profile: `SafeFrame` capability facade + profile selection.
   **(done, phase 1)** pandas-shaped chaining — `SafeColumn`, `df[mask]`,
   `groupby(by)[col].agg()`, column reducers.
   **(done, phase 2)** attribute access (`df.salary`, `df.groupby('sex').salary`),
   look-alike `pd`/`np` namespaces (`np.log`/`np.where`, `pd.cut`/`pd.crosstab`/
   `pd.to_datetime`, in `namespaces.py`), and natural `df.assign(col=np.log(df['x']))`.
   **(done, phase 3)** `smf.ols/logit/poisson("y ~ age + C(sex)", data=df).fit()
   .summary()` via our own whitelisted formula parser (`formula.py`,
   `formula_api.py`) — the user string is validated and reconstructed, never
   handed to patsy; per-coefficient suppression; `.predict`/`.resid` unreachable.
   **(done, phase 4)** coverage sweep (`tools/coverage_sweep.py`, 92% of the
   py2m idiom corpus) + safe wins: `np` constants, `pd.to_numeric`, `corr`,
   `rename`/`fillna`/`dropna`/`drop`, `a*b` formula interactions.
   **(done, phase 5)** plotting (`charts.py`): `.plot.bar()/.line()/...` on
   aggregate results only, `df['x'].hist()` redirected to a suppressed binned
   frequency; `render=spec|plotly|png|html|ascii` chosen at the API. Raw plotting
   (`df.plot()`, `.plot.scatter`, `.plot` on a raw column) is refused.
   **(done, phase 6)** order statistics: `min`/`max`/`quantile`/`describe`/
   `boxplot` on a numeric `SafeColumn`, under one rule — a value is releasable
   iff `min(#<=v, #>=v) >= min_n` (≥ min_n observations at/beyond it). Median &
   quartiles pass; extremes pass only if shared (rounded/coded/boolean data) or
   `winsorize=p`. Categorical extremes refused; boxplot omits outliers.
   Next: coverage of grouped extremes / more idioms as needed.

### Plotting model (phase 5)

A chart is a rendering of a `Released` (already-suppressed) aggregate, never of
raw data. `PlotAccessor` lives on `Released` (`value_counts().plot.bar()`) and
refuses anything that isn't an aggregated table; there is no `.plot` on a raw
`SafeFrame`, and a `SafeColumn`'s `.plot` allows only `.hist()`, which is
redirected to a suppressed binned frequency with round (non-min/max-revealing)
bin edges. Because the data is already suppressed, plotly/matplotlib embedding
raw arrays is a non-issue — the only array at plot time is the aggregate. The
chart *spec* (type + suppressed data) is the security boundary; `render_chart`
encodes it to the transport the caller asked for.
3. **(done)** statsmodels (`ols`/`logit`/`poisson`) + lifelines (`cox`/
   `kaplan_meier`) safe verbs, with per-coefficient / at-risk suppression and no
   user formula strings.
4. polars adapter + safe verbs (lazy-plan introspection); the STRICT `SafeFrame`
   could wrap a polars `LazyFrame` for free query-plan auditing.
5. viz adapter — inspect the figure's **backing arrays**, not the picture
   (plotly embeds raw arrays in its JSON; scatter = full data; box = extremes).
6. Fold into m2py as the `language="python"` frontend; wire `/run_extended`.

### Known gaps / next steps (honest list)

- `protect`'s richer rules (dominance, p%, secondary suppression) aren't wired
  into the safe verbs yet — only `min_n` + rounding. The hooks exist.
- `SafeFrame.where` supports a single column-vs-literal comparison; no compound
  predicates yet (compose with multiple `.where` calls).
- No viz verbs yet — plotting is the highest-risk surface and is deferred.
- OLS/GLM term support for interaction/transform terms defaults to full `n`
  (only main-effect categorical levels are individually suppressed).

## Causal inference

Much of it is coefficient-table output, which slots into the safe-regression
pattern:

- **Difference-in-differences / event study** — already expressible as
  `smf.ols('y ~ treat*post')`; no new library.
- **RDD (local linear)** — filter to a bandwidth + `y ~ D + run`.
- **Panel fixed effects, two-way-FE DiD, IV (2SLS), clustered SEs** — via
  `pyfixest` (`stats.py`): `df.feols(y=, x=[], fe=[], cluster=)` and
  `df.iv(y=, x=[], endog=, instruments=[])`. The formula is built from validated
  column names (never a user string → no formulaic eval); categorical covariates
  are one-hot encoded with sub-`min_n` levels dropped; fixed effects are absorbed
  (never reported); results are coefficient tables with per-term suppression.
- **Average treatment effect (DoWhy)** — `df.ate(outcome=, treatment=,
  confounders=[], method='weighting'|'matching'|'stratification'|'regression')`.
  A *curated verb*, not a library-mimicking facade: DoWhy's raw `CausalModel`
  API is deliberately **not** importable, because its multi-step flow exposes
  per-unit internals (propensity scores, matched sets). We release only the
  aggregate effect + CI + SE + arm sizes; treatment must be binary with each arm
  >= min_n; the input frame is copied so DoWhy never mutates it. (Contrast with
  lifelines/pyfixest, which we *do* expose in native syntax because their fitted
  objects are aggregate-shaped.)
- **Deferred:** *synthetic control* (donor weights can be disclosive); exposing
  propensity scores as a private `SafeColumn` (easy follow-on to `ate`).

## Non-goals (handled elsewhere)

Data-side pre-processing and the `risk` k-anonymity gate live in `protect`.
Timing side channels are out of scope for v1.
