# safepython — design

## Goal

Let an analyst write a *familiar subset of Python* (pandas now; polars,
statsmodels, lifelines, matplotlib, plotly later) against private,
individual-level data and get back **aggregate results only** — never a row, a
scalar extreme, or any view of an individual. Like microdata.no, but the input
language is restricted Python rather than a bespoke DSL.

## Posture (decided)

- **Sandbox, not translator.** The server runs the user's AST-gated Python
  directly. This is the path the m2py *safestat* spec (2026-06-29) deliberately
  rejected for sensitive data, so safepython is scoped to **public/local** data
  and treated as a research track. `protected` keeps the sandbox enabled as a
  deliberate research configuration; `sensitive` forbids it (use the
  translate-to-artifact frontend there).
- **Standalone now, fold into m2py later.** Built against a thin adapter
  interface so it slots in beside `py2m`/`r2m` as a Python *frontend*.
- **Reuse, don't reimplement.** All data-side and result-side disclosure
  control is the existing [`protect`](../protect) package. safepython only owns
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
   secondary suppression. Already built; safepython calls it.

## The load-bearing boundary: provenance

The mediator **cannot decide disclosure from a raw result's values.** A table of
means that happen to be integers is indistinguishable from a table of counts; a
scalar mean is indistinguishable from a scalar max. (We learned this the honest
way — an early value-sniffing heuristic released a means table.) Therefore:

- **Raw pandas results are default-denied.** Compute freely for intermediate
  steps, but the *released* value must carry provenance.
- **`safepython.safe` verbs are the trusted release path.** Each computes the
  aggregate *together with its group counts*, runs `protect.suppress`, and
  returns a `Released` value the mediator trusts. `safe` is **policy-bound**:
  `min_n` defaults to the policy floor and callers may only make it stricter.

This boundary is the seed of a **phase-2 SafeFrame facade**: a capability proxy
whose every verb records how it aggregated, so a richer pandas-like surface can
release results while keeping provenance.

## ProtectionLevel

One ordered level per source (`public < protected < sensitive`), resolved
most-restrictive-wins into a single `Policy` (`policy.py`) that drives `min_n`,
rounding, logging, auth, and whether the sandbox is permitted. Same shape as the
safestat spec, so it collapses onto m2py's `resolve_policy` when integrated.

## Relationship to the existing repos

| Repo | Role | safepython uses it as |
|---|---|---|
| `protect` | SDC verbs (`suppress`, `protect`, `risk`, `profile`) | the disclosure-control engine — called, never reimplemented |
| `m2py` | microdata emulator, `m2py_runtime` (pandas+polars ops), `py2m`/`r2m` frontends | the host it folds into; `safepython` becomes the `language="python"` frontend |
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

1. **(done)** Vertical slice: gate + runtime + mediator + pandas safe verbs +
   policy + red-team suite (`tests/attacks/`).
2. polars adapter + safe verbs (lazy-plan introspection).
3. statsmodels / lifelines adapters (release `.summary()`; deny per-observation
   attributes like `.predict`, `.resid`, `.fittedvalues`, per-subject curves).
4. viz adapter — inspect the figure's **backing arrays**, not the picture
   (plotly embeds raw arrays in its JSON; scatter = full data; box = extremes).
5. Phase-2 SafeFrame facade for a richer, provenance-carrying surface.
6. Fold into m2py as the `language="python"` frontend; wire `/run_extended`.

## Non-goals (handled elsewhere)

Data-side pre-processing and the `risk` k-anonymity gate live in `protect`.
Timing side channels are out of scope for v1.
