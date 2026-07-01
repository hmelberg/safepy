# Plan: R support in STRICT mode

**Decision: R is a *translated* safe surface, never executed.** R is a different
language, so it cannot ride the Python AST gate. Instead `safepy/r_api.py` **parses**
a restricted dplyr/base-R surface and **translates** it to the same
backend-neutral release core (`SafeVerbs`) the pandas and polars dialects use.
User R is never `eval`/`source`-ed — there is no system/file/code-execution
surface — so the dialect is **safe by construction**, needs no R at runtime, and
matches DESIGN's "sensitive / translate-to-artifact" posture and the `m2py/r2m`
prior art (an R→microdata translator).

## Architecture

- `run(code, sources, dialect="r")` bypasses the Python gate/runtime (`api._run_r`)
  and calls `r_api.translate_r`, which returns a suppressed `Released` mediated
  exactly like the other dialects.
- The parser is **default-deny**: only whitelisted verbs (`group_by`, `summarise`/
  `summarize`, `count`, `filter`) and aggregation functions
  (`mean`/`sum`/`median`/`sd`→std/`var`/`n`→size) are recognised; anything else —
  including extremes (`max`/`min`/`quantile`), column extraction (`pull`, `df$x`),
  row ops (`head`/`slice`/`arrange`), and code execution (`system(...)`) — is
  refused. A sanitising catch-all in `_run_r` ensures no raw exception carries a
  data value out.
- Terminal `summarise`/`count` route to `SafeVerbs.group_agg` / `value_counts`;
  `filter` produces a private intermediate frame that exits only via the terminal
  summary (like the polars `filter`). Both `|>` and `%>%` pipes are accepted.

## Status (2026-07-02)

`df |> [filter(x OP v) |>] group_by(g) |> summarise(m = fn(x))`, `df |> count(g)`,
and base-R `lm(y ~ x, data=df)` / `glm(y ~ x, family=binomial|poisson, data=df)`
→ the shared `ols`/`logit`/`poisson` verbs. Equivalence-tested against the pandas
dialect (aggregations + `lm`≡`ols`), with suppression (region `Z`, n=2) and a
red-team suite (`tests/test_strict_r.py`, 26 tests) — extremes, disclosive/unknown
verbs, raw column access, unknown column/source, and code-execution attempts all
refused.

## Direction: translate to the shared facade (reuse everything) — "option 1"

Anvil (hosted) runs Python only — **no R, no subprocess, no Node** — so on the
current deployment the translate approach is the *only* one available, and R is
never executed. Rather than hand-map each R verb to `SafeVerbs`, the translator
targets the **pandas STRICT facade** (`SafeFrame` + the safe `pd`/`np`
namespaces + the model/plot verbs). Everything the Python dialect can already do
— models, plots, `ate`, native compute, per-coefficient suppression, the whole
audited surface — becomes reachable from R for the cost of a *mapping*, not a
reimplementation. Proof landed: `lm(salary ~ pid, data=df)` returns byte-for-byte
the same suppressed coefficient table as `df.ols(y='salary', x=['pid'])`.

## Analysis: is it worth it, and will it be good enough?

**Verdict: yes, for a curated, well-documented subset — with tidyverse/dplyr as
the primary target.** The common data-science loop (manipulate → group →
summarise → model → plot) is *verb-shaped*, and dplyr is essentially the same
grammar-of-data-manipulation as the pandas facade, so the mapping is natural and
covers most real analysis. It will **not** be a general R interpreter; it is a
capability-facade dialect, exactly like STRICT pandas/polars.

**The crux is the parser, not the safety.** Safety is free (translate ⇒ only
suppressed aggregates exit; no execution surface). The limiting factor is how
much R we can *parse* reliably. With R installed we'd use `Rscript` parse-only
(`getParseData(parse(text=))`, never evaluated) and cover the full grammar — but
hosted Anvil has no R, so we must **hand-roll a parser**. The current
regex-per-verb approach is fine for `group_by`/`summarise`/`count`/`filter`/`lm`
but will not scale to `mutate(k = log(x) + y*2)`, `case_when`, joins, or formulas.
**The enabling investment is a small proper R expression parser** (tokenizer +
recursive-descent / Pratt) that evaluates R expressions against the `SafeColumn`
algebra the facade already exposes. Everything else is verb dispatch on top.

### Coverage assessment (what maps, how cleanly)

**tidyverse (dplyr/tidyr) — strong; the primary target.** Near 1:1 to the facade:

| R (dplyr) | facade | notes |
|---|---|---|
| `filter(x OP v)` | `df[mask]` | ✓ (done, single-predicate) |
| `group_by()+summarise(fn(x))` | `groupby().agg()` | ✓ (done, single stat) → multi-stat |
| `count(g)` | `value_counts` | ✓ (done) |
| `mutate(k = expr)` | `assign(k=expr)` | needs the expression parser |
| `select(a, b)` / `select(-a)` | `df[[...]]` / `drop` | easy |
| `rename(new = old)` | `rename` | easy |
| `arrange(x)` / `desc(x)` | `sort_values` | easy (shaping) |
| `distinct()` | `drop_duplicates` | easy |
| `transmute()` | `assign`+`select` | easy |
| `case_when()` / `if_else()` / `recode()` | `np.where`/`where`/`replace` | needs expr parser |
| `across(cols, fn)` | multi-col agg | medium |
| `left_join()` etc. | `merge` | medium (key inference) |
| `pivot_longer/wider()` | `melt`/`pivot_table` | medium |
| `slice_max/min/head/tail`, `pull`, `top_n` | — | **refused** (extremes / row extraction) |

**base R — medium; cover the analysis idioms, refuse the row-poking.** Maps:
`aggregate(y ~ g, data=df, FUN=mean)` → groupby; `table(x)` / `table(x, y)` →
value_counts / crosstab; `tapply(x, g, mean)` → groupby; `df[df$x >= v, ]` →
filter; `mean(df$x)` / `colMeans`/`sapply(df, mean)` → reducers; `lm`/`glm`/`aov`
/`cor`/`quantile`/`summary` → the model/stat verbs (`lm`/`glm` ✓ done). Refuse
the positional/row idioms (`df[1, ]`, `df$x[1]`, `head`). base R is syntactically
diverse (`$`, `[[`, `<-`, formulas), so it costs more parser work per idiom than
dplyr — hence *secondary*.

**data.table — core pattern only.** `dt[i, .(m = mean(x)), by = g]` is one
powerful construct (i=filter, j=aggregate, by=group) that maps to
filter→groupby→summarise. Worth supporting that shape. The advanced surface
(`:=` in-place update, `.SD`/`.N`/`.SDcols`, `set()`, `dt[...][...]` chaining) is
high-effort / low-return and should be **out of scope** — document it as such.

### Honest limits

- **No arbitrary R**: custom functions, `for`/`apply` with closures, arbitrary
  packages, metaprogramming — out of scope by design (curated subset).
- **NSE / scoping**: dplyr treats bare names as columns; we need a simple
  column-vs-let-bound-name model (small, but real).
- **Statements & intermediates**: base-R scripts assign intermediate frames
  (`x <- df |> ...; x |> ...`). Needs a light multi-statement model (the current
  translator is single-pipeline).
- **Parser ceiling**: the hand-rolled parser caps the long tail; the Rscript
  parse-only upgrade removes that ceiling *if/when* a VM with R exists.

## Phased plan

1. **(done)** First slice + `lm`/`glm` → facade; retarget to the STRICT facade.
2. **Expression parser** (the enabling investment): tokenizer + recursive-descent
   for R expressions → evaluate against `SafeColumn`. Unlocks `mutate`, `filter`
   with compound predicates, `case_when`/`if_else`, `transmute`.
3. **Tidyverse breadth**: `select`/`rename`/`arrange`/`distinct`, multi-stat
   `summarise`, `across`, `mutate`; then `left_join`/`pivot_*`.
4. **base-R analysis idioms**: `aggregate`, `table`, `tapply`, `cor`, `summary`,
   `df[cond, ]`, `mean(df$x)`; multi-statement scripts with `<-`.
5. **data.table** core `dt[i, j, by]`.
6. **Extras** (as with pandas/polars): plots (`ggplot`/`hist`/`boxplot` → chart),
   survival (`coxph`/`survfit`), `.ate`-style curated verbs exposed under R names.
7. **Later, if a VM with R exists**: swap in the Rscript parse-only front end for
   full-grammar coverage (still never evaluated).
