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

## Status (first slice landed, 2026-07-02)

`df |> [filter(x OP v) |>] group_by(g) |> summarise(m = fn(x))` and
`df |> count(g)` — equivalence-tested against the pandas dialect across
mean/sum/median/sd/var, with suppression (region `Z`, n=2) and a red-team suite
(`tests/test_strict_r.py`, 23 tests) covering extremes, disclosive/unknown verbs,
raw column access, unknown column/source, and code-execution attempts — all refused.

## Next

- Multi-stat `summarise` (→ compound frame), whole-frame `summarise` (no group).
- `mutate` (derived columns), `select`/`rename`/`recode`, `distinct`.
- Base-R idioms: `aggregate(y ~ g, data=df, FUN=mean)`, `table(df$x)`,
  `df[df$x >= v, ]`.
- Robustness upgrade: swap the hand-rolled parser for an **Rscript parse-only**
  path (`getParseData(parse(text=))`, never evaluated) to cover the full R
  grammar — R is installed; parsing does not execute.
- Models/plots via the same delegation the polars dialect uses.
