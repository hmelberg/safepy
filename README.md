# safepy

Submit a script in a **familiar analysis language** against private,
individual-level data and get back **aggregate results only** — never a row, a
scalar extreme, or any view of an individual. Think microdata.no, but the input
is a restricted subset of real languages instead of a bespoke DSL.

Analysts write **pandas, polars, R, or SQL**; every dialect produces the same
kind of result — a suppressed aggregate — through **one shared, audited release
core**. Disclosure control itself is delegated to the [`protect`](../protect)
package; safepy owns the *language frontends* + the trusted release path. For the
full picture start at [docs/OVERVIEW.md](docs/OVERVIEW.md), then
[DESIGN.md](DESIGN.md) and the per-dialect [`docs/plan-*.md`](docs).

> **Status:** runnable; **580 tests passing** across four dialects (pandas,
> polars, R, DuckDB). Two profiles (OPEN sandbox / STRICT capability facade).

## Dialects — many languages, one audited release core

Selected via `run(code, sources, profile=STRICT, dialect=...)`:

| `dialect` | How it works |
|---|---|
| `"pandas"` (default) | `SafeFrame`/`SafeColumn` capability facade mirroring real pandas call shapes |
| `"polars"` | polars facade + **native polars compute**; suppression byte-identical to pandas |
| `"r"` | a restricted dplyr/base-R surface **translated, never executed**, mapped to the facade |
| `"duckdb"` | SQL **executed in a locked engine** (no file/network), AST-gated, released via the shared suppressor |

```python
run("df.groupby('sex')['salary'].mean()", {"df": df}, profile=STRICT)                      # pandas
run("df.group_by('sex').agg(pl.col('salary').mean())", {"df": pl_df}, profile=STRICT, dialect="polars")
run("df |> group_by(sex) |> summarise(m = mean(salary))", {"df": df}, profile=STRICT, dialect="r")
run("SELECT sex, avg(salary) FROM df GROUP BY sex", {"df": df}, profile=STRICT, dialect="duckdb")
```

## Two profiles, one engine

- **OPEN** (public/local) — real pandas + the raw frame are in scope; safety
  comes from the gate (deny-list) and the mediator (raw results refused unless
  produced through a `safe.*` verb). *Probably safe; audit surface is all of
  pandas.*
- **STRICT** (protected/sensitive) — only a `SafeFrame` facade and the safe-verb
  library are in scope; no pandas, no raw frame. `df.head()` doesn't exist.
  *Safe by construction; audit surface is the small SafeFrame method list.*

In STRICT mode `df` is a `SafeFrame` that mirrors pandas' real call shapes —
selection, boolean masks, `groupby(...)[col].agg()`, column reducers — while the
disclosive verbs simply don't exist:

```python
from safepy import run
from safepy.policy import Profile

S = dict(profile=Profile.STRICT)
run("df.groupby('sex')['salary'].mean()", {"df": df}, **S)              # suppressed table
run("df[df['age'] >= 40]['salary'].median()", {"df": df}, **S)         # suppressed scalar
run("df[(df['age'] >= 40) & (df['sex'] == 'F')].groupby('region')['salary'].mean()", {"df": df}, **S)
run("df['region'].value_counts()", {"df": df}, **S)
run('smf.ols("salary ~ age + C(sex)", data=df).fit().summary()', {"df": df}, **S)  # formula API
run("df.kaplan_meier(duration='dur', event='died', by='sex')", {"df": df}, **S)  # lifelines

run("df['salary'].max()", {"df": df}, **S)   # ok=False: extremes reveal individuals
run("df.head()", {"df": df}, **S)            # ok=False: not a SafeFrame method
```

## How it works

```
code ─▶ AST gate ─▶ restricted runtime ─▶ output mediator ─▶ protect.suppress ─▶ result
```

- **AST gate** — default-deny: simple assignments + one final expression; no
  loops/defs/lambdas/comprehensions/imports; dunder ban; row-dump, extreme, and
  callable-taking verbs blocked.
- **Mediator** — the only exit. Raw pandas results are refused (no provenance);
  only results produced through `safe.*` are released.
- **`safe.*` verbs** — compute an aggregate *with its group counts*, suppress
  small cells via `protect`, return a release-checked value. Bound to the
  protection policy (`min_n` can only get stricter).

## Example

```python
from safepy import run, ProtectionLevel
import pandas as pd

df = pd.read_parquet("salaries.parquet")   # private, individual-level

# allowed: aggregate with suppression
run("safe.group_agg(df, 'sex', 'salary', 'mean')", {"df": df}, ProtectionLevel.PROTECTED)
#  -> SafeResult(ok=True, kind='table', payload={...}, audit={'cells_suppressed': ...})

# blocked at the gate
run("df.head()",            {"df": df})   # ok=False: row dump
run("df['salary'].max()",   {"df": df})   # ok=False: returns an individual value
run("eval('1+1')",          {"df": df})   # ok=False: code-execution escape

# blocked at the mediator (no provenance)
run("df.groupby('sex')['salary'].mean()", {"df": df})  # ok=False: use safe.group_agg
```

## Develop

```bash
# uses the sibling ../protect repo without installing it (see conftest.py)
python -m pytest -q
```

`tests/attacks/` is the executable threat model — each test is a leak vector
that must stay blocked. Add new attacks there first.
