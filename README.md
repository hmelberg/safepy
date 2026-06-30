# safepython

Run a familiar **subset of Python** against private, individual-level data and
get back **aggregate results only** — never a row, a scalar extreme, or any view
of an individual. Think microdata.no, but the input language is restricted
Python instead of a bespoke DSL.

Disclosure control itself is delegated to the [`protect`](../protect) package;
safepython owns the *language frontend* — the static gate, the sandbox, output
mediation, and a set of curated safe verbs. See [DESIGN.md](DESIGN.md).

> **Status:** runnable vertical slice (pandas). Research track, scoped to
> public/local data — see the posture note in DESIGN.md.

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
from safepython import run, ProtectionLevel
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
