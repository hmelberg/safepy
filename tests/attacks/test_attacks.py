"""The executable threat model.

Every test here is a known leak vector. A passing test means the vector is
*blocked* — either rejected by the AST gate (preferred, cheap) or refused by the
output mediator (the statistical backstop). When you think of a new attack, add
it here first; it should fail, then make it pass.
"""

import pytest

from safepython import run, ProtectionLevel
from tests.fixtures import salaries

DF = salaries()


def _run(code, level=ProtectionLevel.PROTECTED):
    return run(code, {"df": DF}, level)


def _blocked(code, level=ProtectionLevel.PROTECTED):
    r = _run(code, level)
    assert r.ok is False, f"expected BLOCKED but released: {code!r} -> {r.payload!r}"
    return r


# ---- direct row dumps -------------------------------------------------------

@pytest.mark.parametrize("code", [
    "df.head()",
    "df.tail(10)",
    "df.sample(5)",
    "df.iloc[0]",
    "df.loc[0]",
    "df.values",
    "df.to_numpy()",
    "df.to_csv()",
    "df.to_dict()",
    "df['salary'].tolist()",
    "df.itertuples()",
    "df[0]",            # positional indexing
    "df['salary'][0]",  # positional indexing on a series
])
def test_row_dumps_blocked(code):
    r = _blocked(code)
    assert r.error["kind"] in {"validation", "ValidationError", "syntax",
                               "attribute", "subscript", "call", "name"}


# ---- extremes / positional reducers return individual values ----------------

@pytest.mark.parametrize("code", [
    "df['salary'].max()",
    "df['salary'].min()",
    "df['salary'].idxmax()",
    "df.nlargest(1, 'salary')",
    "df['salary'].describe()",
    "df.groupby('sex')['salary'].first()",
    "df['salary'].quantile(0.99)",
])
def test_extremes_blocked(code):
    _blocked(code)


# ---- code-execution escapes -------------------------------------------------

@pytest.mark.parametrize("code", [
    "eval('1+1')",
    "exec('x=1')",
    "__import__('os').system('echo hi')",
    "getattr(df, 'head')()",
    "df.__class__.__bases__[0].__subclasses__()",
    "open('/etc/passwd').read()",
    "df.apply(lambda r: r['salary'], axis=1)",
    "df['salary'].map(lambda x: x)",
    "df.query('salary > 100000')",
    "df.eval('salary * 2')",
    "df.pipe(print)",
])
def test_escapes_blocked(code):
    _blocked(code)


# ---- structural rules -------------------------------------------------------

@pytest.mark.parametrize("code", [
    "for i in range(3):\n    df\n",
    "while True:\n    df\n",
    "def f():\n    return df\n",
    "import os\nos",
    "lambda: df",
    "[r for r in df]",
    "x = df\nprint(x)",     # print is an unknown bare-name call
    "df; df",               # non-final bare expression
])
def test_structure_blocked(code):
    _blocked(code)


# ---- statistical disclosure caught by the mediator (not the gate) -----------

def test_bare_scalar_refused():
    # mean() passes the gate (it's a legitimate verb) but a lone scalar has no
    # provenance the mediator can verify, so it is refused.
    _blocked("df['salary'].mean()")


def test_ungrouped_mean_table_refused():
    # a means table without paired counts cannot be small-cell suppressed.
    _blocked("df.groupby('region')['salary'].mean()")
