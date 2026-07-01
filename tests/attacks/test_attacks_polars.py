"""Red-team for the polars STRICT dialect.

Every test is a known leak vector for the polars *surface*. A passing test means
the vector is blocked — by the AST gate (preferred), the SafeExpr/SafePolarsFrame
facade (default-deny), or the output mediator. The facade is the primary boundary
in STRICT: disclosive methods should simply not exist.
"""

import polars as pl
import pytest

from safepy import run
from safepy.policy import Profile
from tests.fixtures import salaries

PL_DF = pl.from_pandas(salaries())


def _run(code):
    return run(code, {"df": PL_DF}, profile=Profile.STRICT, dialect="polars")


def _blocked(code):
    r = _run(code)
    assert r.ok is False, f"expected BLOCKED but released: {code!r} -> {r.payload!r}"
    return r


# ---- direct row dumps / raw exports on the frame ----------------------------

@pytest.mark.parametrize("code", [
    "df.head()",
    "df.tail(10)",
    "df.sample(5)",
    "df.get_column('salary')",
    "df.get_columns()",
    "df.to_pandas()",
    "df.to_numpy()",
    "df.to_dict()",
    "df.to_dicts()",
    "df.to_arrow()",
    "df.rows()",
    "df.row(0)",
    "df.item(0, 0)",
    "df.iter_rows()",
    "df.map_rows(len)",
    "df.partition_by('sex')",
    "df['salary']",              # SafePolarsFrame has no __getitem__
    "df[0]",                     # positional indexing
])
def test_frame_row_dumps_blocked(code):
    _blocked(code)


# ---- expression-level extremes / order stats / raw exports ------------------

@pytest.mark.parametrize("expr", [
    "pl.col('salary').max()",
    "pl.col('salary').min()",
    "pl.col('salary').arg_max()",
    "pl.col('salary').sort()",
    "pl.col('salary').top_k(3)",
    "pl.col('salary').bottom_k(3)",
    "pl.col('salary').gather(0)",
    "pl.col('salary').first()",
    "pl.col('salary').last()",
    "pl.col('salary').slice(0, 5)",
    "pl.col('salary').implode()",
    "pl.col('salary').quantile(0.99)",
    "pl.col('salary').to_list()",
    "pl.col('salary').to_physical()",
    "pl.col('salary').map_elements(len)",
    "pl.col('salary').map_batches(len)",
    "pl.col('salary').over('sex')",
])
def test_expression_disclosive_blocked(expr):
    # in a select (would materialize) and in a filter (would subset by value)
    _blocked(f"import polars as pl\ndf.select({expr})")


# ---- aggregate-of-all-rows (list-ification) ---------------------------------

@pytest.mark.parametrize("code", [
    "import polars as pl\ndf.group_by('sex').agg(pl.col('salary'))",            # no reducer -> list per group
    "import polars as pl\ndf.group_by('sex').agg(pl.col('salary').implode())",  # explicit list
    "import polars as pl\ndf.select(pl.col('salary'))",                          # dangling column frame
    "import polars as pl\ndf.filter(pl.col('salary') == pl.col('salary').max())",  # isolate the max row
])
def test_aggregate_all_rows_blocked(code):
    _blocked(code)


# ---- code-execution escapes (gate) ------------------------------------------

@pytest.mark.parametrize("code", [
    "eval('1+1')",
    "__import__('os').system('echo hi')",
    "df.__class__.__bases__[0].__subclasses__()",
    "import polars as pl\ndf.select(pl.col('salary').map_elements(lambda x: x))",  # lambda -> gate
    "[r for r in df]",
    "import os\nos",
])
def test_code_escapes_blocked(code):
    _blocked(code)


# ---- dangling intermediates -------------------------------------------------

@pytest.mark.parametrize("code", [
    "import polars as pl\ndf.filter(pl.col('salary') >= 0)",
    "import polars as pl\ndf.with_columns(pl.col('salary') * 2)",
    "import polars as pl\ndf.group_by('sex')",
    "import polars as pl\npl.col('salary')",
])
def test_dangling_intermediates_blocked(code):
    _blocked(code)
