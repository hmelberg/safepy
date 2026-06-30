"""Positive cases: legitimate aggregate analysis is allowed AND suppressed."""

from safepython import run, ProtectionLevel
from tests.fixtures import salaries

DF = salaries()


def _run(code, level=ProtectionLevel.PROTECTED):
    return run(code, {"df": DF}, level)


def test_safe_group_agg_releases_and_suppresses():
    # mean salary by sex, via the curated safe verb (computes paired counts).
    r = _run("safe.group_agg(df, 'sex', 'salary', 'mean')")
    assert r.ok and r.kind == "table"
    assert r.audit["verb"] == "group_agg"
    assert r.audit["cells_suppressed"] == 0  # F and M are both large groups


def test_small_group_is_suppressed():
    # region 'Z' has only 2 people (< min_n=5) -> its cell must be blanked.
    r = _run("safe.group_agg(df, 'region', 'salary', 'mean', min_n=5)")
    assert r.ok
    values = dict(zip(r.payload["index"], r.payload["values"]))
    assert values["Z"] is None          # suppressed
    assert values["A"] is not None      # released


def test_frequency_table_is_suppressed():
    r = _run("safe.value_counts(df, 'region')")
    assert r.ok and r.kind == "table"
    values = dict(zip(r.payload["index"], r.payload["values"]))
    assert values["Z"] is None          # count of 2 < min_n -> blanked


def test_public_level_allows_finer_detail():
    # under 'public', min_n=1, so the small region survives.
    r = run("safe.value_counts(df, 'region')", {"df": DF}, ProtectionLevel.PUBLIC)
    assert r.ok
    values = dict(zip(r.payload["index"], r.payload["values"]))
    assert values["Z"] == 2


def test_raw_pandas_result_is_refused():
    # raw pandas has no provenance -> default-denied, even for a count table.
    r = _run("df['region'].value_counts()")
    assert r.ok is False
    assert "safepython.safe" in r.error["message"]


def test_sensitive_level_uses_strict_profile():
    # SENSITIVE runs in the STRICT capability profile: safe verbs work, but the
    # source is a SafeFrame and raw pandas is not in scope.
    r = run("safe.group_agg(df, 'sex', 'salary', 'mean')",
            {"df": DF}, ProtectionLevel.SENSITIVE)
    assert r.ok and r.audit["profile"] == "strict"

    # raw pandas is unavailable under STRICT (no `pd`, source is a SafeFrame)
    r2 = run("df['region'].value_counts()", {"df": DF}, ProtectionLevel.SENSITIVE)
    assert r2.ok is False
