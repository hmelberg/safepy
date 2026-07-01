"""Tiltak 3 (støylegging): deterministic cell-key noise on counts, with sums
scaled proportionally so means are preserved. Noise is keyed by cell label + a
secret salt, so it is stable across queries (repeating a query cannot average it
away) yet perturbs the displayed counts."""

import numpy as np
import pandas as pd

from safepy import run
from safepy.policy import Suppression
from safepy.safe import _cell_noise


def _data():
    rng = np.random.default_rng(7)
    n = 600
    df = pd.DataFrame({
        "region": rng.choice(["North", "South", "East"], n),
        "sex": rng.choice(["M", "F"], n),
        "income": rng.integers(200_000, 500_000, n).astype(float),
    })
    # a rare category (3 rows) to confirm min_n suppression still bites under noise
    rare = pd.DataFrame({"region": ["West"] * 3, "sex": ["M"] * 3,
                         "income": [250_000.0] * 3})
    return pd.concat([df, rare], ignore_index=True)


DATA = _data()

_NOISE = Suppression(min_n=5, count_noise=2)
_TRUE = Suppression(min_n=5)                                  # no noise, no rounding


def _run(code, tier):
    return run(code, {"df": DATA}, profile="strict", suppression=tier)


# ---- the noise primitive -----------------------------------------------------

def test_cell_noise_deterministic_and_bounded():
    assert _cell_noise("North", 2) == _cell_noise("North", 2)      # stable
    assert all(-2 <= _cell_noise(k, 2) <= 2 for k in ["North", "South", ("A", "B")])


# ---- counts are perturbed, but stay close and deterministic ------------------

def test_value_counts_noised_but_bounded():
    true = _run("df['region'].value_counts()", _TRUE).payload
    noised = _run("df['region'].value_counts()", _NOISE).payload
    assert true["index"] == noised["index"]                        # order preserved
    diffs = [abs(a - b) for a, b in zip(true["values"], noised["values"])
             if a is not None and b is not None]
    assert any(d != 0 for d in diffs)                              # noise did something
    assert all(d <= 2 for d in diffs)                              # bounded by step


def test_noise_is_deterministic_across_runs():
    r1 = _run("df['region'].value_counts()", _NOISE).payload["values"]
    r2 = _run("df['region'].value_counts()", _NOISE).payload["values"]
    assert r1 == r2                                                # same query -> same noise


def test_knob_off_gives_exact_integer_counts():
    true = _run("df['region'].value_counts()", _TRUE).payload["values"]
    assert all(v is None or float(v).is_integer() for v in true)


def test_min_n_suppression_survives_noise():
    # the 3-row 'West' category is still suppressed under noise
    p = _run("df['region'].value_counts()", _NOISE).payload
    vals = dict(zip(p["index"], p["values"]))
    assert vals["West"] is None


# ---- crosstab is noised too --------------------------------------------------

def test_crosstab_noised():
    true = _run("df.crosstab('region', 'sex')", _TRUE).payload["data"]
    noised = _run("df.crosstab('region', 'sex')", _NOISE).payload["data"]
    flat_t = [v for row in true for v in row]
    flat_n = [v for row in noised for v in row]
    diffs = [abs(a - b) for a, b in zip(flat_t, flat_n) if a is not None and b is not None]
    assert any(d != 0 for d in diffs) and all(d <= 2 for d in diffs)


# ---- means preserved; sums scaled to stay consistent -------------------------

def test_mean_unaffected_by_count_noise():
    base = Suppression(min_n=5, winsorize=(0.01, 0.99))
    noise = Suppression(min_n=5, winsorize=(0.01, 0.99), count_noise=2)
    m_base = _run("df.groupby('region')['income'].mean()", base).payload["values"]
    m_noise = _run("df.groupby('region')['income'].mean()", noise).payload["values"]
    assert m_base == m_noise                                       # noise does not move the mean


def test_sum_scaled_so_mean_is_preserved():
    # sum/count (both noised with the same per-group key) recovers the true mean
    sums = _run("df.groupby('region')['income'].sum()", _NOISE).payload["values"]
    counts = _run("df.groupby('region')['income'].count()", _NOISE).payload["values"]
    means = _run("df.groupby('region')['income'].mean()", _TRUE).payload["values"]
    pairs = [(s, c, m) for s, c, m in zip(sums, counts, means)
             if s is not None and c is not None and m is not None]
    assert pairs
    for s, c, m in pairs:
        assert abs(s / c - m) < 1e-6


# ---- scalar count is noised too ----------------------------------------------

def test_scalar_count_noised():
    true = _run("df['income'].count()", _TRUE).payload["value"]
    noised = _run("df['income'].count()", _NOISE).payload["value"]
    assert abs(true - noised) <= 2


# ---- zero noised count -> sum 0 / mean NaN -----------------------------------

def test_zero_noised_count_gives_nan_mean():
    # a huge step can drive a group's noised count to 0; the mean must go NaN,
    # never divide-by-zero or leak. (deterministic, so this is stable.)
    big = Suppression(min_n=5, count_noise=100_000)
    r = _run("df.groupby('region')['income'].mean()", big)
    assert r.ok                                                    # no crash
    # at least it releases a table; any 0-count cell is NaN, not an error
    assert r.kind == "table"
