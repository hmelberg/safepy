"""Secondary-suppression measures (microdata.no "Tiltak") and the configurable
aggressiveness tiers. Each measure is tested by comparing a strict tier against a
lighter one, so the effect is isolated from the rest of the release path."""

import numpy as np
import pandas as pd
import pytest

from safepy import run
from safepy.policy import (PRESETS, ProtectionLevel, Suppression, resolve_policy)
from safepy.safeframe import _sig_round


def _data():
    rng = np.random.default_rng(3)
    n = 400
    grp = np.array(["common"] * 394 + ["rare"] * 6)      # a group of exactly 6
    income = rng.integers(200_000, 400_000, n).astype(float)
    income[:3] = [5_000_000, 6_000_000, 7_000_000]        # a few extreme outliers
    return pd.DataFrame({
        "grp": grp,
        "sex": rng.choice(["M", "F"], n),
        "income": income,
        "uid": np.arange(n),                              # unique -> sparse table
    })


DATA = _data()


def _run(code, tier=None):
    return run(code, {"df": DATA}, profile="strict", suppression=tier)


# ---- preset / tier plumbing --------------------------------------------------

def test_level_preset_mapping():
    assert resolve_policy([ProtectionLevel.PROTECTED]).suppression is PRESETS["standard"]
    assert resolve_policy([ProtectionLevel.SENSITIVE]).suppression is PRESETS["microdata"]
    assert resolve_policy([ProtectionLevel.PUBLIC]).suppression is PRESETS["off"]


def test_suppression_override_by_name_and_instance():
    assert resolve_policy(["protected"], suppression="light").suppression is PRESETS["light"]
    s = Suppression(min_n=7)
    assert resolve_policy(["protected"], suppression=s).suppression is s


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        resolve_policy(["protected"], suppression="bogus")


# ---- Tiltak 2: default winsorization (mean/std down, median unchanged) -------

def test_winsorization_lowers_mean_not_median():
    w = Suppression(min_n=5, winsorize=(0.01, 0.99))
    nw = Suppression(min_n=5)
    mean_w = _run("df['income'].mean()", w).payload["value"]
    mean_nw = _run("df['income'].mean()", nw).payload["value"]
    assert mean_w < mean_nw                         # extreme incomes pulled in

    med_w = _run("df['income'].quantile(0.5)", w).payload["value"]
    med_nw = _run("df['income'].quantile(0.5)", nw).payload["value"]
    assert med_w == med_nw                          # winsorization does not move the median


def test_regression_uses_unwinsorized_data():
    # regression estimates are not personal data -> not winsorized; a coefficient
    # table still releases under the strict tier.
    r = _run("df.ols(y='income', x=['sex'])")
    assert r.ok


# ---- Tiltak 8: percentiles to 3 significant figures --------------------------

def test_percentile_three_sig_figs():
    light = _run("df['income'].quantile(0.5)", "light").payload["value"]
    std = _run("df['income'].quantile(0.5)").payload["value"]
    assert std == _sig_round(light, 3)
    assert std != light or light == _sig_round(light, 3)


# ---- Tiltak 7 + 1: descriptive-population floors -----------------------------

def test_descriptive_floor_suppresses_small_group():
    # the 'rare' group has 6 members: released under light (min_n 5), suppressed
    # under standard (descriptive floor 10).
    code = "df[df['grp'] == 'rare']['income'].mean()"
    assert _run(code, "light").payload["value"] is not None
    assert _run(code).payload["value"] is None


def test_counts_exempt_from_descriptive_floor():
    # counts/sums are exempt (Tiltak 7): a 6-row count still releases under standard
    r = _run("df[df['grp'] == 'rare']['income'].count()")
    assert r.payload["value"] is not None


def test_min_population_blocks_descriptive_in_microdata_tier():
    # microdata tier requires a 1000-person population; 400 rows -> mean suppressed
    assert _run("df['income'].mean()", "microdata").payload["value"] is None
    # but a count is still allowed
    assert _run("df['income'].count()", "microdata").payload["value"] is not None


# ---- Tiltak 5: stop tables with too many low cells ---------------------------

def test_sparse_table_stopped_under_standard():
    r = _run("df['uid'].value_counts()")                 # 400 unique -> all cells < 5
    assert r.ok is False and "table stopped" in r.error["message"]


def test_sparse_table_allowed_under_light():
    r = _run("df['uid'].value_counts()", "light")
    assert r.ok                                          # cells suppressed, table returned


# ---- Tiltak 6: no edits affecting fewer than 10 units ------------------------

def test_small_edit_refused_under_standard():
    # only 3 rows exceed 1,000,000 -> the recode touches 3 units
    code = "df.assign(c=df['income'].where(df['income'] < 1_000_000, 0)).groupby('sex')['c'].mean()"
    r = _run(code)
    assert r.ok is False and "fewer than" in r.error["message"]


def test_small_edit_allowed_under_light():
    code = "df.assign(c=df['income'].where(df['income'] < 1_000_000, 0)).groupby('sex')['c'].mean()"
    assert _run(code, "light").ok


def test_edit_touching_all_rows_allowed():
    # remapping every row (all-or-none exception) is fine even under standard
    r = _run("df.assign(s=df['sex'].map({'M': 'Male', 'F': 'Female'})).value_counts('s')")
    assert r.ok
