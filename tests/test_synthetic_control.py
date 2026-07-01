"""Synthetic control: weights/synthetic are private intermediates; only the
aggregate effect path + ATT + diagnostics are released, with an exit guard
against a concentrated synthetic."""

import numpy as np
import pandas as pd
import pytest

from safepy import run
from safepy.policy import Profile

pytest.importorskip("pysyncon")


def _panel(effect=3.0, concentrated=False, size=100, seed=0):
    rng = np.random.default_rng(seed)
    units = [f"c{i}" for i in range(10)] + ["T"]
    times = list(range(1, 21))
    ttime = 15
    bases = {u: rng.normal(0, 1) for u in units}
    if concentrated:
        bases["c0"] = bases["T"]        # one donor perfectly matches -> weight ~1
    rows = []
    for u in units:
        for t in times:
            eff = effect if (u == "T" and t >= ttime) else 0.0
            noise = 0.02 if concentrated else 0.3
            y = bases[u] + 0.5 * t + eff + rng.normal(0, noise)
            rows.append({"unit": u, "time": t, "y": y, "size": size})
    return pd.DataFrame(rows)


def _run(df, code):
    return run(code, {"df": df}, profile=Profile.STRICT)


def test_sc_recovers_att_and_releases_effect_path():
    df = _panel()
    r = _run(df, "df.synthetic_control(unit='unit', time='time', outcome='y', "
                 "treated_unit='T', treatment_time=15, unit_size='size')")
    assert r.ok and r.payload["type"] == "synthetic_control"
    assert 2.0 < r.payload["att"] < 4.0                    # true effect ~3
    assert len(r.payload["effect_path"]["gap"]) == 20
    assert r.payload["rmspe_pre"] is not None


def test_weights_never_in_payload():
    df = _panel()
    r = _run(df, "df.synthetic_control(unit='unit', time='time', outcome='y', "
                 "treated_unit='T', treatment_time=15, unit_size='size')")
    blob = str(r.payload)
    assert "weight" not in blob.replace("max_weight", "").replace("effective", "")
    # (max_weight / effective_donors are aggregates OF the weights, not the weights)
    assert "c0" not in blob and "c1" not in blob            # no per-donor values


def test_concentrated_synthetic_refused_without_unit_size():
    df = _panel(concentrated=True)
    r = _run(df, "df.synthetic_control(unit='unit', time='time', outcome='y', "
                 "treated_unit='T', treatment_time=15)")
    assert r.ok is False and "concentrated" in r.error["message"]


def test_concentrated_ok_with_aggregate_units():
    # same concentrated fit, but units certified as aggregates of >= min_n
    df = _panel(concentrated=True, size=100)
    r = _run(df, "df.synthetic_control(unit='unit', time='time', outcome='y', "
                 "treated_unit='T', treatment_time=15, unit_size='size')")
    assert r.ok


def test_small_units_refused_with_unit_size():
    df = _panel(size=3)                                     # 3 individuals per unit < min_n
    r = _run(df, "df.synthetic_control(unit='unit', time='time', outcome='y', "
                 "treated_unit='T', treatment_time=15, unit_size='size')")
    assert r.ok is False and "min_n" in r.error["message"]


def test_too_few_pre_periods_refused():
    df = _panel()
    r = _run(df, "df.synthetic_control(unit='unit', time='time', outcome='y', "
                 "treated_unit='T', treatment_time=2, unit_size='size')")
    assert r.ok is False


def test_raw_pysyncon_not_importable():
    df = _panel()
    r = _run(df, "from pysyncon import Synth\nSynth()")
    assert r.ok is False
