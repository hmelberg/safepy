"""Library-faithful pyfixest usage: feols("Y ~ X | f").summary() and pf.feols(...)."""

import numpy as np
import pandas as pd
import pytest

from safepy import run
from safepy.policy import Profile

pytest.importorskip("pyfixest")


def _panel(n=800, seed=0):
    rng = np.random.default_rng(seed)
    firm = rng.integers(0, 25, n)
    year = rng.integers(2000, 2006, n)
    x = rng.normal(0, 1, n)
    z = rng.normal(0, 1, n)
    endog = 0.6 * z + rng.normal(0, 1, n)
    y = 2 * x + 1.5 * endog + 0.1 * firm + rng.normal(0, 1, n)
    return pd.DataFrame({"y": y, "x": x, "z": z, "endog": endog, "g": rng.choice(list("AB"), n),
                         "firm": firm.astype(str), "year": year.astype(str)})


DF = _panel()


def _strict(code):
    return run(code, {"df": DF}, profile=Profile.STRICT)


def test_df_feols_formula_summary():
    r = _strict("df.feols('y ~ x | firm + year').summary()")
    assert r.ok and r.payload["family"] == "feols"
    assert {t["term"] for t in r.payload["terms"]} == {"x"}   # FE absorbed
    assert r.payload["fixed_effects"] == ["firm", "year"]


def test_pf_feols_import_form():
    code = ("from pyfixest import feols\n"
            "feols('y ~ x | firm', data=df).summary()")
    r = _strict(code)
    assert r.ok and r.payload["family"] == "feols"


def test_pf_module_form_and_tidy():
    code = ("import pyfixest as pf\n"
            "pf.feols('y ~ x + C(g)', data=df).tidy()")
    r = _strict(code)
    assert r.ok
    assert any("g" in t["term"] for t in r.payload["terms"])   # categorical level reported


def test_iv_via_formula():
    r = _strict("df.feols('y ~ x | firm | endog ~ z').summary()")
    assert r.ok
    assert {"x", "endog"} <= {t["term"] for t in r.payload["terms"]}


def test_clustered_via_vcov():
    code = "import pyfixest as pf\npf.feols('y ~ x | firm', data=df, vcov={'CRV1': 'firm'}).summary()"
    assert _strict(code).ok


def test_predict_is_private_column():
    r = _strict("df.feols('y ~ x | firm').predict().mean()")
    assert r.ok and r.kind == "scalar"


def test_dangling_result_refused():
    assert _strict("df.feols('y ~ x | firm')").ok is False


# ---- the formula string cannot smuggle code ---------------------------------

@pytest.mark.parametrize("fml", [
    "y ~ np.log(x) | firm",           # transform / call
    "y ~ x | firm | nope ~ z",        # unknown column in IV part
    "y ~ x | firm + secret",          # unknown FE column
])
def test_formula_injection_refused(fml):
    assert _strict(f"df.feols({fml!r}).summary()").ok is False


def test_bad_vcov_refused():
    code = "import pyfixest as pf\npf.feols('y ~ x', data=df, vcov='evil').summary()"
    assert _strict(code).ok is False
