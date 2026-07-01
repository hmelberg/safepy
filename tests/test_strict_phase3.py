"""Phase 3: statsmodels formula API in STRICT mode, via our own parser.

Asserts familiar `smf.ols("y ~ x").fit().summary()` works, that formula strings
can't smuggle code (we parse them, patsy never sees the raw string), and that
per-observation results stay unreachable.
"""

import numpy as np
import pandas as pd
import pytest

from safepy import run
from safepy.policy import Profile

pytest.importorskip("statsmodels")


def _df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 70, n)
    sex = rng.choice(["F", "M"], n)
    region = np.where(np.arange(n) < 3, "Z", rng.choice(["A", "B"], n))  # Z: n=3
    salary = 20000 + 700 * age + 4000 * (sex == "M") + rng.normal(0, 3000, n)
    died = (rng.random(n) < 0.3).astype(int)
    return pd.DataFrame({"age": age, "sex": sex, "region": region,
                         "salary": salary, "died": died})


DF = _df()


def _strict(code):
    return run(code, {"df": DF}, profile=Profile.STRICT)


# ---- the familiar formula API works -----------------------------------------

def test_ols_formula_summary():
    r = _strict('smf.ols("salary ~ age + C(sex)", data=df).fit().summary()')
    assert r.ok and r.payload["type"] == "regression" and r.payload["family"] == "ols"
    terms = {t["term"] for t in r.payload["terms"]}
    assert any("age" in t for t in terms) and any("sex" in t for t in terms)


def test_ols_bare_categorical():
    r = _strict('smf.ols("salary ~ age + sex", data=df).fit().summary()')
    assert r.ok


def test_logit_formula():
    r = _strict('smf.logit("died ~ age", data=df).fit().summary()')
    assert r.ok and r.payload["family"] == "logit"


def test_interaction_term():
    r = _strict('smf.ols("salary ~ age + age:sex", data=df).fit().summary()')
    assert r.ok


def test_small_category_coefficient_suppressed():
    r = _strict('smf.ols("salary ~ C(region)", data=df).fit().summary()')
    assert r.ok
    zrows = [t for t in r.payload["terms"] if "Z" in t["term"]]
    assert zrows and all(t["coef"] is None for t in zrows)


# ---- formula strings cannot smuggle code ------------------------------------

@pytest.mark.parametrize("formula", [
    "salary ~ np.log(age)",            # transform / call -> not an identifier
    "salary ~ I(age**2)",              # patsy I() -> rejected
    "salary ~ age) + C(sex)",          # garbage token
    "salary ~ __import__('os')",       # injection attempt
    "salary ~ age + nope",             # unknown column
    "salary ~ age ~ sex",              # two '~'
])
def test_formula_injection_refused(formula):
    code = f'smf.ols({formula!r}, data=df).fit().summary()'
    r = _strict(code)
    assert r.ok is False


# ---- per-observation outputs and dangling results stay closed ---------------

def test_predict_unreachable():
    r = _strict('smf.ols("salary ~ age", data=df).fit().predict()')
    assert r.ok is False


def test_dangling_model_and_results_refused():
    assert _strict('smf.ols("salary ~ age", data=df)').ok is False          # SafeModel
    assert _strict('smf.ols("salary ~ age", data=df).fit()').ok is False    # SafeResults


def test_unknown_smf_attr():
    r = _strict('smf.mixedlm("salary ~ age", data=df)')
    assert r.ok is False and "not available" in r.error["message"]
