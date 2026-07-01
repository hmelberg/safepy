"""The ``smf`` look-alike: smf.ols(formula, data=df).fit().summary().

Mirrors statsmodels' formula API surface, but the formula is parsed by our own
whitelisted parser (``formula.parse_formula``) and reconstructed before it ever
reaches patsy. Only ``.summary()`` is releasable (a suppressed regression table);
per-observation results (``.predict``/``.resid``/``.fittedvalues``) are not
exposed, because no method returns them. Coefficient suppression reuses
``StatsMixin._release_coeffs``/``_support`` — a dummy for a sub-``min_n``
category is blanked.
"""

from __future__ import annotations

from .errors import DisclosureError
from .formula import parse_formula
from .result import Released

_FITTERS = ("ols", "logit", "poisson")


def _unwrap(df):
    return df._df if getattr(df, "_is_safeframe", False) else df


class SafeStats:
    """Injected into the STRICT namespace as ``smf``."""

    def __init__(self, verbs):
        self._verbs = verbs

    def ols(self, formula, data): return SafeModel(self._verbs, "ols", formula, data)
    def logit(self, formula, data): return SafeModel(self._verbs, "logit", formula, data)
    def poisson(self, formula, data): return SafeModel(self._verbs, "poisson", formula, data)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raise DisclosureError(
            f"smf.{name} is not available in safepy (supported: {', '.join(_FITTERS)})")


class SafeModel:
    """An unfitted model. Call ``.fit()``."""

    def __init__(self, verbs, family, formula, data):
        self._verbs = verbs
        self._family = family
        self._formula = formula
        self._data = data

    def fit(self, **kw):
        import statsmodels.formula.api as smf

        df = _unwrap(self._data)
        outcome, rhs, base = parse_formula(self._formula, list(df.columns))
        canonical = f"{outcome} ~ {rhs}"  # built from validated tokens only
        fitter = {"ols": smf.ols, "logit": smf.logit, "poisson": smf.poisson}[self._family]
        fitted = (fitter(canonical, data=df).fit() if self._family == "ols"
                  else fitter(canonical, data=df).fit(disp=0))
        return SafeResults(self._verbs, self._family, fitted, df, base)


class SafeResults:
    """A fitted model. Only ``.summary()`` is releasable."""

    def __init__(self, verbs, family, fitted, df, base):
        self._verbs = verbs
        self._family = family
        self._fitted = fitted
        self._df = df
        self._base = base

    def summary(self):
        m = self._fitted
        support = self._verbs._support(m.params.index, self._df, self._base, int(m.nobs))
        return self._verbs._release_coeffs(
            m.params, m.conf_int(), m.pvalues, support,
            family=self._family, n=int(m.nobs))

    # Per-observation outputs are private COLUMNS, not forbidden: they return a
    # SafeColumn (like any private column, e.g. salary), so you can aggregate or
    # histogram them but never see individual values. A bare .predict() is a
    # dangling SafeColumn and is refused by the mediator.
    def margeff(self, **kw):
        """Average marginal effects (logit/poisson/probit). Aggregate, with the
        same per-term suppression as the coefficients."""
        from .stats import _num
        m = self._fitted
        if not hasattr(m, "get_margeff"):
            raise DisclosureError("marginal effects are not available for this model")
        mf = m.get_margeff().summary_frame()
        support = self._verbs._support(mf.index, self._df, self._base, int(m.nobs))
        k = self._verbs._policy.min_n
        rows, suppressed = [], []
        for term, row in mf.iterrows():
            blank = support.get(str(term), int(m.nobs)) < k
            rows.append({
                "term": str(term),
                "dydx": None if blank else _num(row.get("dy/dx")),
                "se": None if blank else _num(row.get("Std. Err.")),
                "pvalue": None if blank else _num(row.get("Pr(>|z|)")),
            })
            if blank:
                suppressed.append(str(term))
        return Released(
            {"type": "marginal_effects", "family": self._family, "n": int(m.nobs),
             "terms": rows},
            audit={"kind": "regression", "verb": "margeff", "min_n": k,
                   "terms_suppressed": suppressed, "backend": "statsmodels"})

    def predict(self, **kw):
        return self._as_column(self._fitted.predict(), "predicted")

    @property
    def fittedvalues(self):
        return self._as_column(self._fitted.fittedvalues, "fitted")

    @property
    def resid(self):
        r = getattr(self._fitted, "resid", None)
        if r is None:
            raise DisclosureError("residuals are not available for this model")
        return self._as_column(r, "resid")

    def _as_column(self, arr, name):
        import numpy as np
        import pandas as pd

        from .safeframe import SafeColumn
        s = pd.Series(np.asarray(arr), index=self._df.index, name=name)
        return SafeColumn(s, self._verbs)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raise DisclosureError(f"results.{name} is not available; use .summary(), "
                              ".predict(), .fittedvalues or .resid")
