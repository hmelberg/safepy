"""Idiomatic pyfixest usage via a safe facade.

Enables the library's own syntax:

    from pyfixest import feols
    feols("Y ~ X1 | f1 + f2", data=df).summary()
    feols("Y ~ X1 | f1 | endog ~ z", data=df, vcov={"CRV1": "f1"}).tidy()

The formula is parsed by our whitelisted ``parse_fixest_formula`` and rebuilt
from validated tokens before it reaches pyfixest/formulaic, so the string can't
smuggle code. ``vcov`` is validated too (only iid/robust or a cluster on a real
column). Fixed effects are absorbed (never reported); categorical coefficients
are per-level suppressed; ``.summary()``/``.tidy()`` release the coefficient
table; ``.predict()`` is a private ``SafeColumn``.
"""

from __future__ import annotations

from .errors import DisclosureError
from .formula import parse_fixest_formula

_VCOV_STRINGS = frozenset({"iid", "hetero", "HC1", "HC2", "HC3"})


def _validate_vcov(vcov, columns):
    if vcov is None:
        return "iid"
    if isinstance(vcov, str):
        if vcov not in _VCOV_STRINGS:
            raise DisclosureError(f"unknown vcov {vcov!r}")
        return vcov
    if isinstance(vcov, dict):
        out = {}
        for kind, col in vcov.items():
            if kind not in ("CRV1", "CRV3"):
                raise DisclosureError(f"unknown vcov type {kind!r}")
            for c in str(col).replace("+", " ").split():
                if c not in columns:
                    raise DisclosureError(f"unknown cluster column: {c}")
            out[kind] = col
        return out
    raise DisclosureError("invalid vcov")


class SafeFeolsResults:
    """A fitted pyfixest model. Only aggregates / private columns leave."""

    def __init__(self, verbs, model, support, n, index, fe, cluster):
        self._verbs = verbs
        self._model = model
        self._support = support
        self._n = n
        self._index = index
        self._fe = fe
        self._cluster = cluster

    def summary(self, **kw): return self._release()
    def tidy(self, **kw): return self._release()

    def _release(self):
        return self._verbs._release_fixest(
            self._model, self._support, family="feols", n=self._n,
            fe=self._fe or None, cluster=self._cluster)

    def predict(self, **kw):
        import numpy as np
        import pandas as pd

        from .safeframe import SafeColumn
        s = pd.Series(np.asarray(self._model.predict()).ravel(), index=self._index,
                      name="predicted")
        return SafeColumn(s, self._verbs)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raise DisclosureError(
            f"feols result .{name} is not available; use .summary(), .tidy() or "
            ".predict()")


class SafePyfixest:
    """What ``import pyfixest`` / ``from pyfixest import feols`` resolves to."""

    def feols(self, fml, data, vcov=None, **kw):
        return self._fit("feols", fml, data, vcov)

    def fepois(self, fml, data, vcov=None, **kw):
        return self._fit("fepois", fml, data, vcov)

    def _fit(self, which, fml, data, vcov):
        import pyfixest as pf

        from .safeframe import SafeFrame
        if not isinstance(data, SafeFrame):
            raise DisclosureError("feols needs data=<the data frame>")
        verbs, raw = data._verbs, data._df
        canonical, base, fe_cols = parse_fixest_formula(fml, list(raw.columns))
        vcov2 = _validate_vcov(vcov, raw.columns)
        fitter = pf.feols if which == "feols" else pf.fepois
        model = fitter(canonical, data=raw, vcov=vcov2)
        n = int(raw.shape[0])
        support = verbs._support(model.tidy().index, raw, [], n)
        cluster = next(iter(vcov2.values())) if isinstance(vcov2, dict) else None
        return SafeFeolsResults(verbs, model, support, n, raw.index, fe_cols, cluster)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raise DisclosureError(
            f"pf.{name} is not available in safepy (supported: feols, fepois)")


FITTERS = (SafeFeolsResults,)
