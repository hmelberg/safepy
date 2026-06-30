"""Safe regression and survival verbs (statsmodels + lifelines).

Mixed into :class:`safepython.safe.SafeVerbs`, so they are reachable as
``safe.ols(...)``, ``safe.cox(...)`` and as ``SafeFrame`` methods.

Two disclosure dangers drive every design choice here (see DESIGN.md):

1. **String mini-languages are hidden code.** A statsmodels/patsy formula string
   is ``eval``-ed by patsy, and the outer AST gate cannot see inside a string
   literal. So we **never accept a user formula**. The caller passes column
   *names*; we validate each against ``^[A-Za-z_]\\w*$`` and against the actual
   columns, then build the formula ourselves. No user text reaches patsy.

2. **A coefficient can identify an individual.** A dummy for a categorical level
   with only a handful of members leaks those members' outcomes; an intercept on
   a tiny sample does the same. So after fitting we compute the *support* (number
   of observations) behind every term and blank any coefficient, CI, and p-value
   whose support is below ``min_n``. For survival curves we drop the tail where
   the at-risk set falls below ``min_n`` — that tail is individual event times.

Only aggregate summaries are ever returned. Per-observation artifacts
(``.predict``, ``.resid``, ``.fittedvalues``, per-subject survival) are never
exposed, because there is no verb that returns them.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .errors import DisclosureError
from .result import Released

_IDENT = re.compile(r"^[A-Za-z_]\w*$")
# matches a patsy categorical term: sex[T.M], C(sex)[T.M]
_CAT_TERM = re.compile(r"^(?:C\()?(\w+)\)?\[T\.(.+?)\]$")


def _num(v):
    """Float or None (for NaN/inf), JSON-safe."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
        return None
    return float(v)


def _validate_idents(*names):
    for n in names:
        if not isinstance(n, str) or not _IDENT.match(n):
            raise DisclosureError(
                f"invalid column name {n!r}: only plain identifiers are allowed "
                "(no formulas, expressions, or function calls)")


def _unwrap(df):
    return df._df if getattr(df, "_is_safeframe", False) else df


class StatsMixin:
    """Regression and survival verbs. Expects ``self._policy`` (a Policy)."""

    # ---- linear / generalised linear models (statsmodels) ------------------

    def ols(self, df, *, y, x, **kw):
        return self._smf(df, "ols", y, x, **kw)

    def logit(self, df, *, y, x, **kw):
        return self._smf(df, "logit", y, x, **kw)

    def poisson(self, df, *, y, x, **kw):
        return self._smf(df, "poisson", y, x, **kw)

    def _smf(self, df, family, y, x):
        import statsmodels.formula.api as smf

        df = _unwrap(df)
        xs = [x] if isinstance(x, str) else list(x)
        _validate_idents(y, *xs)
        missing = [c for c in [y, *xs] if c not in df.columns]
        if missing:
            raise DisclosureError(f"unknown column(s): {missing}")

        # build the formula ourselves from validated names; wrap non-numeric
        # predictors as categorical explicitly.
        terms = []
        for c in xs:
            if pd.api.types.is_numeric_dtype(df[c]):
                terms.append(c)
            else:
                terms.append(f"C({c})")
        formula = f"{y} ~ " + " + ".join(terms)

        fitter = {"ols": smf.ols, "logit": smf.logit, "poisson": smf.poisson}[family]
        model = fitter(formula, data=df).fit(disp=0) if family != "ols" \
            else fitter(formula, data=df).fit()
        return self._release_coeffs(model.params, model.conf_int(), model.pvalues,
                                    self._support(model.params.index, df, xs, int(model.nobs)),
                                    family=family, n=int(model.nobs))

    # ---- Cox proportional hazards (lifelines) ------------------------------

    def cox(self, df, *, duration, event, x):
        from lifelines import CoxPHFitter

        df = _unwrap(df)
        xs = [x] if isinstance(x, str) else list(x)
        _validate_idents(duration, event, *xs)
        missing = [c for c in [duration, event, *xs] if c not in df.columns]
        if missing:
            raise DisclosureError(f"unknown column(s): {missing}")
        k = self._policy.min_n

        # numeric design matrix; one-hot non-numeric covariates and drop any
        # level with fewer than min_n members (a singleton level is disclosive).
        pieces, support = {}, {}
        for c in xs:
            if pd.api.types.is_numeric_dtype(df[c]):
                pieces[c] = df[c]
                support[c] = int(df[c].notna().sum())
            else:
                dummies = pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True)
                for col in dummies.columns:
                    n = int(dummies[col].sum())
                    if n >= k:
                        pieces[col] = dummies[col].astype(float)
                        support[col] = n
        if not pieces:
            raise DisclosureError("no covariates with sufficient support to fit a model")

        model_df = pd.DataFrame({duration: df[duration], event: df[event], **pieces})
        cph = CoxPHFitter().fit(model_df, duration_col=duration, event_col=event)
        s = cph.summary
        params = s["coef"]
        ci = pd.DataFrame({0: s["coef lower 95%"], 1: s["coef upper 95%"]})
        return self._release_coeffs(params, ci, s["p"], support,
                                    family="cox", n=int(df.shape[0]),
                                    extra={"hazard_ratio": {t: _num(np.exp(params[t]))
                                                            for t in params.index}})

    # ---- Kaplan-Meier survival curve (lifelines) ---------------------------

    def kaplan_meier(self, df, *, duration, event, by=None):
        df = _unwrap(df)
        _validate_idents(duration, event, *( [by] if by else [] ))
        for c in [duration, event] + ([by] if by else []):
            if c not in df.columns:
                raise DisclosureError(f"unknown column: {c}")
        k = self._policy.min_n

        if by is None:
            curves = {"all": self._km_curve(df[duration], df[event], k)}
        else:
            curves = {}
            for level, sub in df.groupby(by, observed=True):
                if len(sub) < k:
                    continue  # whole group too small to disclose anything
                curves[str(level)] = self._km_curve(sub[duration], sub[event], k)

        return Released({"type": "survival", "method": "kaplan_meier",
                         "by": by, "curves": curves},
                        audit={"kind": "survival", "verb": "kaplan_meier",
                               "min_n": k, "by": by, "backend": "lifelines"})

    def _km_curve(self, durations, events, k):
        from lifelines import KaplanMeierFitter

        kmf = KaplanMeierFitter().fit(durations, events)
        et = kmf.event_table            # index=time, has 'at_risk'
        sf = kmf.survival_function_     # index=time, col 'KM_estimate'
        # only release time points where the at-risk set is >= min_n; the tail
        # below that reveals individual event times.
        times, surv = [], []
        for t in sf.index:
            at_risk = int(et.loc[t, "at_risk"]) if t in et.index else 0
            if at_risk >= k:
                times.append(_num(t))
                surv.append(_num(sf.loc[t].iloc[0]))
        return {"time": times, "survival": surv}

    # ---- shared release path ----------------------------------------------

    def _support(self, terms, df, xs, n):
        """Observations behind each term; intercept & continuous terms get n,
        categorical level terms get that level's count."""
        out = {}
        for term in terms:
            if term in ("Intercept", "const"):
                out[term] = n
                continue
            m = _CAT_TERM.match(str(term))
            if m and m.group(1) in df.columns:
                col, level = m.group(1), m.group(2)
                out[term] = int((df[col].astype(str) == level).sum())
            else:
                out[term] = n
        return out

    def _release_coeffs(self, params, ci, pvalues, support, *, family, n, extra=None):
        k = self._policy.min_n
        params = params.copy()
        ci = ci.copy()
        ci.columns = [0, 1]
        suppressed = []
        for term in list(params.index):
            if support.get(term, n) < k:
                params[term] = np.nan
                ci.loc[term] = np.nan
                suppressed.append(str(term))

        rows = []
        for term in params.index:
            blanked = pd.isna(params[term])
            rows.append({
                "term": str(term),
                "coef": _num(params[term]),
                "ci_low": _num(ci.loc[term, 0]),
                "ci_high": _num(ci.loc[term, 1]),
                "pvalue": None if blanked else _num(pvalues.get(term)),
                **({"hazard_ratio": (None if blanked else (extra or {}).get("hazard_ratio", {}).get(term))}
                   if extra and "hazard_ratio" in extra else {}),
            })

        return Released(
            {"type": "regression", "family": family, "n": n, "terms": rows},
            audit={"kind": "regression", "verb": family, "min_n": k,
                   "terms_suppressed": suppressed, "backend":
                   "lifelines" if family == "cox" else "statsmodels"})
