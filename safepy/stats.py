"""Safe regression and survival verbs (statsmodels + lifelines).

Mixed into :class:`safepy.safe.SafeVerbs`, so they are reachable as
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
    if v is None:
        return None
    if isinstance(v, float) and not np.isfinite(v):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return float(v)


def _scalar(v):
    """Coerce a possibly-array-like effect to a single float (None if empty)."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        arr = np.asarray(v, dtype=float).ravel()
        return float(arr.mean()) if arr.size else None


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
        model_df, support = self._survival_design(df, duration, event, xs)
        cph = CoxPHFitter().fit(model_df, duration_col=duration, event_col=event)
        s = cph.summary
        params = s["coef"]
        ci = pd.DataFrame({0: s["coef lower 95%"], 1: s["coef upper 95%"]})
        return self._release_coeffs(params, ci, s["p"], support,
                                    family="cox", n=int(df.shape[0]),
                                    extra={"hazard_ratio": {t: _num(np.exp(params[t]))
                                                            for t in params.index}})

    def _survival_design(self, df, duration, event, xs):
        """Numeric design matrix for a survival model. One-hot non-numeric
        covariates and drop any level with fewer than min_n members (a singleton
        level is disclosive). Returns (model_df, per-covariate support)."""
        _validate_idents(duration, event, *xs)
        missing = [c for c in [duration, event, *xs] if c not in df.columns]
        if missing:
            raise DisclosureError(f"unknown column(s): {missing}")
        k = self._policy.min_n
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
        return pd.DataFrame({duration: df[duration], event: df[event], **pieces}), support

    # ---- parametric accelerated-failure-time models (lifelines) ------------

    def weibull_aft(self, df, *, duration, event, x):
        from lifelines import WeibullAFTFitter
        return self._aft(df, WeibullAFTFitter, "weibull_aft", duration, event, x)

    def lognormal_aft(self, df, *, duration, event, x):
        from lifelines import LogNormalAFTFitter
        return self._aft(df, LogNormalAFTFitter, "lognormal_aft", duration, event, x)

    def loglogistic_aft(self, df, *, duration, event, x):
        from lifelines import LogLogisticAFTFitter
        return self._aft(df, LogLogisticAFTFitter, "loglogistic_aft", duration, event, x)

    def _aft(self, df, fitter_cls, family, duration, event, x):
        df = _unwrap(df)
        xs = [x] if isinstance(x, str) else list(x)
        model_df, support = self._survival_design(df, duration, event, xs)
        fitted = fitter_cls().fit(model_df, duration_col=duration, event_col=event)
        k = self._policy.min_n
        n = int(df.shape[0])
        rows, suppressed = [], []
        for idx, row in fitted.summary.iterrows():
            param, cov = idx if isinstance(idx, tuple) else ("", idx)
            term = f"{param}:{cov}" if param else str(cov)
            blank = support.get(cov, n) < k
            rows.append({
                "term": term,
                "coef": None if blank else _num(row.get("coef")),
                "ci_low": None if blank else _num(row.get("coef lower 95%")),
                "ci_high": None if blank else _num(row.get("coef upper 95%")),
                "pvalue": None if blank else _num(row.get("p")),
            })
            if blank:
                suppressed.append(term)
        return Released(
            {"type": "regression", "family": family, "n": n, "terms": rows},
            audit={"kind": "regression", "verb": family, "min_n": k,
                   "terms_suppressed": suppressed, "backend": "lifelines"})

    # ---- fixed-effects / IV regression (pyfixest) --------------------------

    def _numeric_design(self, df, xs):
        """One-hot non-numeric covariates (dropping sub-min_n levels) into a
        numeric design with identifier-safe names. Returns (pieces, support)."""
        k = self._policy.min_n
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
                        name = re.sub(r"\W", "_", str(col))
                        pieces[name] = dummies[col].astype(float)
                        support[name] = n
        return pieces, support

    def feols(self, df, *, y, x, fe=None, cluster=None):
        import pyfixest as pf

        df = _unwrap(df)
        xs = [x] if isinstance(x, str) else list(x)
        fes = [] if fe is None else ([fe] if isinstance(fe, str) else list(fe))
        clusters = [cluster] if cluster else []
        _validate_idents(y, *xs, *fes, *clusters)
        for c in [y, *xs, *fes, *clusters]:
            if c not in df.columns:
                raise DisclosureError(f"unknown column: {c}")

        pieces, support = self._numeric_design(df, xs)
        if not pieces:
            raise DisclosureError("no covariates with sufficient support")
        data = {y: df[y], **pieces}
        for f in fes:
            data[f] = df[f].astype("category")
        if cluster:
            data[cluster] = df[cluster]
        model_df = pd.DataFrame(data)

        formula = f"{y} ~ " + " + ".join(pieces.keys())
        if fes:
            formula += " | " + " + ".join(fes)
        vcov = {"CRV1": cluster} if cluster else "hetero"
        m = pf.feols(formula, data=model_df, vcov=vcov)
        return self._release_fixest(m, support, family="feols", n=int(df.shape[0]),
                                    fe=fes, cluster=cluster)

    def iv(self, df, *, y, x=None, endog, instruments, fe=None, cluster=None):
        import pyfixest as pf

        df = _unwrap(df)
        xs = [] if x is None else ([x] if isinstance(x, str) else list(x))
        endogs = [endog] if isinstance(endog, str) else list(endog)
        zs = [instruments] if isinstance(instruments, str) else list(instruments)
        fes = [] if fe is None else ([fe] if isinstance(fe, str) else list(fe))
        clusters = [cluster] if cluster else []
        _validate_idents(y, *xs, *endogs, *zs, *fes, *clusters)
        for c in [y, *xs, *endogs, *zs, *fes, *clusters]:
            if c not in df.columns:
                raise DisclosureError(f"unknown column: {c}")
        for c in [*endogs, *zs]:
            if not pd.api.types.is_numeric_dtype(df[c]):
                raise DisclosureError(f"endogenous/instrument column '{c}' must be numeric")

        pieces, support = self._numeric_design(df, xs)
        data = {y: df[y], **pieces}
        for c in [*endogs, *zs]:
            data[c] = df[c]
        for f in fes:
            data[f] = df[f].astype("category")
        if cluster:
            data[cluster] = df[cluster]
        model_df = pd.DataFrame(data)

        exog = " + ".join(pieces.keys()) if pieces else "1"
        parts = [f"{y} ~ {exog}"]
        if fes:
            parts.append(" + ".join(fes))
        parts.append(f"{' + '.join(endogs)} ~ {' + '.join(zs)}")
        formula = " | ".join(parts)
        vcov = {"CRV1": cluster} if cluster else "hetero"
        m = pf.feols(formula, data=model_df, vcov=vcov)
        n = int(df.shape[0])
        for e in endogs:
            support[e] = n  # endogenous regressors use the full sample
        return self._release_fixest(m, support, family="iv", n=n,
                                    fe=fes, cluster=cluster)

    def _release_fixest(self, m, support, *, family, n, fe, cluster):
        k = self._policy.min_n
        rows, suppressed = [], []
        for term, row in m.tidy().iterrows():
            blank = support.get(str(term), n) < k
            rows.append({
                "term": str(term),
                "coef": None if blank else _num(row["Estimate"]),
                "se": None if blank else _num(row["Std. Error"]),
                "ci_low": None if blank else _num(row["2.5%"]),
                "ci_high": None if blank else _num(row["97.5%"]),
                "pvalue": None if blank else _num(row["Pr(>|t|)"]),
            })
            if blank:
                suppressed.append(str(term))
        return Released(
            {"type": "regression", "family": family, "n": n,
             "fixed_effects": fe, "cluster": cluster, "terms": rows},
            audit={"kind": "regression", "verb": family, "min_n": k,
                   "terms_suppressed": suppressed, "backend": "pyfixest"})

    # ---- average treatment effect via propensity methods (DoWhy) -----------

    _ATE_METHODS = {
        "weighting": "backdoor.propensity_score_weighting",
        "matching": "backdoor.propensity_score_matching",
        "stratification": "backdoor.propensity_score_stratification",
        "regression": "backdoor.linear_regression",
    }

    _REFUTERS = {
        "placebo": "placebo_treatment_refuter",
        "random_common_cause": "random_common_cause",
        "data_subset": "data_subset_refuter",
    }

    def _ate_guard(self, df, outcome, treatment, cs, method=None):
        _validate_idents(outcome, treatment, *cs)
        for c in [outcome, treatment, *cs]:
            if c not in df.columns:
                raise DisclosureError(f"unknown column: {c}")
        if method is not None and method not in self._ATE_METHODS:
            raise DisclosureError(
                f"unknown method {method!r}; choose one of {sorted(self._ATE_METHODS)}")
        counts = df[treatment].value_counts()
        if len(counts) != 2:
            raise DisclosureError("treatment must be binary")
        if int(counts.min()) < self._policy.min_n:
            raise DisclosureError("a treatment arm is smaller than min_n")
        return counts

    def _dowhy_fit(self, data, outcome, treatment, cs, method):
        from dowhy import CausalModel
        params = {"weighting_scheme": "ips_weight"} if method == "weighting" else None
        model = CausalModel(data=data, treatment=treatment, outcome=outcome, common_causes=cs)
        iden = model.identify_effect(proceed_when_unidentifiable=True)
        est = model.estimate_effect(iden, method_name=self._ATE_METHODS[method],
                                    method_params=params, confidence_intervals=True)
        return model, iden, est

    def ate(self, df, *, outcome, treatment, confounders, method="weighting"):
        """Average treatment effect under backdoor adjustment (propensity
        weighting/matching/stratification, or regression). Returns only the
        aggregate effect + CI; matched pairs and per-unit propensity scores are
        never exposed. Treatment must be binary and each arm >= min_n."""
        import contextlib
        import io
        import logging

        df = _unwrap(df)
        cs = [confounders] if isinstance(confounders, str) else list(confounders)
        counts = self._ate_guard(df, outcome, treatment, cs, method)
        k = self._policy.min_n

        logging.getLogger("dowhy").setLevel(logging.ERROR)
        data = df[[outcome, treatment, *cs]].copy()  # DoWhy mutates its input
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _, _, est = self._dowhy_fit(data, outcome, treatment, cs, method)
            try:
                lo, hi = est.get_confidence_intervals()
            except Exception:
                lo = hi = None
            try:
                se = est.get_standard_error()
            except Exception:
                se = None

        return Released(
            {"type": "causal_estimate", "estimand": "ate", "method": method,
             "effect": _num(est.value), "ci_low": _num(lo), "ci_high": _num(hi),
             "se": _num(se), "n": int(df.shape[0]),
             "groups": {str(v): int(c) for v, c in counts.items()}},
            audit={"kind": "causal", "verb": "ate", "method": method, "min_n": k,
                   "backend": "dowhy"})

    def refute_ate(self, df, *, outcome, treatment, confounders,
                   method="weighting", refuter="placebo"):
        """Robustness check on an ATE: re-estimate under a refuter (placebo
        treatment, a random common cause, or a data subset) and report the
        original vs refuted effect + p-value. All aggregate."""
        import contextlib
        import io
        import logging

        df = _unwrap(df)
        cs = [confounders] if isinstance(confounders, str) else list(confounders)
        if refuter not in self._REFUTERS:
            raise DisclosureError(
                f"unknown refuter {refuter!r}; choose one of {sorted(self._REFUTERS)}")
        self._ate_guard(df, outcome, treatment, cs, method)
        k = self._policy.min_n

        logging.getLogger("dowhy").setLevel(logging.ERROR)
        data = df[[outcome, treatment, *cs]].copy()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model, iden, est = self._dowhy_fit(data, outcome, treatment, cs, method)
            ref = model.refute_estimate(iden, est, method_name=self._REFUTERS[refuter])

        p = None
        rr = getattr(ref, "refutation_result", None)
        if isinstance(rr, dict):
            p = rr.get("p_value")
        return Released(
            {"type": "causal_refutation", "refuter": refuter, "method": method,
             "estimated_effect": _num(_scalar(ref.estimated_effect)),
             "new_effect": _num(_scalar(ref.new_effect)), "p_value": _num(_scalar(p))},
            audit={"kind": "causal", "verb": "refute_ate", "refuter": refuter,
                   "min_n": k, "backend": "dowhy"})

    def propensity(self, df, *, treatment, confounders):
        """Propensity score P(treatment=1 | confounders) as a private SafeColumn
        (aggregate/histogram only, never revealed per unit). Composes with
        assign for overlap plots: df.assign(ps=df.propensity(...))."""
        import statsmodels.api as sm

        from .safeframe import SafeColumn
        df = _unwrap(df)
        cs = [confounders] if isinstance(confounders, str) else list(confounders)
        _validate_idents(treatment, *cs)
        for c in [treatment, *cs]:
            if c not in df.columns:
                raise DisclosureError(f"unknown column: {c}")
        counts = df[treatment].value_counts()
        if len(counts) != 2:
            raise DisclosureError("treatment must be binary")
        if int(counts.min()) < self._policy.min_n:
            raise DisclosureError("a treatment arm is smaller than min_n")

        pieces, _ = self._numeric_design(df, cs)
        if not pieces:
            raise DisclosureError("no usable confounders")
        X = sm.add_constant(pd.DataFrame(pieces, index=df.index).astype(float))
        vals = sorted(counts.index)
        ybin = df[treatment].map({vals[0]: 0, vals[1]: 1}).astype(float)
        try:
            model = sm.Logit(ybin, X).fit(disp=0)
        except Exception:
            raise DisclosureError("the propensity model failed to converge")
        ps = pd.Series(np.asarray(model.predict(X)), index=df.index, name="propensity")
        return SafeColumn(ps, self)

    # ---- restricted mean survival time (lifelines) -------------------------

    def rmst(self, df, *, duration, event, t, by=None):
        from lifelines import KaplanMeierFitter
        from lifelines.utils import restricted_mean_survival_time

        df = _unwrap(df)
        _validate_idents(duration, event, *([by] if by else []))
        for c in [duration, event] + ([by] if by else []):
            if c not in df.columns:
                raise DisclosureError(f"unknown column: {c}")
        k = self._policy.min_n

        def one(sub):
            kmf = KaplanMeierFitter().fit(sub[duration], sub[event])
            return _num(restricted_mean_survival_time(kmf, t=t))

        if by is None:
            if len(df) < k:
                raise DisclosureError("too few observations to release an RMST")
            values = {"all": one(df)}
        else:
            values = {str(g): one(sub) for g, sub in df.groupby(by, observed=True)
                      if len(sub) >= k}
            if not values:
                raise DisclosureError("no group with >= min_n members")

        return Released({"type": "rmst", "t": t, "by": by, "values": values},
                        audit={"kind": "table", "verb": "rmst", "min_n": k,
                               "t": t, "by": by, "backend": "lifelines"})

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
        import math

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
        # median survival time: released only if the risk set there is >= min_n
        median = None
        med = float(kmf.median_survival_time_)
        if math.isfinite(med):
            rows = et[et.index <= med]
            if len(rows) and int(rows["at_risk"].iloc[-1]) >= k:
                median = _num(med)
        return {"time": times, "survival": surv, "median": median}

    # ---- log-rank test: compare survival between groups --------------------

    def logrank(self, df, *, duration, event, by):
        from lifelines.statistics import multivariate_logrank_test

        df = _unwrap(df)
        _validate_idents(duration, event, by)
        for c in (duration, event, by):
            if c not in df.columns:
                raise DisclosureError(f"unknown column: {c}")
        k = self._policy.min_n

        sizes = df.groupby(by, observed=True).size()
        keep = sizes[sizes >= k].index
        sub = df[df[by].isin(keep)]
        if sub[by].nunique() < 2:
            raise DisclosureError("log-rank needs >= 2 groups each with >= min_n members")

        res = multivariate_logrank_test(sub[duration], sub[by], sub[event])
        return Released(
            {"type": "test", "test": "logrank",
             "statistic": _num(res.test_statistic), "p_value": _num(res.p_value),
             "df": int(sub[by].nunique() - 1),
             "groups": {str(g): int(n) for g, n in sizes.items() if n >= k}},
            audit={"kind": "test", "verb": "logrank", "min_n": k,
                   "groups_dropped": int((sizes < k).sum()), "backend": "lifelines"})

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
