"""Idiomatic lifelines usage via safe facades.

Enables the traditional style:

    from lifelines import CoxPHFitter
    cph = CoxPHFitter()
    cph.fit(df, duration_col="dur", event_col="died")
    cph.summary                          # aggregate -> released
    cph.predict_partial_hazard().mean()  # per-subject -> a private SafeColumn

The imported names resolve to these facades (never the real lifelines classes);
``from lifelines import X`` is wired through the runtime's controlled
``__import__``. Fitted objects expose only aggregate summaries and
per-observation outputs *as SafeColumns* (aggregate/histogram only, never
revealed). Raw per-subject accessors and any un-implemented attribute raise a
clear error.

The safe result logic is reused from ``StatsMixin`` (the policy-bound ``safe``
verbs), reached via the SafeFrame/SafeColumn passed to ``.fit`` — so there is no
separate policy wiring.
"""

from __future__ import annotations

from .errors import DisclosureError
from .result import Released


def _as_column(arr, index, name, verbs):
    import numpy as np
    import pandas as pd

    from .safeframe import SafeColumn
    s = pd.Series(np.asarray(arr).ravel(), index=index, name=name)
    return SafeColumn(s, verbs)


class SafeCoxPHFitter:
    def __init__(self, **kw):
        self._res = self._fitted = self._verbs = self._model_df = self._index = None

    def fit(self, df, duration_col=None, event_col=None, **kw):
        import numpy as np
        import pandas as pd

        from lifelines import CoxPHFitter

        from .safeframe import SafeFrame
        if not isinstance(df, SafeFrame):
            raise DisclosureError("fit expects the data frame")
        if not duration_col or not event_col:
            raise DisclosureError("fit needs duration_col and event_col")
        verbs, raw = df._verbs, df._df
        x = [c for c in raw.columns if c not in (duration_col, event_col)]
        model_df, support = verbs._survival_design(raw, duration_col, event_col, x)
        cph = CoxPHFitter().fit(model_df, duration_col=duration_col, event_col=event_col)
        s = cph.summary
        params = s["coef"]
        ci = pd.DataFrame({0: s["coef lower 95%"], 1: s["coef upper 95%"]})
        self._res = verbs._release_coeffs(
            params, ci, s["p"], support, family="cox", n=int(raw.shape[0]),
            extra={"hazard_ratio": {t: float(np.exp(params[t])) for t in params.index}})
        self._fitted, self._model_df, self._index, self._verbs = cph, model_df, raw.index, verbs
        return self

    @property
    def summary(self): return self._require()
    def print_summary(self, **kw): return self._require()

    def predict_partial_hazard(self, **kw):
        return _as_column(self._fitted.predict_partial_hazard(self._model_df),
                          self._index, "partial_hazard", self._verbs)

    def predict_log_partial_hazard(self, **kw):
        return _as_column(self._fitted.predict_log_partial_hazard(self._model_df),
                          self._index, "log_partial_hazard", self._verbs)

    def _require(self):
        if self._res is None:
            raise DisclosureError("call .fit(...) first")
        return self._res

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raise DisclosureError(
            f"CoxPHFitter.{name} is not available; use .summary, .print_summary(), "
            "or predict_partial_hazard()")


class SafeKaplanMeierFitter:
    def __init__(self, **kw):
        self._curve = self._verbs = None

    def fit(self, durations, event_observed=None, **kw):
        from .safeframe import SafeColumn
        if not isinstance(durations, SafeColumn):
            raise DisclosureError("fit expects a duration column, e.g. df['dur']")
        self._verbs = durations._verbs
        ev = event_observed._s if isinstance(event_observed, SafeColumn) else event_observed
        self._curve = self._verbs._km_curve(durations._s, ev, self._verbs._policy.min_n)
        return self

    def plot(self, **kw):
        from .charts import chart_released
        c = self._need()
        data = {"type": "series", "name": "survival",
                "index": [str(t) for t in c["time"]], "values": c["survival"]}
        return chart_released("line", data, {"verb": "kaplan_meier", "backend": "lifelines"})

    @property
    def survival_function_(self):
        c = self._need()
        return Released({"type": "series", "name": "survival",
                         "index": [str(t) for t in c["time"]], "values": c["survival"]},
                        audit={"kind": "table", "verb": "kaplan_meier", "backend": "lifelines"})

    @property
    def median_survival_time_(self):
        return Released({"type": "scalar", "stat": "median_survival",
                         "value": self._need()["median"], "n": None},
                        audit={"kind": "scalar", "verb": "kaplan_meier", "backend": "lifelines"})

    def print_summary(self, **kw): return self.survival_function_

    def _need(self):
        if self._curve is None:
            raise DisclosureError("call .fit(...) first")
        return self._curve

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raise DisclosureError(
            f"KaplanMeierFitter.{name} is not available; use .survival_function_, "
            ".median_survival_time_, or .plot()")


class SafeWeibullAFTFitter:
    def __init__(self, **kw):
        self._res = None

    def fit(self, df, duration_col=None, event_col=None, **kw):
        from .safeframe import SafeFrame
        if not isinstance(df, SafeFrame):
            raise DisclosureError("fit expects the data frame")
        if not duration_col or not event_col:
            raise DisclosureError("fit needs duration_col and event_col")
        raw = df._df
        x = [c for c in raw.columns if c not in (duration_col, event_col)]
        self._res = df._verbs.weibull_aft(df, duration=duration_col, event=event_col, x=x)
        return self

    @property
    def summary(self): return self._require()
    def print_summary(self, **kw): return self._require()

    def _require(self):
        if self._res is None:
            raise DisclosureError("call .fit(...) first")
        return self._res

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raise DisclosureError(f"WeibullAFTFitter.{name} is not available; use .summary")


class SafeLifelinesModule:
    """What ``import lifelines`` / ``from lifelines import X`` resolves to."""

    def __init__(self):
        self.CoxPHFitter = SafeCoxPHFitter
        self.KaplanMeierFitter = SafeKaplanMeierFitter
        self.WeibullAFTFitter = SafeWeibullAFTFitter

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raise DisclosureError(
            f"lifelines.{name} is not available in safepy (supported: CoxPHFitter, "
            "KaplanMeierFitter, WeibullAFTFitter)")


# fitter facades whose dangling instance the mediator should refuse with guidance
FITTERS = (SafeCoxPHFitter, SafeKaplanMeierFitter, SafeWeibullAFTFitter)
