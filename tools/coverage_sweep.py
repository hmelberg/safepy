"""Phase-4 coverage sweep: how much real pandas/statsmodels runs unchanged in
STRICT mode?

The idiom corpus is drawn from the `py2m` translator's test inputs (the sister
project's catalogue of "normal pandas that real users write"). Each idiom is
tagged:

    want  — a legitimate analysis we would like to support
    block — disclosive/by-design; it MUST stay blocked

Run:  python tools/coverage_sweep.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "protect"))

import numpy as np
import pandas as pd

from safepy import run
from safepy.policy import Profile


def _df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 8, n)                      # 8 groups, each ~25
    return pd.DataFrame({
        "x": rng.normal(10, 3, n), "y": rng.normal(5, 2, n), "z": rng.normal(0, 1, n),
        "a": rng.integers(0, 10, n), "b": rng.integers(0, 10, n), "c": rng.integers(0, 5, n),
        "g": [f"g{i}" for i in g],
        "age": rng.integers(20, 70, n),
        "sex": rng.choice(["F", "M"], n),
        "income": rng.integers(20000, 90000, n),
        "region": np.where(np.arange(n) < 3, "Z", rng.choice(["A", "B"], n)),
        "died": (rng.random(n) < 0.3).astype(int),
    })


DF = _df()

# (tag, code)
IDIOMS = [
    # --- selection / filtering / reducers ---
    ("want", "df['x'].mean()"),
    ("want", "df.groupby('g')['x'].mean()"),
    ("want", "df.groupby('g').salary.mean()".replace("salary", "x")),
    ("want", "df['g'].value_counts()"),
    ("want", "df[df['x'] > 0]['y'].mean()"),
    ("want", "df[(df['a'] > 2) & (df['b'] < 9)]['x'].mean()"),
    ("want", "pd.crosstab(df['sex'], df['region'])"),
    # --- derived columns / transforms ---
    ("want", "df.assign(x2=df['a'] + 1, y2=df['b'] * 2).groupby('g')['x2'].mean()"),
    ("want", "df.assign(l=np.log(df['income'])).groupby('g')['l'].mean()"),
    ("want", "df.assign(m=np.where(df['a'] > 5, 1, 0)).groupby('g')['m'].mean()"),
    ("want", "df.assign(s=np.sqrt(df['a'])).groupby('g')['s'].mean()"),
    ("want", "df.assign(q=pd.qcut(df['income'], 4, labels=False)).groupby('q')['income'].mean()"),
    ("want", "df.assign(bnd=pd.cut(df['age'], bins=[0,30,60], labels=[1,2])).groupby('bnd')['income'].mean()"),
    ("want", "df.assign(bnd=pd.cut(df['age'], bins=[0,30,np.inf], labels=[1,2])).groupby('bnd')['income'].mean()"),
    ("want", "df.assign(xn=pd.to_numeric(df['x'])).groupby('g')['xn'].mean()"),
    # --- aggregation variants ---
    ("want", "df.groupby('g').agg(m=('x','mean'))"),
    ("want", "df.groupby('g').agg({'x':'var'})"),
    ("want", "df[['x','y']].corr()"),
    # --- shaping (harmless) then aggregate ---
    ("want", "df.rename(columns={'a':'aa'}).groupby('g')['x'].mean()"),
    ("want", "df.fillna(0).groupby('g')['x'].mean()"),
    ("want", "df.dropna(subset=['income']).groupby('g')['x'].mean()"),
    ("want", "df.drop(columns=['c']).groupby('g')['x'].mean()"),
    # --- models ---
    ("want", "smf.ols('y ~ x', data=df).fit().summary()"),
    ("want", "smf.ols('y ~ x + z', data=df).fit().summary()"),
    ("want", "smf.ols('y ~ age*sex', data=df).fit().summary()"),
    ("want", "smf.logit('died ~ x', data=df).fit().summary()"),
    # --- MUST stay blocked (disclosive / code-eval) ---
    ("block", "df['x'].describe()"),
    ("block", "df['x'].max()"),
    ("block", "df.groupby('g')['x'].transform('mean')"),
    ("block", "df.query('age > 18')"),
    ("block", "df.sample(n=10)"),
    ("block", "smf.ols('y ~ I(x*z)', data=df).fit().summary()"),
    ("block", "df.sort_values('x')"),
]


def main():
    want_ok = want_total = 0
    unsupported, leaks = [], []
    for tag, code in IDIOMS:
        ok = run(code, {"df": DF}, profile=Profile.STRICT).ok
        if tag == "want":
            want_total += 1
            if ok:
                want_ok += 1
            else:
                unsupported.append(code)
        else:  # block
            if ok:
                leaks.append(code)  # a by-design-block idiom that slipped through!
        mark = "ok   " if ok else "BLOCK"
        print(f"  [{tag:5}] {mark} {code}")

    pct = 100 * want_ok / want_total if want_total else 0
    print(f"\ncoverage: {want_ok}/{want_total} want-idioms supported ({pct:.0f}%)")
    if unsupported:
        print("\nunsupported (gaps):")
        for c in unsupported:
            print("  - " + c)
    if leaks:
        print("\n!!! by-design-block idioms that were NOT blocked (regressions):")
        for c in leaks:
            print("  - " + c)


if __name__ == "__main__":
    main()
