"""A tiny synthetic individual-level frame used across tests."""

import numpy as np
import pandas as pd


def salaries() -> pd.DataFrame:
    # 50 individuals; 'sex' has two well-populated groups, 'region' has one
    # deliberately tiny group ("Z": 2 people) to exercise suppression.
    n = 50
    idx = np.arange(n)
    sex = np.where(idx % 2 == 0, "F", "M")
    region = np.where(idx < 2, "Z", np.where(idx < 26, "A", "B"))
    salary = 30000 + idx * 1000
    name = [f"person_{i}" for i in idx]
    return pd.DataFrame({"pid": idx, "name": name, "sex": sex,
                         "region": region, "salary": salary})
