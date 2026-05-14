"""Linear probing wrappers.

Train a linear model from (group activation X) → (label y). Report R² (regression)
or accuracy (classification), with a shuffle-label baseline as control.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split


def probe_regression(X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
                     seed: int = 0, shuffle_baseline: bool = True) -> dict:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)
    m = LinearRegression().fit(Xtr, ytr)
    r2 = float(m.score(Xte, yte))
    out = {"r2": r2, "n_train": int(len(Xtr)), "n_test": int(len(Xte))}
    if shuffle_baseline:
        rng = np.random.default_rng(seed)
        ytr_s = rng.permutation(ytr)
        m0 = LinearRegression().fit(Xtr, ytr_s)
        out["r2_shuffle"] = float(m0.score(Xte, yte))
    return out


def probe_classification(X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
                         seed: int = 0, shuffle_baseline: bool = True) -> dict:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed,
                                          stratify=y if len(set(y)) > 1 else None)
    m = LogisticRegression(max_iter=1000, multi_class="auto").fit(Xtr, ytr)
    acc = float(m.score(Xte, yte))
    out = {"acc": acc, "n_train": int(len(Xtr)), "n_test": int(len(Xte))}
    if shuffle_baseline:
        rng = np.random.default_rng(seed)
        ytr_s = rng.permutation(ytr)
        try:
            m0 = LogisticRegression(max_iter=1000).fit(Xtr, ytr_s)
            out["acc_shuffle"] = float(m0.score(Xte, yte))
        except Exception:
            out["acc_shuffle"] = float("nan")
    return out
