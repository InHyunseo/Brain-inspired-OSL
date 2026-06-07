"""Linear probe with episode-level split (no within-episode leakage).

Ported verbatim from osl_analysis ``utils/probe.py`` — sim-agnostic.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix


def episode_split(episode_ids: np.ndarray, test_size: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    unique = np.unique(episode_ids)
    rng.shuffle(unique)
    n_test = max(1, int(round(len(unique) * test_size))) if len(unique) > 1 else 0
    test_eps = set(unique[:n_test])
    test_mask = np.array([eid in test_eps for eid in episode_ids])
    return ~test_mask, test_mask


def probe_classification_episode_split(
    X: np.ndarray,
    y: np.ndarray,
    episode_ids: np.ndarray,
    test_size: float = 0.2,
    seed: int = 0,
    shuffle_baseline: bool = True,
    return_weights: bool = False,
) -> dict:
    tr, te = episode_split(episode_ids, test_size=test_size, seed=seed)
    if te.sum() == 0 or len(np.unique(y[tr])) < 2:
        return {
            "acc": float("nan"), "macro_f1": float("nan"),
            "per_label_f1": [], "confusion": [],
            "n_train": int(tr.sum()), "n_test": int(te.sum()),
            "n_train_episodes": int(len(np.unique(episode_ids[tr]))),
            "n_test_episodes": int(len(np.unique(episode_ids[te]))),
        }

    m = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
    y_pred = m.predict(X[te])
    classes = m.classes_.tolist()
    f1_per = f1_score(y[te], y_pred, labels=classes, average=None, zero_division=0)
    cm = confusion_matrix(y[te], y_pred, labels=classes)

    out = {
        "acc": float(m.score(X[te], y[te])),
        "macro_f1": float(f1_score(y[te], y_pred, average="macro", zero_division=0)),
        "per_label_f1": [float(v) for v in f1_per],
        "classes": [int(c) for c in classes],
        "confusion": cm.astype(int).tolist(),
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "n_train_episodes": int(len(np.unique(episode_ids[tr]))),
        "n_test_episodes": int(len(np.unique(episode_ids[te]))),
    }
    if shuffle_baseline:
        rng = np.random.default_rng(seed + 1)
        y_perm = rng.permutation(y[tr])
        m0 = LogisticRegression(max_iter=2000).fit(X[tr], y_perm)
        out["acc_shuffle"] = float(m0.score(X[te], y[te]))
    if return_weights:
        out["weights"] = m.coef_.astype(np.float32).tolist()
        out["intercept"] = m.intercept_.astype(np.float32).tolist()
    return out


def probe_weights_episode(
    X: np.ndarray, y: np.ndarray, episode_ids: np.ndarray, seed: int = 0,
):
    tr, _ = episode_split(
        episode_ids,
        test_size=0.0 if len(np.unique(episode_ids)) <= 1 else 0.2,
        seed=seed,
    )
    if len(np.unique(y[tr])) < 2:
        return None, None
    m = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
    return m.coef_.astype(np.float32), np.asarray(m.classes_, dtype=np.int64)
