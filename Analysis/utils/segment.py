"""Behavior segmentation for Pendulum POMDP traces.

Two primary modes:
- 'balance': |angle| < BALANCE_THR for >= MIN_BALANCE_LEN consecutive steps
- 'swing':   not balance, and |angle| > SWING_THR
- 'transition': everything else

In the OSL connectome version this file is replaced by run/cast/turn labels;
the rest of the pipeline doesn't change.
"""
from __future__ import annotations

import numpy as np

BALANCE_THR = np.pi / 4
SWING_THR = np.pi / 2
MIN_BALANCE_LEN = 5

LABELS = ("balance", "swing", "transition")
LABEL_TO_INT = {l: i for i, l in enumerate(LABELS)}


def segment_episode(angle: np.ndarray) -> np.ndarray:
    """Return label per timestep, dtype int (indexes into LABELS)."""
    abs_a = np.abs(angle)
    T = len(angle)
    out = np.full(T, LABEL_TO_INT["transition"], dtype=np.int64)
    out[abs_a > SWING_THR] = LABEL_TO_INT["swing"]

    # Balance requires a run of length >= MIN_BALANCE_LEN.
    near = abs_a < BALANCE_THR
    i = 0
    while i < T:
        if near[i]:
            j = i
            while j < T and near[j]:
                j += 1
            if j - i >= MIN_BALANCE_LEN:
                out[i:j] = LABEL_TO_INT["balance"]
            i = j
        else:
            i += 1
    return out


def segment_ratios(labels: np.ndarray) -> dict:
    return {
        name: float(np.mean(labels == LABEL_TO_INT[name]))
        for name in LABELS
    }
