"""Behavior labels + segment utilities for 2D-OSL traces.

Labels collapse the env's 6 event flags (``event_is_*`` from
``src/envs/events.py::classify_event``) into a single categorical label per
timestep with a fixed priority order. Five mutually-exclusive labels:

    0 = RUN        — forward locomotion (is_run)
    1 = CAST       — head sweep while stopped (low_sweep | high_cast_like)
    2 = TURN       — body reorientation (turn_like, not running)
    3 = SPIN       — pathological over-rotation (spin_like)
    4 = STOP       — stopped, no sweep/turn/spin

Mirrors osl_analysis ``utils/segment.py`` (segment_ratios / run_length /
transition_matrix) so the rest of the pipeline is label-set agnostic.
"""
from __future__ import annotations

import numpy as np

LABELS = ("RUN", "CAST", "TURN", "SPIN", "STOP")
LABEL_TO_INT = {name: i for i, name in enumerate(LABELS)}
INT_TO_LABEL = {i: name for name, i in LABEL_TO_INT.items()}
N_LABELS = len(LABELS)

# Event-flag keys recorded per timestep (see eval_dump trace `events` field).
EVENT_KEYS = (
    "event_is_run",
    "event_is_stop",
    "event_is_low_sweep",
    "event_is_high_cast_like",
    "event_is_turn_like",
    "event_is_spin_like",
)


def labels_from_event_flags(events: np.ndarray) -> np.ndarray:
    """Collapse per-timestep event flags into a single int8 label array.

    ``events`` is ``(T, 6)`` in EVENT_KEYS order. Priority (high→low):
    SPIN > CAST > TURN > RUN > STOP. SPIN first because it is pathological;
    CAST before TURN because an active head sweep is the salient behavior.
    """
    events = np.asarray(events)
    T = events.shape[0]
    is_run = events[:, 0].astype(bool)
    is_stop = events[:, 1].astype(bool)
    is_low_sweep = events[:, 2].astype(bool)
    is_high_cast = events[:, 3].astype(bool)
    is_turn = events[:, 4].astype(bool)
    is_spin = events[:, 5].astype(bool)

    out = np.full(T, LABEL_TO_INT["STOP"], dtype=np.int8)
    cast = is_low_sweep | is_high_cast
    # Apply in reverse priority so higher priority overwrites.
    out[is_run] = LABEL_TO_INT["RUN"]
    out[is_turn & ~is_run] = LABEL_TO_INT["TURN"]
    out[cast] = LABEL_TO_INT["CAST"]
    out[is_spin] = LABEL_TO_INT["SPIN"]
    return out


def segment_ratios(labels: np.ndarray) -> dict:
    out = {}
    for name, i in LABEL_TO_INT.items():
        out[name] = float(np.mean(labels == i)) if len(labels) else 0.0
    return out


def run_length(mask: np.ndarray) -> np.ndarray:
    """For each True position, count consecutive Trues ending at that index."""
    mask = np.asarray(mask).astype(bool)
    out = np.zeros(len(mask), dtype=np.int64)
    c = 0
    for i, v in enumerate(mask):
        c = c + 1 if v else 0
        out[i] = c
    return out


def transition_matrix(labels: np.ndarray, n: int = N_LABELS) -> np.ndarray:
    M = np.zeros((n, n), dtype=np.int64)
    for a, b in zip(labels[:-1], labels[1:]):
        M[int(a), int(b)] += 1
    return M
