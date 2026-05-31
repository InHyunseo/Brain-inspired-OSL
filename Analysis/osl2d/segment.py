"""Behavior labels + segment utilities for 2D-OSL traces.

Labels collapse the env's 6 event flags (``event_is_*`` from
``src/envs/events.py::classify_event``) into a single categorical label per
timestep. We use three mutually-exclusive labels, because only two behaviors are
both well-defined and analytically interesting:

    0 = RUN             — forward locomotion (is_run). Ablating it trivially stops
                          progress, so it serves only as a control.
    1 = ACTIVE_SENSING  — head sweep while stopped ("casting"): the animal actively
                          moves its sensors to gather gradient information. This is
                          the behavior of interest.
    2 = OTHER           — everything else (body reorientation / stop / spin). The
                          old TURN label was a catch-all (≈75% of steps, "not a
                          clean run, not a sweep") and STOP/SPIN were too rare
                          (<0.2%) to analyze, so they are merged here as background.

Priority (high→low): ACTIVE_SENSING > RUN > OTHER — an active head sweep is the
salient behavior and overrides locomotion when both flags fire.

Mirrors osl_analysis ``utils/segment.py`` (segment_ratios / run_length /
transition_matrix) so the rest of the pipeline is label-set agnostic.
"""
from __future__ import annotations

import numpy as np

LABELS = ("RUN", "ACTIVE_SENSING", "OTHER")
LABEL_TO_INT = {name: i for i, name in enumerate(LABELS)}
INT_TO_LABEL = {i: name for name, i in LABEL_TO_INT.items()}
N_LABELS = len(LABELS)
# Short display alias for figures/talks.
LABEL_DISPLAY = {"RUN": "RUN", "ACTIVE_SENSING": "active sensing", "OTHER": "other"}

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

    ``events`` is ``(T, 6)`` in EVENT_KEYS order. Three labels, priority (high→low):
    ACTIVE_SENSING > RUN > OTHER. Active head sweeps (low_sweep | high_cast_like)
    win because they are the salient information-gathering behavior; everything
    that is neither a clean run nor an active sweep falls into OTHER.
    """
    events = np.asarray(events)
    T = events.shape[0]
    is_run = events[:, 0].astype(bool)
    is_low_sweep = events[:, 2].astype(bool)
    is_high_cast = events[:, 3].astype(bool)
    active_sensing = is_low_sweep | is_high_cast

    out = np.full(T, LABEL_TO_INT["OTHER"], dtype=np.int8)
    out[is_run] = LABEL_TO_INT["RUN"]
    out[active_sensing] = LABEL_TO_INT["ACTIVE_SENSING"]   # overrides run
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
