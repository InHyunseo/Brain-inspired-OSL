"""Shared trace loaders for the 2D-OSL phase scripts.

Mirrors osl_analysis ``_io.py`` but for the 2D npz schema produced by
:mod:`Analysis.osl2d.eval_dump` (kinematics/events instead of pose/wind/pid).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import numpy as np

from Analysis.osl2d.eval_dump import KINEMATIC_KEYS
from Analysis.osl2d.segment import EVENT_KEYS


@dataclass
class TraceSet:
    obs: np.ndarray             # (N_steps, 5)
    h: np.ndarray               # (N_steps, H)
    action: np.ndarray          # (N_steps, 3)
    reward: np.ndarray          # (N_steps,)
    label: np.ndarray           # (N_steps,) int8
    kinematics: np.ndarray      # (N_steps, 9)
    events: np.ndarray          # (N_steps, 6)
    episode_id: np.ndarray      # (N_steps,) int64 — global ep id
    seed: np.ndarray            # (N_steps,) int64
    ckpt_label: np.ndarray      # (N_steps,) object
    episode_lengths: np.ndarray  # (N_episodes,)
    success: np.ndarray         # (N_episodes,) int8

    def kin(self, name: str) -> np.ndarray:
        """Column accessor for the kinematics array by name."""
        return self.kinematics[:, KINEMATIC_KEYS.index(name)]


def find_trace_files(run_dir: Path, ckpt_labels=None):
    base = run_dir / "analysis" / "traces"
    files = []
    if ckpt_labels:
        for cl in ckpt_labels:
            files += sorted(glob(str(base / cl / "eval_seed*_ep*.npz")))
    else:
        files += sorted(glob(str(base / "*" / "eval_seed*_ep*.npz")))
    seen, out = set(), []
    for f in files:
        if f in seen:
            continue
        seen.add(f)
        out.append(Path(f))
    return out


def load_traces(run_dir: Path, ckpt_labels=None) -> TraceSet:
    paths = find_trace_files(run_dir, ckpt_labels)
    if not paths:
        raise FileNotFoundError(
            f"No trace files under {run_dir}/analysis/traces (ckpt_labels={ckpt_labels})."
        )
    obs_l, h_l, act_l, rew_l, lab_l, kin_l, ev_l = [], [], [], [], [], [], []
    ep_ids, seeds, ckpt_tags, ep_lens, succ = [], [], [], [], []
    for p in paths:
        data = np.load(p, allow_pickle=True)
        T = len(data["obs"])
        if T == 0:
            continue
        obs_l.append(data["obs"])
        h_l.append(data["h"])
        act_l.append(data["action"])
        rew_l.append(data["reward"])
        lab_l.append(data["label"])
        kin_l.append(
            data["kinematics"] if "kinematics" in data.files
            else np.zeros((T, len(KINEMATIC_KEYS)), dtype=np.float32)
        )
        ev_l.append(
            data["events"] if "events" in data.files
            else np.zeros((T, len(EVENT_KEYS)), dtype=np.float32)
        )
        eid = int(data["episode_id"]) if "episode_id" in data.files else len(ep_lens)
        ep_ids.append(np.full(T, eid, dtype=np.int64))
        s = int(data["seed"]) if "seed" in data.files else 0
        seeds.append(np.full(T, s, dtype=np.int64))
        cl = str(data["ckpt_label"]) if "ckpt_label" in data.files else p.parent.name
        ckpt_tags.append(np.array([cl] * T, dtype=object))
        ep_lens.append(T)
        succ.append(int(data["success"]) if "success" in data.files else 0)

    return TraceSet(
        obs=np.concatenate(obs_l, axis=0),
        h=np.concatenate(h_l, axis=0),
        action=np.concatenate(act_l, axis=0),
        reward=np.concatenate(rew_l, axis=0),
        label=np.concatenate(lab_l, axis=0).astype(np.int8),
        kinematics=np.concatenate(kin_l, axis=0).astype(np.float32),
        events=np.concatenate(ev_l, axis=0).astype(np.float32),
        episode_id=np.concatenate(ep_ids, axis=0),
        seed=np.concatenate(seeds, axis=0),
        ckpt_label=np.concatenate(ckpt_tags, axis=0),
        episode_lengths=np.asarray(ep_lens, dtype=np.int64),
        success=np.asarray(succ, dtype=np.int8),
    )


def analysis_dir(run_dir: Path) -> Path:
    out = run_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def load_group_indices(run_dir: Path, ckpt_label: str) -> dict:
    """Load the per-ckpt group_indices.json written by eval_dump."""
    p = run_dir / "analysis" / "traces" / ckpt_label / "group_indices.json"
    with open(p, "r") as fh:
        return json.load(fh)


def adapter_for_ckpt(run_dir: Path, ckpt_label: str, device: str | None = None):
    """Rebuild a Policy2DAdapter for `ckpt_label` from its training checkpoint.

    Reuses eval_dump's checkpoint resolution so phase3a/3b/4 load the same
    weights the traces were dumped from.
    """
    from Analysis.osl2d.eval_dump import _resolve_ckpt
    from Analysis.osl2d.policy_adapter import Policy2DAdapter

    ckpt_path = _resolve_ckpt(Path(run_dir), ckpt_label)
    return Policy2DAdapter.from_checkpoint(ckpt_path, device=device)
