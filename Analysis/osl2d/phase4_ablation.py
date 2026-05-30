"""Phase 4 — Causal ablation of label-key neuron groups (2D, live env).

For each behavior-key neuron group from phase2c (`neuron_groups.json`):
  - run N deterministic rollouts in a live ``OslEnv`` with that group's hidden
    units zeroed at every step,
  - compare label distribution / return / success against a no-ablation
    baseline from the same seeds.

Unlike the 3D version this needs no ROS2 simulator — ``OslEnv`` is a plain gym
env, so this runs in Colab. Ports osl_analysis ``phase4_ablation.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.envs.osl_env import OslEnv
from Analysis.osl2d._io import analysis_dir, write_json
from Analysis.osl2d.eval_dump import _env_config_from_ckpt, _resolve_ckpt
from Analysis.osl2d.policy_adapter import Policy2DAdapter
from Analysis.osl2d.segment import EVENT_KEYS, LABELS, LABEL_TO_INT, labels_from_event_flags


def _label_distribution(labels: np.ndarray) -> dict:
    return {name: float(np.mean(labels == LABEL_TO_INT[name])) if len(labels) else 0.0
            for name in LABELS}


def _kl(p: dict, q: dict, eps: float = 1e-6) -> float:
    total = 0.0
    for k in LABELS:
        pi = max(p.get(k, 0.0), eps)
        qi = max(q.get(k, 0.0), eps)
        total += pi * np.log(pi / qi)
    return float(total)


def _rollout(env, adapter, seed, patch_indices=None, max_steps=None):
    obs, _ = env.reset(seed=seed)
    hidden = adapter.initial_state()
    rew_buf, ev_buf = [], []
    success = 0
    steps = 0
    T_cap = max_steps if max_steps is not None else 10_000
    while steps < T_cap:
        patch = None
        if patch_indices is not None:
            patch = {"indices": patch_indices, "value": "zero"}
        action, h_next = adapter.step_patched(obs, hidden, patch=patch)
        hidden = h_next
        obs, reward, terminated, truncated, info = env.step(action)
        rew_buf.append(float(reward))
        ev_buf.append([float(info[k]) for k in EVENT_KEYS])
        if info.get("success") or info.get("is_success"):
            success = 1
        steps += 1
        if terminated or truncated:
            break
    rew = np.asarray(rew_buf, dtype=np.float32)
    labels = labels_from_event_flags(np.asarray(ev_buf, dtype=np.float32))
    return {
        "return": float(rew.sum()),
        "steps": int(len(rew)),
        "success": int(success),
        "labels": labels,
        "label_distribution": _label_distribution(labels),
    }


def _agg(runs):
    returns = np.asarray([r["return"] for r in runs], dtype=np.float32)
    steps = np.asarray([r["steps"] for r in runs], dtype=np.float32)
    success = np.asarray([r["success"] for r in runs], dtype=np.float32)
    all_labels = (np.concatenate([r["labels"] for r in runs], axis=0)
                  if runs else np.zeros(0, dtype=np.int8))
    return {
        "n_episodes": int(len(runs)),
        "return_mean": float(returns.mean()) if runs else 0.0,
        "return_std": float(returns.std()) if runs else 0.0,
        "steps_mean": float(steps.mean()) if runs else 0.0,
        "success_rate": float(success.mean()) if runs else 0.0,
        "label_distribution": _label_distribution(all_labels),
    }


def run(run_dir, ckpt_label="best", seeds=(0,), episodes_per_seed: int = 3,
        max_steps=None, noise_stage: int = 2, noise_strength: float = 1.0,
        device: str | None = None):
    run_dir = Path(run_dir)
    out_dir = analysis_dir(run_dir)

    groups_path = out_dir / "neuron_groups.json"
    if not groups_path.exists():
        write_json(out_dir / "phase4_ablation.json",
                   {"skipped": "neuron_groups.json not found — run phase2c first."})
        return
    groups = json.loads(groups_path.read_text()).get("groups", {})

    ckpt_path = _resolve_ckpt(run_dir, ckpt_label)
    adapter = Policy2DAdapter.from_checkpoint(ckpt_path, device=device)
    env_cfg = _env_config_from_ckpt(ckpt_path, noise_stage, noise_strength)
    env = OslEnv(env_cfg)

    results = {"baseline": {}, "ablated": {}, "delta": {}, "meta": {
        "ckpt": str(ckpt_path), "seeds": list(seeds),
        "episodes_per_seed": episodes_per_seed,
    }}

    base_runs = []
    for s in seeds:
        for e in range(int(episodes_per_seed)):
            base_runs.append(_rollout(env, adapter, 10_000 + int(s) * 1000 + e,
                                      patch_indices=None, max_steps=max_steps))
    results["baseline"] = _agg(base_runs)

    for group_name, idx_list in groups.items():
        idx = np.asarray(idx_list, dtype=np.int64)
        run_list = []
        for s in seeds:
            for e in range(int(episodes_per_seed)):
                run_list.append(_rollout(env, adapter, 10_000 + int(s) * 1000 + e,
                                         patch_indices=idx, max_steps=max_steps))
        agg = _agg(run_list)
        results["ablated"][group_name] = agg
        base = results["baseline"]
        results["delta"][group_name] = {
            "d_return_mean": agg["return_mean"] - base["return_mean"],
            "d_success_rate": agg["success_rate"] - base["success_rate"],
            "kl_label_dist_vs_baseline": _kl(agg["label_distribution"], base["label_distribution"]),
            "label_distribution_delta": {
                k: agg["label_distribution"][k] - base["label_distribution"][k]
                for k in LABELS
            },
        }

    write_json(out_dir / "phase4_ablation.json", results)

    group_names = list(results["delta"].keys())
    if group_names:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        d_return = [results["delta"][g]["d_return_mean"] for g in group_names]
        d_success = [results["delta"][g]["d_success_rate"] for g in group_names]
        axes[0].bar(group_names, d_return, color="steelblue")
        axes[0].axhline(0, color="k", lw=0.5)
        axes[0].set_title("Δreturn vs. baseline")
        axes[0].tick_params(axis="x", rotation=30)
        axes[1].bar(group_names, d_success, color="darkorange")
        axes[1].axhline(0, color="k", lw=0.5)
        axes[1].set_title("Δsuccess rate vs. baseline")
        axes[1].tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(out_dir / "phase4_ablation_bars.png", dpi=150)
        plt.close(fig)
    return results


def build_parser():
    p = argparse.ArgumentParser("Analysis.osl2d.phase4_ablation")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default="best")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--episodes-per-seed", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--noise-stage", type=int, default=2)
    p.add_argument("--noise-strength", type=float, default=1.0)
    p.add_argument("--device", default=None)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    run(a.run_dir, a.checkpoint, a.seeds, a.episodes_per_seed, a.max_steps,
        a.noise_stage, a.noise_strength, a.device)


if __name__ == "__main__":
    main()
