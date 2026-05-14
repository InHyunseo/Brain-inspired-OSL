"""Phase 3b — Activation patching (causal ablation).

For each (seed × group × mode), run deterministic eval rollouts with the
hidden state of `group` patched at every step, and measure:
- mean episode return
- balance-segment ratio
vs. unpatched baseline.

Groups: motor / command / inter (hidden state slices)
Modes : zero / mean / flip
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from Analysis.eval_dump import load_policy
from Analysis.ncp_policy import action_to_env
from Analysis.pendulum_pomdp import VelocityMaskedPendulum
from Analysis.utils.segment import LABEL_TO_INT, segment_episode

DEVICE = torch.device("cpu")
N_EPISODES = 20
MAX_STEPS = 200


def rollout_patched(policy, env_seed: int, indices, mode: str | None):
    env = VelocityMaskedPendulum()
    obs, _ = env.reset(seed=env_seed)
    h = policy.initial_state(1, device=DEVICE)
    patch = None
    if mode is not None and indices is not None and len(indices) > 0:
        patch = {"indices": torch.as_tensor(indices, dtype=torch.long), "value": mode}
    angles, rewards = [], []
    for _ in range(MAX_STEPS):
        o = torch.from_numpy(obs).float().unsqueeze(0)
        a, _, _, h = policy.act(o, h, deterministic=True, patch=patch)
        ns, r, term, trunc, info = env.step(action_to_env(a[0]))
        rewards.append(float(r)); angles.append(float(info["angle"]))
        obs = ns
        if term or trunc:
            break
    env.close()
    angles = np.asarray(angles); rewards = np.asarray(rewards)
    labels = segment_episode(angles) if len(angles) else np.array([])
    bal = float(np.mean(labels == LABEL_TO_INT["balance"])) if len(labels) else 0.0
    return {"return": float(rewards.sum()), "balance_ratio": bal, "T": int(len(angles))}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", default="ncp_pendulum")
    p.add_argument("--episodes", type=int, default=N_EPISODES)
    p.add_argument("--modes", nargs="+", default=["zero", "mean", "flip"])
    return p.parse_args()


def main():
    args = parse_args()
    base = Path(__file__).parent / "runs" / args.run_id
    fig_dir = Path(__file__).parent / "figures"
    fig_dir.mkdir(exist_ok=True)

    seed_dirs = sorted([p for p in base.glob("seed_*") if (p / "policy.pt").exists()])
    if not seed_dirs:
        raise SystemExit(f"no seeds under {base}")

    rows = []
    for sd in seed_dirs:
        seed = int(sd.name.split("_")[1])
        policy = load_policy(sd / "policy.pt", DEVICE)
        with (sd / "group_indices.json").open() as f:
            gi_raw = json.load(f)
        groups = {k: gi_raw[k] for k in ("motor", "command", "inter")}

        # Baseline (no patch)
        base_runs = [rollout_patched(policy, 20_000 + seed * 1000 + e, None, None)
                     for e in range(args.episodes)]
        base_ret = float(np.mean([r["return"] for r in base_runs]))
        base_bal = float(np.mean([r["balance_ratio"] for r in base_runs]))
        rows.append({"seed": seed, "group": "baseline", "mode": "none",
                     "return": base_ret, "balance_ratio": base_bal,
                     "delta_return": 0.0, "delta_balance": 0.0})

        for gname, idx in groups.items():
            for mode in args.modes:
                runs = [rollout_patched(policy, 20_000 + seed * 1000 + e, idx, mode)
                        for e in range(args.episodes)]
                ret = float(np.mean([r["return"] for r in runs]))
                bal = float(np.mean([r["balance_ratio"] for r in runs]))
                rows.append({"seed": seed, "group": gname, "mode": mode,
                             "return": ret, "balance_ratio": bal,
                             "delta_return": ret - base_ret,
                             "delta_balance": bal - base_bal})
                print(f"seed={seed} {gname}/{mode}: ret={ret:.1f} (Δ{ret-base_ret:+.1f}) "
                      f"bal={bal:.2f} (Δ{bal-base_bal:+.2f})")

    out = base / "phase3b_results.json"
    with out.open("w") as f:
        json.dump(rows, f, indent=2)
    print(f"saved → {out}")

    # Aggregate plot: mean Δbalance by (group, mode) across seeds
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        if r["group"] == "baseline":
            continue
        agg[(r["group"], r["mode"])].append(r["delta_balance"])
    groups_seen = sorted({k[0] for k in agg})
    modes_seen = sorted({k[1] for k in agg})
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(groups_seen))
    w = 0.25
    for i, m in enumerate(modes_seen):
        vals = [np.mean(agg[(g, m)]) for g in groups_seen]
        errs = [np.std(agg[(g, m)]) for g in groups_seen]
        ax.bar(x + (i - 1) * w, vals, w, yerr=errs, label=m, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(groups_seen)
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(-0.05, color="r", lw=0.5, ls="--", label="−5% threshold")
    ax.set_ylabel("Δ balance_ratio vs baseline")
    ax.set_title("Phase 3b: group ablation effect")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "phase3b_ablation_bars.png", dpi=120)
    plt.close(fig)
    print(f"figures → {fig_dir}")


if __name__ == "__main__":
    main()
