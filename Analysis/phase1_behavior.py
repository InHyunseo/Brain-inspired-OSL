"""Phase 1 — Behavioral quantification.

Reads all eval_*.npz under runs/{run_id}/seed_*/ and computes:
- per-episode return
- segment ratios (balance / swing / transition)
- angle PSD (Welch) — identify oscillation peak frequency
- kinematic summary (angle var, angvel var, action var)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from Analysis.utils.segment import LABELS, LABEL_TO_INT, segment_episode, segment_ratios

DT = 0.05   # Pendulum-v1 step is 0.05s


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", default="ncp_pendulum")
    return p.parse_args()


def main():
    args = parse_args()
    base = Path(__file__).parent / "runs" / args.run_id
    fig_dir = Path(__file__).parent / "figures"
    fig_dir.mkdir(exist_ok=True)

    eps = sorted(base.glob("seed_*/eval_ep*.npz"))
    if not eps:
        raise SystemExit(f"no eval npz under {base}")
    print(f"found {len(eps)} episodes")

    returns, ratios, balance_angles, all_angles, all_actions, all_angvels = [], [], [], [], [], []
    for p in eps:
        d = np.load(p)
        returns.append(float(d["reward"].sum()))
        labels = segment_episode(d["angle"])
        ratios.append(segment_ratios(labels))
        # collect balance-segment angles for PSD
        balance_angles.append(d["angle"][labels == LABEL_TO_INT["balance"]])
        all_angles.append(d["angle"])
        all_actions.append(d["action"].squeeze())
        all_angvels.append(d["angvel"])

    mean_ratio = {l: float(np.mean([r[l] for r in ratios])) for l in LABELS}
    stats = {
        "n_episodes": len(eps),
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "segment_ratio_mean": mean_ratio,
        "angle_var": float(np.var(np.concatenate(all_angles))),
        "angvel_var": float(np.var(np.concatenate(all_angvels))),
        "action_var": float(np.var(np.concatenate(all_actions))),
    }

    # PSD on concatenated balance-segment angles (the candidate oscillatory mode locus)
    bal = np.concatenate([b for b in balance_angles if len(b) > 8]) if balance_angles else np.array([])
    if len(bal) > 16:
        f, P = welch(bal - bal.mean(), fs=1.0 / DT, nperseg=min(64, len(bal)))
        peak = float(f[np.argmax(P)])
        stats["balance_psd_peak_hz"] = peak
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.semilogy(f, P)
        ax.set_xlabel("Hz"); ax.set_ylabel("PSD(angle)")
        ax.set_title(f"balance-segment angle PSD (peak={peak:.2f} Hz)")
        fig.tight_layout()
        fig.savefig(fig_dir / "phase1_psd_balance.png", dpi=120)
        plt.close(fig)
    else:
        stats["balance_psd_peak_hz"] = None

    # Segment ratio bar
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(LABELS, [mean_ratio[l] for l in LABELS])
    ax.set_ylabel("episode fraction")
    ax.set_title(f"segment ratios ({stats['n_episodes']} eps)")
    fig.tight_layout()
    fig.savefig(fig_dir / "phase1_segment_ratios.png", dpi=120)
    plt.close(fig)

    out = base / "phase1_stats.json"
    with out.open("w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"figures → {fig_dir}")


if __name__ == "__main__":
    main()
