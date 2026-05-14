"""Phase 2 — Dynamical systems analysis.

For each (obs_t, h_t) in every eval episode, compute the Jacobian
J_t = d h_{t+1} / d h_t and analyze its spectrum:

1. Pool eigenvalues by behavior segment label (balance / swing) and compare
   distributions (Cliff's delta on |λ_top|, on Im(λ_top)).
2. Within each segment, do block-eigendecomposition by hidden-state group
   (motor / command / inter) — report which group hosts the dominant
   oscillatory or slow mode.
3. Test hypothesis: balance-segment top complex λ has |arg(λ)/(2π·dt)|
   matching Phase 1 PSD peak frequency.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from Analysis.eval_dump import load_policy
from Analysis.utils.jacobian import block_decompose, dominant_summary, jacobian_at
from Analysis.utils.segment import LABEL_TO_INT, segment_episode

DT = 0.05


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", default="ncp_pendulum")
    p.add_argument("--stride", type=int, default=2, help="Jacobian every k steps")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    A = a[:, None]; B = b[None, :]
    return float((np.sum(A > B) - np.sum(A < B)) / (len(a) * len(b)))


def main():
    args = parse_args()
    device = torch.device(args.device)
    base = Path(__file__).parent / "runs" / args.run_id
    fig_dir = Path(__file__).parent / "figures"
    fig_dir.mkdir(exist_ok=True)

    seed_dirs = sorted([p for p in base.glob("seed_*") if (p / "policy.pt").exists()])
    if not seed_dirs:
        raise SystemExit(f"no trained seeds under {base}")

    # eigenvalues pooled by segment, across all seeds
    pool = defaultdict(list)   # segment_name -> list of (top_abs, top_imag, top_arg)
    group_pool = defaultdict(lambda: defaultdict(list))  # seg -> grp -> top_abs

    for sd in seed_dirs:
        policy = load_policy(sd / "policy.pt", device)
        with (sd / "group_indices.json").open() as f:
            gi_raw = json.load(f)
        gi = {k: np.asarray(v, dtype=np.int64) for k, v in gi_raw.items()
              if k in ("motor", "command", "inter")}

        for npz in sorted(sd.glob("eval_ep*.npz")):
            d = np.load(npz)
            labels = segment_episode(d["angle"])
            T = len(d["obs"])
            for t in range(0, T - 1, args.stride):
                seg = labels[t]
                if seg == LABEL_TO_INT["transition"]:
                    continue
                seg_name = "balance" if seg == LABEL_TO_INT["balance"] else "swing"
                J = jacobian_at(policy, d["obs"][t], d["h"][t])
                w, _ = np.linalg.eig(J)
                summ = dominant_summary(w)
                pool[seg_name].append((summ["top_abs"], summ["top_imag"], summ["top_arg"]))
                blocks = block_decompose(J, gi)
                for gname, info in blocks.items():
                    bw = info["eigvals"]
                    group_pool[seg_name][gname].append(float(np.max(np.abs(bw))))

    # ---- summary stats ----
    def arr(seg, col):
        return np.asarray([p[col] for p in pool[seg]], dtype=np.float32) if pool[seg] else np.array([])

    bal_abs = arr("balance", 0); swg_abs = arr("swing", 0)
    bal_imag = np.abs(arr("balance", 1)); swg_imag = np.abs(arr("swing", 1))
    bal_arg = np.abs(arr("balance", 2)); swg_arg = np.abs(arr("swing", 2))

    delta_abs = cliffs_delta(bal_abs, swg_abs)
    delta_imag = cliffs_delta(bal_imag, swg_imag)

    # Expected oscillation freq (Hz) inferred from balance-segment complex top λ:
    osc_freq_hz = float(np.median(bal_arg[bal_arg > 1e-3]) / (2 * np.pi * DT)) \
        if (bal_arg > 1e-3).any() else None

    # Compare with Phase 1 PSD peak if available
    p1 = base / "phase1_stats.json"
    psd_peak = None
    if p1.exists():
        psd_peak = json.load(p1.open()).get("balance_psd_peak_hz")

    summary = {
        "n_jacobians": {k: len(v) for k, v in pool.items()},
        "balance_top_abs_mean": float(bal_abs.mean()) if len(bal_abs) else None,
        "swing_top_abs_mean": float(swg_abs.mean()) if len(swg_abs) else None,
        "balance_top_abs_imag_mean": float(bal_imag.mean()) if len(bal_imag) else None,
        "cliffs_delta_top_abs (balance vs swing)": delta_abs,
        "cliffs_delta_top_imag (balance vs swing)": delta_imag,
        "balance_inferred_osc_freq_hz": osc_freq_hz,
        "phase1_psd_peak_hz": psd_peak,
        "group_top_abs_mean": {
            seg: {g: float(np.mean(vs)) for g, vs in d.items()}
            for seg, d in group_pool.items()
        },
    }

    with (base / "phase2_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # ---- plot: eigenvalue scatter on complex plane, segment-colored ----
    fig, ax = plt.subplots(figsize=(5, 5))
    if pool["swing"]:
        s = np.asarray(pool["swing"])
        ax.scatter(s[:, 0] * np.cos(s[:, 2]), s[:, 0] * np.sin(s[:, 2]),
                   s=6, alpha=0.4, label="swing", c="tab:orange")
    if pool["balance"]:
        b = np.asarray(pool["balance"])
        ax.scatter(b[:, 0] * np.cos(b[:, 2]), b[:, 0] * np.sin(b[:, 2]),
                   s=6, alpha=0.4, label="balance", c="tab:blue")
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), "k-", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Re(λ_top)"); ax.set_ylabel("Im(λ_top)")
    ax.set_title(f"Dominant eigenvalue per segment (Cliff's δ |λ|={delta_abs:.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "phase2_spectrum_by_segment.png", dpi=120)
    plt.close(fig)
    print(f"figures → {fig_dir}")


if __name__ == "__main__":
    main()
