"""Phase 3a — Linear probing of group activations.

Pool hidden states across all eval episodes (all seeds), slice by group
(motor / command / inter / all), and fit a linear model to predict:
- angle (continuous)
- angvel (continuous) — the *masked* variable; if a group's activations
  encode it well, that group hosts the working-memory representation
- segment label (categorical)

Reports R² (regression) / accuracy (classification) with shuffle baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from Analysis.utils.probe import probe_classification, probe_regression
from Analysis.utils.segment import segment_episode


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", default="ncp_pendulum")
    return p.parse_args()


def main():
    args = parse_args()
    base = Path(__file__).parent / "runs" / args.run_id
    fig_dir = Path(__file__).parent / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Pool data across seeds
    H_all, angle_all, angvel_all, seg_all = [], [], [], []
    gi = None
    for sd in sorted(base.glob("seed_*")):
        if gi is None and (sd / "group_indices.json").exists():
            with (sd / "group_indices.json").open() as f:
                gi_raw = json.load(f)
            gi = {k: np.asarray(v, dtype=np.int64)
                  for k, v in gi_raw.items() if k in ("motor", "command", "inter")}
        for npz in sorted(sd.glob("eval_ep*.npz")):
            d = np.load(npz)
            H_all.append(d["h"])
            angle_all.append(d["angle"])
            angvel_all.append(d["angvel"])
            seg_all.append(segment_episode(d["angle"]))
    if not H_all or gi is None:
        raise SystemExit("no data")
    H = np.concatenate(H_all)
    angle = np.concatenate(angle_all)
    angvel = np.concatenate(angvel_all)
    seg = np.concatenate(seg_all)
    print(f"pooled timesteps: {len(H)}, hidden dim: {H.shape[1]}")

    groups = {**gi, "all": np.arange(H.shape[1])}
    results = {}
    for gname, idx in groups.items():
        X = H[:, idx]
        results[gname] = {
            "angle_R2": probe_regression(X, angle, seed=0),
            "angvel_R2": probe_regression(X, angvel, seed=0),
            "segment_acc": probe_classification(X, seg, seed=0),
            "n_features": int(len(idx)),
        }

    with (base / "phase3a_probe.json").open("w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

    # Heatmap: group × label, value = R² or acc (with shuffle subtracted)
    rows = list(groups.keys())
    cols = ["angle (R²)", "angvel (R²)", "segment (acc)"]
    M = np.zeros((len(rows), len(cols)))
    for i, g in enumerate(rows):
        M[i, 0] = results[g]["angle_R2"]["r2"]
        M[i, 1] = results[g]["angvel_R2"]["r2"]
        M[i, 2] = results[g]["segment_acc"]["acc"]

    fig, ax = plt.subplots(figsize=(5, 3))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=15)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title("Linear probe: group activations → label")
    fig.tight_layout()
    fig.savefig(fig_dir / "phase3a_probe_heatmap.png", dpi=120)
    plt.close(fig)
    print(f"figures → {fig_dir}")


if __name__ == "__main__":
    main()
