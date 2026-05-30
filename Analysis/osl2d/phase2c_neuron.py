"""Phase 2c — Per-neuron contribution to each behavior label.

Uses logistic-regression probe weights as the contribution proxy:
    contribution[label, neuron] = |W[label, neuron]|

Saves phase2c_neuron_contrib.npz, phase2c_neuron_heatmap.png,
phase2c_top_neurons.json, and neuron_groups.json (consumed by phase4_ablation).

Group overlap is computed against the cell-type / third partition stored in the
trace dir's group_indices.json (written by eval_dump).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Analysis.osl2d._io import analysis_dir, load_group_indices, load_traces, write_json
from Analysis.osl2d.probe import probe_weights_episode
from Analysis.osl2d.segment import INT_TO_LABEL


def _load_groups(run_dir: Path, ckpt_labels) -> dict:
    """Best-effort load of the group_indices for the first ckpt label."""
    if not ckpt_labels:
        # Discover any ckpt subdir under analysis/traces.
        base = run_dir / "analysis" / "traces"
        subs = [p.name for p in base.glob("*") if (p / "group_indices.json").exists()]
        ckpt_labels = subs[:1]
    for cl in (ckpt_labels or []):
        try:
            gi = load_group_indices(run_dir, cl)
        except Exception:
            continue
        return {k: np.asarray(v, dtype=np.int64)
                for k, v in gi.items() if k not in ("state_size", "backbone")}
    return {}


def _group_membership(top_idx: np.ndarray, group_indices: dict) -> dict:
    out = {}
    for name, idx in group_indices.items():
        if name == "all":
            continue
        idx = np.asarray(idx)
        inter = np.intersect1d(top_idx, idx)
        out[name] = {
            "count": int(inter.size),
            "fraction_of_group": float(inter.size / max(idx.size, 1)),
            "fraction_of_top": float(inter.size / max(top_idx.size, 1)),
        }
    return out


def run(run_dir: Path, ckpt_labels=None, top_k: int = 16, seed: int = 0):
    traces = load_traces(run_dir, ckpt_labels)
    out_dir = analysis_dir(run_dir)
    if traces.h.shape[1] == 0:
        write_json(out_dir / "phase2c_top_neurons.json", {"skipped": "no hidden state."})
        return

    W, classes = probe_weights_episode(
        traces.h, traces.label.astype(int), traces.episode_id, seed=seed,
    )
    if W is None:
        write_json(out_dir / "phase2c_top_neurons.json", {"skipped": "single-class trace."})
        return

    contrib = np.abs(W)
    if contrib.shape[0] == 1 and len(classes) == 2:
        contrib = np.vstack([contrib, contrib])
    np.savez_compressed(out_dir / "phase2c_neuron_contrib.npz",
                        contrib=contrib, classes=classes)

    group_indices = _load_groups(run_dir, ckpt_labels)

    top_per_label = {}
    neuron_groups = {}
    for row_i, cls in enumerate(classes):
        if row_i >= contrib.shape[0]:
            break
        order = np.argsort(-contrib[row_i])
        top = order[:top_k].astype(int)
        name = INT_TO_LABEL.get(int(cls), str(cls))
        top_per_label[name] = {
            "indices": top.tolist(),
            "weights_abs": contrib[row_i, top].tolist(),
            "group_overlap": _group_membership(top, group_indices) if group_indices else {},
        }
        neuron_groups[name] = top.tolist()
    write_json(out_dir / "phase2c_top_neurons.json", {
        "top_k": int(top_k),
        "per_label": top_per_label,
        "classes": [int(c) for c in classes],
    })
    write_json(out_dir / "neuron_groups.json", {
        "from": "phase2c",
        "top_k": int(top_k),
        "groups": neuron_groups,
    })

    overall = contrib.sum(axis=0)
    col_order = np.argsort(-overall)
    M = contrib[:, col_order]
    fig, ax = plt.subplots(figsize=(10, max(2.5, 0.4 * len(classes))))
    im = ax.imshow(M, aspect="auto", cmap="magma")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([INT_TO_LABEL.get(int(c), str(c)) for c in classes])
    ax.set_xlabel("hidden unit (sorted by total |w|)")
    ax.set_title("Phase 2c — neuron contribution by behavior label")
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / "phase2c_neuron_heatmap.png", dpi=150)
    plt.close(fig)


def build_parser():
    p = argparse.ArgumentParser("Analysis.osl2d.phase2c_neuron")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt-labels", nargs="*", default=None)
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    run(Path(a.run_dir), a.ckpt_labels, a.top_k, a.seed)


if __name__ == "__main__":
    main()
