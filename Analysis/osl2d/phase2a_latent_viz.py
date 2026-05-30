"""Phase 2a — PCA → UMAP of hidden states, colored by behavior label.

Ported from osl_analysis ``phase2a_latent_viz.py`` to the 2D label set.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from Analysis.osl2d._io import analysis_dir, load_traces, write_json
from Analysis.osl2d.segment import INT_TO_LABEL

_CMAP = {0: "#d62728", 1: "#1f77b4", 2: "#9467bd", 3: "#7f7f7f", 4: "#2ca02c"}


def _umap_or_pca2(X: np.ndarray, seed: int = 0) -> np.ndarray:
    try:
        import umap  # type: ignore
        reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=30, min_dist=0.1)
        return reducer.fit_transform(X)
    except Exception as exc:
        print(f"[phase2a] UMAP unavailable ({exc}); falling back to PCA-2.")
        return PCA(n_components=2, random_state=seed).fit_transform(X)


def run(run_dir: Path, ckpt_labels=None, max_points: int = 30000, seed: int = 0):
    traces = load_traces(run_dir, ckpt_labels)
    out_dir = analysis_dir(run_dir)
    H = traces.h
    if H.shape[1] == 0:
        write_json(out_dir / "phase2a_separation.json",
                   {"skipped": "no recurrent hidden state."})
        return

    rng = np.random.default_rng(seed)
    if len(H) > max_points:
        sub = rng.choice(len(H), size=max_points, replace=False)
        H, y, ck = H[sub], traces.label[sub], traces.ckpt_label[sub]
    else:
        y, ck = traces.label, traces.ckpt_label

    n_pca = min(50, H.shape[1])
    Z = PCA(n_components=n_pca, random_state=seed).fit_transform(H)
    Z2 = _umap_or_pca2(Z, seed=seed)

    sep = {}
    if len(np.unique(y)) > 1:
        try:
            sep["silhouette"] = float(silhouette_score(Z2, y, sample_size=min(5000, len(y))))
            sep["calinski_harabasz"] = float(calinski_harabasz_score(Z2, y))
        except Exception as exc:
            sep["error"] = str(exc)
    write_json(out_dir / "phase2a_separation.json", {
        "n_points": int(len(y)),
        "pca_components": int(n_pca),
        "labels_present": sorted(int(v) for v in np.unique(y)),
        **sep,
    })

    ckpts = sorted(set(traces.ckpt_label.tolist()))
    ck_str = np.array([str(t) for t in ck])
    for cl in (ckpts or ["all"]):
        mask = np.ones(len(y), dtype=bool) if cl == "all" else (ck_str == cl)
        if mask.sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        for lab_int, lab_name in INT_TO_LABEL.items():
            m = mask & (y == lab_int)
            if m.sum() == 0:
                continue
            ax.scatter(Z2[m, 0], Z2[m, 1], s=4, alpha=0.5,
                       color=_CMAP.get(lab_int, "k"), label=lab_name)
        ax.set_title(f"Hidden-state 2D map ({cl})")
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
        ax.legend(markerscale=2, fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"phase2a_latent_{cl}.png", dpi=150)
        plt.close(fig)


def build_parser():
    p = argparse.ArgumentParser("Analysis.osl2d.phase2a_latent_viz")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt-labels", nargs="*", default=None)
    p.add_argument("--max-points", type=int, default=30000)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    run(Path(a.run_dir), a.ckpt_labels, a.max_points, a.seed)


if __name__ == "__main__":
    main()
