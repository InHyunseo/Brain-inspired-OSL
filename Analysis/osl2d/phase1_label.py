"""Phase 1 — behavior label distribution, transition matrix, per-episode
composition, and cast-frequency PSD (2D-specific).

Ports osl_analysis ``phase1_label.py`` to the 2D label set, and adds a Welch
PSD of the head-relative-angle timeseries (concatenated over CAST segments) to
estimate the cast oscillation frequency — the 2D analogue of the 3D cast
metric. Compare the peak against larva ethograms (~1-3 Hz).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Analysis.osl2d._io import analysis_dir, load_traces, write_json
from Analysis.osl2d.segment import (
    LABELS, LABEL_TO_INT, N_LABELS, segment_ratios, transition_matrix,
)

_DT = 0.1  # env step seconds (EnvConfig.dt)
_LABEL_COLORS = ["#d62728", "#1f77b4", "#9467bd", "#7f7f7f", "#2ca02c"]


def _cast_psd(head_rel: np.ndarray, label: np.ndarray) -> dict:
    """Welch PSD peak of head-relative-angle over CAST-labeled steps."""
    try:
        from scipy.signal import welch
    except Exception:
        return {"peak_hz": None, "freqs": [], "power": []}
    mask = label == LABEL_TO_INT["CAST"]
    if mask.sum() < 16:
        return {"peak_hz": None, "freqs": [], "power": []}
    sig = head_rel[mask].astype(np.float64)
    sig = sig - sig.mean()
    nper = int(min(128, len(sig)))
    f, P = welch(sig, fs=1.0 / _DT, nperseg=nper)
    peak = float(f[int(np.argmax(P))]) if len(f) else None
    return {"peak_hz": peak, "freqs": f.tolist(), "power": P.tolist()}


def run(run_dir: Path, ckpt_labels=None):
    traces = load_traces(run_dir, ckpt_labels)
    out_dir = analysis_dir(run_dir)

    overall_ratios = segment_ratios(traces.label)
    M = transition_matrix(traces.label, n=N_LABELS)
    row_sum = M.sum(axis=1, keepdims=True).clip(min=1)
    M_norm = M / row_sum

    n_eps = len(traces.episode_lengths)
    per_ep = np.zeros((n_eps, N_LABELS), dtype=np.float32)
    pos = 0
    for ep_idx, T in enumerate(traces.episode_lengths):
        lab = traces.label[pos:pos + T]
        for i in range(N_LABELS):
            per_ep[ep_idx, i] = float(np.mean(lab == i)) if T else 0.0
        pos += T

    head_rel = traces.kin("head_relative_angle_rad")
    psd = _cast_psd(head_rel, traces.label)

    payload = {
        "n_episodes": int(n_eps),
        "n_steps": int(len(traces.label)),
        "overall_ratios": overall_ratios,
        "transition_matrix_counts": M.tolist(),
        "transition_matrix_normalized": M_norm.tolist(),
        "label_order": list(LABELS),
        "cast_psd_peak_hz": psd["peak_hz"],
        "success_rate": float(np.mean(traces.success)) if len(traces.success) else 0.0,
        "checkpoints": sorted(set(map(str, traces.ckpt_label.tolist()))),
    }
    write_json(out_dir / "phase1_label_stats.json", payload)

    # Distribution + transition heatmap
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(LABELS, [overall_ratios[name] for name in LABELS], color="steelblue")
    axes[0].set_ylabel("fraction of steps")
    axes[0].set_title("Behavior label distribution")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=0.3)

    im = axes[1].imshow(M_norm, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_xticks(range(N_LABELS)); axes[1].set_xticklabels(LABELS, rotation=30)
    axes[1].set_yticks(range(N_LABELS)); axes[1].set_yticklabels(LABELS)
    axes[1].set_xlabel("next"); axes[1].set_ylabel("current")
    axes[1].set_title("Label transition matrix (row-normalized)")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "phase1_label_distribution.png", dpi=150)
    plt.close(fig)

    # Per-episode stacked area
    fig, ax = plt.subplots(figsize=(10, 3.5))
    xs = np.arange(n_eps)
    ax.stackplot(xs, per_ep.T, labels=LABELS, colors=_LABEL_COLORS)
    ax.set_xlim(0, max(1, n_eps - 1))
    ax.set_ylim(0, 1)
    ax.set_xlabel("episode index (concatenation order)")
    ax.set_ylabel("fraction")
    ax.set_title("Per-episode behavior composition")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "phase1_per_episode_composition.png", dpi=150)
    plt.close(fig)

    # Cast PSD
    if psd["peak_hz"] is not None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.semilogy(psd["freqs"], np.asarray(psd["power"]) + 1e-12, color="#9467bd")
        ax.axvline(psd["peak_hz"], color="red", ls="--",
                   label=f"peak {psd['peak_hz']:.2f} Hz")
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("PSD (head-rel angle, CAST)")
        ax.set_title("Cast oscillation spectrum")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "phase1_cast_psd.png", dpi=150)
        plt.close(fig)

    return payload


def build_parser():
    p = argparse.ArgumentParser("Analysis.osl2d.phase1_label")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt-labels", nargs="*", default=None)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    run(Path(a.run_dir), a.ckpt_labels)


if __name__ == "__main__":
    main()
