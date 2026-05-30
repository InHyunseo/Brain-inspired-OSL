"""Phase 3a — Jacobian eigenmode analysis per behavior label.

Loads the policy adapter for each checkpoint, samples S timesteps per label,
computes J_t = d h_{t+1}/d h_t at each, and pools eigenvalues in the complex
plane. Ports osl_analysis ``phase3a_jacobian.py`` with the 3D ROS2 adapter
swapped for :class:`Analysis.osl2d.policy_adapter.Policy2DAdapter`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Analysis.osl2d._io import adapter_for_ckpt, analysis_dir, load_traces, write_json
from Analysis.osl2d.jacobian import jacobian_at, dominant_summary
from Analysis.osl2d.segment import INT_TO_LABEL

_CMAP = {0: "#d62728", 1: "#1f77b4", 2: "#9467bd", 3: "#7f7f7f", 4: "#2ca02c"}


def run(run_dir: Path, ckpt_labels=None, samples_per_label: int = 200, seed: int = 0,
        device: str | None = None):
    traces = load_traces(run_dir, ckpt_labels)
    out_dir = analysis_dir(run_dir)
    if traces.h.shape[1] == 0:
        write_json(out_dir / "phase3a_summary.json", {"skipped": "no hidden state."})
        return

    rng = np.random.default_rng(seed)
    ckpts = sorted(set(traces.ckpt_label.tolist()))

    summary = {
        "caveat": "Local linearization J_t = d h_{t+1}/d h_t around observed h_t. "
                  "Nonlinear regime not captured.",
        "per_checkpoint": {},
    }
    for cl in ckpts:
        try:
            adapter = adapter_for_ckpt(run_dir, cl, device=device)
        except FileNotFoundError as exc:
            summary["per_checkpoint"][cl] = {"skipped": str(exc)}
            continue
        cm = (traces.ckpt_label == cl)
        if cm.sum() < 50:
            continue

        per_label = {}
        all_eigs, all_labs = [], []
        for lab_int, lab_name in INT_TO_LABEL.items():
            mask = cm & (traces.label == lab_int)
            n = int(mask.sum())
            if n == 0:
                per_label[lab_name] = {"n": 0}
                continue
            idx_pool = np.where(mask)[0]
            pick = rng.choice(idx_pool, size=min(samples_per_label, n), replace=False)
            top_abs, top_arg, n_osc, n_slow = [], [], 0, 0
            for t in pick:
                J = jacobian_at(adapter, traces.obs[t], traces.h[t])
                w = np.linalg.eigvals(J)
                d = dominant_summary(w)
                top_abs.append(d["top_abs"])
                top_arg.append(d["top_arg"])
                n_osc += int(d["is_oscillatory"])
                n_slow += int(d["n_slow_real"] > 0)
                all_eigs.append(w)
                all_labs.append(lab_int)
            per_label[lab_name] = {
                "n_samples": int(len(pick)),
                "top_abs_mean": float(np.mean(top_abs)),
                "top_abs_p90": float(np.percentile(top_abs, 90)),
                "top_arg_mean_rad": float(np.mean(top_arg)),
                "oscillatory_fraction": float(n_osc / len(pick)),
                "slow_real_fraction": float(n_slow / len(pick)),
            }
        summary["per_checkpoint"][cl] = per_label

        if all_eigs:
            E = np.concatenate(all_eigs)
            L = np.repeat(all_labs, [len(e) for e in all_eigs])
            fig, ax = plt.subplots(figsize=(6, 5))
            for lab_int, lab_name in INT_TO_LABEL.items():
                m = (L == lab_int)
                if m.sum() == 0:
                    continue
                ax.scatter(E[m].real, E[m].imag, s=4, alpha=0.5,
                           color=_CMAP.get(lab_int, "k"), label=lab_name)
            theta = np.linspace(0, 2 * np.pi, 360)
            ax.plot(np.cos(theta), np.sin(theta), color="black", lw=0.5, alpha=0.4)
            ax.axhline(0, color="k", lw=0.3, alpha=0.4)
            ax.axvline(0, color="k", lw=0.3, alpha=0.4)
            ax.set_xlabel("Re(λ)"); ax.set_ylabel("Im(λ)")
            ax.set_title(f"Jacobian eigenvalues ({cl})")
            ax.legend(markerscale=2, fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / f"phase3a_eigvals_{cl}.png", dpi=150)
            plt.close(fig)

    write_json(out_dir / "phase3a_summary.json", summary)


def build_parser():
    p = argparse.ArgumentParser("Analysis.osl2d.phase3a_jacobian")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt-labels", nargs="*", default=None)
    p.add_argument("--samples-per-label", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    run(Path(a.run_dir), a.ckpt_labels, a.samples_per_label, a.seed, a.device)


if __name__ == "__main__":
    main()
