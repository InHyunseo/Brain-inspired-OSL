"""Phase 3b — Fixed/slow point search per behavior label.

For each (ckpt_label, label): seed L-BFGS from the mean observed hidden state
under a representative (mean) observation, minimize ||F(h, obs) - h||^2, then
classify each found point's stability by its Jacobian eigenvalues.

Ports osl_analysis ``phase3b_fixedpoint.py`` with the 2D adapter.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from Analysis.osl2d._io import adapter_for_ckpt, analysis_dir, load_traces, write_json
from Analysis.osl2d.jacobian import jacobian_at, dominant_summary
from Analysis.osl2d.segment import LABELS, INT_TO_LABEL


def _find_fixed_point(adapter, obs_np, h0_np, max_iter: int = 300, tol: float = 1e-8):
    device = adapter.device
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    h = torch.tensor(h0_np, dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([h], max_iter=max_iter, tolerance_grad=tol,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        _, _, h_next = adapter.forward(obs_t, h)
        loss = ((h_next.squeeze(0) - h) ** 2).sum()
        loss.backward()
        return loss

    final_loss = opt.step(closure)
    h_final = h.detach().cpu().numpy()
    speed = float(final_loss.detach().cpu()) if final_loss is not None else float("nan")
    return h_final, speed


def run(run_dir: Path, ckpt_labels=None, n_starts: int = 3, seed: int = 0,
        device: str | None = None):
    traces = load_traces(run_dir, ckpt_labels)
    out_dir = analysis_dir(run_dir)
    if traces.h.shape[1] == 0:
        write_json(out_dir / "phase3b_fixedpoints.json", {"skipped": "no hidden state."})
        return

    rng = np.random.default_rng(seed)
    ckpts = sorted(set(traces.ckpt_label.tolist()))

    results = {"per_checkpoint": {}}
    for cl in ckpts:
        try:
            adapter = adapter_for_ckpt(run_dir, cl, device=device)
        except FileNotFoundError as exc:
            results["per_checkpoint"][cl] = {"skipped": str(exc)}
            continue
        cm = (traces.ckpt_label == cl)
        per_label = {}
        for lab_int, lab_name in INT_TO_LABEL.items():
            mask = cm & (traces.label == lab_int)
            n = int(mask.sum())
            if n < 5:
                per_label[lab_name] = {"n": n}
                continue
            obs_mean = traces.obs[mask].mean(axis=0).astype(np.float32)
            h_mean = traces.h[mask].mean(axis=0).astype(np.float32)
            points = []
            for _ in range(int(n_starts)):
                jitter = rng.normal(scale=0.1, size=h_mean.shape).astype(np.float32)
                h0 = h_mean + jitter
                h_star, speed = _find_fixed_point(adapter, obs_mean, h0)
                J = jacobian_at(adapter, obs_mean, h_star)
                w = np.linalg.eigvals(J)
                d = dominant_summary(w)
                points.append({
                    "h_star_norm": float(np.linalg.norm(h_star)),
                    "residual_speed_sq": speed,
                    **{k: v for k, v in d.items()},
                })
            per_label[lab_name] = {"n_samples_used_to_seed": n, "points": points}
        results["per_checkpoint"][cl] = per_label

    write_json(out_dir / "phase3b_fixedpoints.json", results)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    width = 0.18
    xs = np.arange(len(LABELS))
    for i, cl in enumerate(ckpts):
        per_label = results["per_checkpoint"].get(cl, {})
        ys = []
        for lab_name in LABELS:
            pts = per_label.get(lab_name, {}).get("points", [])
            if not pts:
                ys.append(np.nan); continue
            best = min(pts, key=lambda p: p["residual_speed_sq"])
            ys.append(best["top_abs"])
        ax.bar(xs + i * width, ys, width=width, label=cl)
    ax.set_xticks(xs + width * (len(ckpts) - 1) / 2)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("|λ_top| at best fixed/slow point")
    ax.axhline(1.0, color="red", lw=0.8, ls="--", alpha=0.6, label="|λ|=1")
    ax.set_title("Phase 3b — stability of fixed/slow points")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "phase3b_fixedpoint_summary.png", dpi=150)
    plt.close(fig)


def build_parser():
    p = argparse.ArgumentParser("Analysis.osl2d.phase3b_fixedpoint")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt-labels", nargs="*", default=None)
    p.add_argument("--n-starts", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    run(Path(a.run_dir), a.ckpt_labels, a.n_starts, a.seed, a.device)


if __name__ == "__main__":
    main()
