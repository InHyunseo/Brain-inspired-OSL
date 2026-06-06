"""PPO-GRU active-sensing top-k PCA across stage-2 noise strengths.

For stage 2 alpha = 0.3, 0.4, 0.5:
  1. dump/cache stochastic traces,
  2. fit an episode-split linear probe,
  3. take the top ~10% GRU units for ACTIVE_SENSING,
  4. draw PCA views of those units.

Outputs are written to:
    runs/ppo_gru_nb_20260531_113633/analysis/noise_sweep/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Analysis.osl2d import eval_dump
from Analysis.osl2d._io import load_traces
from Analysis.osl2d.eval_dump import _env_config_from_ckpt, _resolve_ckpt
from Analysis.osl2d.policy_adapter import Policy2DAdapter
from Analysis.osl2d.probe import probe_weights_episode
from Analysis.osl2d.segment import LABELS, LABEL_TO_INT
from src.envs.osl_env import OslEnv


DEFAULT_RUN_DIR = Path("/home/hyunseo/Personal_Research/OSL/runs/ppo_gru_nb_20260531_113633")
CKPT_LABEL = "final"
NOISE_STAGE = 2
STRENGTHS = (0.3, 0.4, 0.5)
LABEL_COLORS = {
    "RUN": "#1f77b4",
    "ACTIVE_SENSING": "#d62728",
    "OTHER": "#9e9e9e",
}
NOISE_COLORS = {
    0.3: "#2e8b57",
    0.4: "#9467bd",
    0.5: "#d65f5f",
}


def strength_tag(strength: float) -> str:
    return f"n{int(round(float(strength) * 100)):02d}"


def trace_label(strength: float) -> str:
    return f"{CKPT_LABEL}_{strength_tag(strength)}__stoch"


def ensure_traces(
    run_dir: Path,
    strength: float,
    *,
    seeds: tuple[int, ...],
    episodes_per_seed: int,
    max_steps: int,
    device: str,
) -> None:
    """Cache stochastic traces under a per-noise label.

    `eval_dump.collect` resolves checkpoints from the trace label, so for labels
    such as `final_n40__stoch` we inline the tiny custom dump loop and always load
    the bare `final` checkpoint.
    """
    label = trace_label(strength)
    out_dir = run_dir / "analysis" / "traces" / label
    expected = len(seeds) * int(episodes_per_seed)
    existing = list(out_dir.glob("eval_seed*_ep*.npz")) if out_dir.exists() else []
    if len(existing) >= expected:
        print(f"[cache] {label}: {len(existing)} traces")
        return

    ckpt = _resolve_ckpt(run_dir, CKPT_LABEL)
    adapter = Policy2DAdapter.from_checkpoint(ckpt, device=device)
    env_cfg = _env_config_from_ckpt(ckpt, NOISE_STAGE, strength)
    out_dir.mkdir(parents=True, exist_ok=True)

    gi = {k: [int(i) for i in v] for k, v in adapter.node_group_indices.items()}
    gi["state_size"] = int(adapter.n_nodes if adapter.feature_dim > 1 else adapter.state_size)
    gi["feature_dim"] = int(adapter.feature_dim)
    gi["backbone"] = adapter.backbone_kind
    (out_dir / "group_indices.json").write_text(json.dumps(gi, indent=2), encoding="utf-8")

    print(f"[dump] {label}: writing {expected} traces")
    for seed_group in seeds:
        env = OslEnv(env_cfg)
        for ep in range(int(episodes_per_seed)):
            ep_seed = 10_000 + int(seed_group) * 1000 + ep
            traj = eval_dump.rollout(adapter, env, ep_seed, max_steps, stochastic=True)
            np.savez_compressed(
                out_dir / f"eval_seed{seed_group}_ep{ep:03d}.npz",
                **traj,
                episode=int(ep),
                seed=int(seed_group),
                episode_id=int(seed_group) * 10_000 + int(ep),
                ckpt_label=label,
                action_mode="stochastic",
            )
            print(
                f"[dump] {label} seed={seed_group} ep={ep:03d} "
                f"T={len(traj['reward']):4d} success={int(traj['success'])}"
            )
        env.close()


def active_sensing_topk(traces, top_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (top_indices, absolute_probe_weights) for ACTIVE_SENSING."""
    W, classes = probe_weights_episode(traces.h, traces.label.astype(int), traces.episode_id, seed=seed)
    if W is None:
        raise RuntimeError("Linear probe could not be fit: trace has fewer than two labels.")

    active_id = LABEL_TO_INT["ACTIVE_SENSING"]
    if active_id not in set(int(c) for c in classes):
        raise RuntimeError("ACTIVE_SENSING label is absent in this trace set.")

    if W.shape[0] == 1:
        weights = W[0]
    else:
        row = int(np.where(classes == active_id)[0][0])
        weights = W[row]

    contrib = np.abs(weights)
    top_k = max(1, int(round(float(top_frac) * traces.h.shape[1])))
    top = np.argsort(-contrib)[:top_k].astype(int)
    return top, contrib


def balanced_sample(labels: np.ndarray, max_per_label: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        if len(idx) > max_per_label:
            idx = rng.choice(idx, size=max_per_label, replace=False)
        keep.append(idx)
    return np.concatenate(keep) if keep else np.asarray([], dtype=int)


def pca2(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xs = StandardScaler().fit_transform(np.asarray(X, dtype=np.float32))
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(Xs)
    return Z, pca.explained_variance_ratio_


def plot_condition_pcas(rows: dict[float, dict], out_path: Path, max_per_label: int) -> None:
    fig, axes = plt.subplots(1, len(rows), figsize=(18, 5), sharex=False, sharey=False)
    if len(rows) == 1:
        axes = [axes]

    for ax, (strength, row) in zip(axes, rows.items()):
        traces = row["traces"]
        top = row["top"]
        sample_idx = balanced_sample(traces.label, max_per_label=max_per_label, seed=7)
        Z, evr = pca2(traces.h[sample_idx][:, top])
        labs = traces.label[sample_idx]

        for label_name, label_id in LABEL_TO_INT.items():
            m = labs == label_id
            if not np.any(m):
                continue
            alpha = 0.85 if label_name == "ACTIVE_SENSING" else 0.35
            size = 10 if label_name == "ACTIVE_SENSING" else 6
            ax.scatter(
                Z[m, 0],
                Z[m, 1],
                s=size,
                c=LABEL_COLORS[label_name],
                alpha=alpha,
                edgecolors="none",
                label=label_name.replace("_", " ").lower(),
            )

        ratio = float(np.mean(traces.label == LABEL_TO_INT["ACTIVE_SENSING"]))
        ax.set_title(f"stage 2 alpha={strength:.1f}\ntop-{len(top)} AS units, AS={ratio:.1%}")
        ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}%)")
        ax.grid(alpha=0.25)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=3, frameon=False)
    fig.suptitle("PPO-GRU active-sensing top-k PCA by noise condition", y=0.98, fontsize=16)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.92))
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_shared_active_pca(rows: dict[float, dict], out_path: Path, max_points: int) -> None:
    union = sorted({int(i) for row in rows.values() for i in row["top"]})
    X_l, a_l = [], []
    rng = np.random.default_rng(11)
    for strength, row in rows.items():
        traces = row["traces"]
        idx = np.flatnonzero(traces.label == LABEL_TO_INT["ACTIVE_SENSING"])
        if len(idx) > max_points:
            idx = rng.choice(idx, size=max_points, replace=False)
        X_l.append(traces.h[idx][:, union])
        a_l.append(np.full(len(idx), float(strength)))
    X = np.concatenate(X_l, axis=0)
    alpha = np.concatenate(a_l, axis=0)
    Z, evr = pca2(X)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for strength in rows:
        m = alpha == float(strength)
        ax.scatter(
            Z[m, 0],
            Z[m, 1],
            s=12,
            c=NOISE_COLORS.get(float(strength), "#333333"),
            alpha=0.65,
            edgecolors="none",
            label=f"stage2 alpha={strength:.1f}",
        )
    ax.set_title(f"Active-sensing states in shared top-k union subspace\nunion={len(union)} GRU units")
    ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}%)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_overlap(rows: dict[float, dict], out_path: Path) -> np.ndarray:
    strengths = list(rows)
    M = np.zeros((len(strengths), len(strengths)), dtype=float)
    for i, a in enumerate(strengths):
        sa = set(int(x) for x in rows[a]["top"])
        for j, b in enumerate(strengths):
            sb = set(int(x) for x in rows[b]["top"])
            M[i, j] = len(sa & sb) / max(1, len(sa | sb))

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap="viridis")
    labels = [f"a={s:.1f}" for s in strengths]
    ax.set_xticks(range(len(strengths)), labels=labels)
    ax.set_yticks(range(len(strengths)), labels=labels)
    ax.set_title("Top-k active-sensing neuron overlap")
    for i in range(len(strengths)):
        for j in range(len(strengths)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Jaccard")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return M


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--strengths", type=float, nargs="+", default=list(STRENGTHS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--episodes-per-seed", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--device", default="cpu")
    p.add_argument("--top-frac", type=float, default=0.10)
    p.add_argument("--probe-seed", type=int, default=0)
    p.add_argument("--max-per-label", type=int, default=2500)
    p.add_argument("--max-active-per-noise", type=int, default=4000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = run_dir / "analysis" / "noise_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    strengths = tuple(float(s) for s in args.strengths)
    seeds = tuple(int(s) for s in args.seeds)
    for strength in strengths:
        ensure_traces(
            run_dir,
            strength,
            seeds=seeds,
            episodes_per_seed=int(args.episodes_per_seed),
            max_steps=int(args.max_steps),
            device=args.device,
        )

    rows: dict[float, dict] = {}
    for strength in strengths:
        label = trace_label(strength)
        traces = load_traces(run_dir, [label])
        top, contrib = active_sensing_topk(traces, args.top_frac, args.probe_seed)
        rows[strength] = {"label": label, "traces": traces, "top": top, "contrib": contrib}
        print(
            f"[topk] stage2 alpha={strength:.1f}: hidden={traces.h.shape[1]} "
            f"top_k={len(top)} AS_ratio={np.mean(traces.label == LABEL_TO_INT['ACTIVE_SENSING']):.2%}"
        )

    top_k = len(next(iter(rows.values()))["top"])
    pca_by_noise = out_dir / f"active_sensing_top{top_k}_pca_by_noise.png"
    pca_shared = out_dir / f"active_sensing_top{top_k}_pca_shared_active_only.png"
    overlap_png = out_dir / f"active_sensing_top{top_k}_overlap.png"
    json_path = out_dir / f"active_sensing_top{top_k}_pca_summary.json"

    plot_condition_pcas(rows, pca_by_noise, max_per_label=int(args.max_per_label))
    plot_shared_active_pca(rows, pca_shared, max_points=int(args.max_active_per_noise))
    overlap = plot_overlap(rows, overlap_png)

    summary = {
        "run_dir": str(run_dir),
        "noise_stage": NOISE_STAGE,
        "strengths": list(strengths),
        "top_frac": float(args.top_frac),
        "top_k": int(top_k),
        "label": "ACTIVE_SENSING",
        "trace_labels": {str(k): v["label"] for k, v in rows.items()},
        "active_sensing_ratio": {
            str(k): float(np.mean(v["traces"].label == LABEL_TO_INT["ACTIVE_SENSING"]))
            for k, v in rows.items()
        },
        "top_indices": {str(k): [int(i) for i in v["top"]] for k, v in rows.items()},
        "top_weight_abs": {
            str(k): [float(v["contrib"][i]) for i in v["top"]]
            for k, v in rows.items()
        },
        "jaccard_overlap": overlap.tolist(),
        "outputs": {
            "pca_by_noise": str(pca_by_noise),
            "pca_shared_active_only": str(pca_shared),
            "overlap": str(overlap_png),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[saved] {pca_by_noise}")
    print(f"[saved] {pca_shared}")
    print(f"[saved] {overlap_png}")
    print(f"[saved] {json_path}")


if __name__ == "__main__":
    main()
