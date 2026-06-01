"""Noise sweep: does CAST become more causally important as the plume gets more
spatiotemporally variable?

For each noise strength in a sweep (stage 2 = dynamic bump field):
  1. dump stochastic traces (the policy only solves the task with exploration),
  2. recompute the per-behavior top-k neurons (linear-probe |W|),
  3. measure how much each condition's CAST top-k OVERLAPS the clean (0.0) set
     — i.e. is the CAST module stable across environments?
  4. run zero-ablation two ways:
       - FIXED   : always ablate the clean-condition CAST neurons,
       - ADAPTIVE: ablate each condition's own CAST neurons,
     and record Δsuccess vs. the un-ablated baseline.
Saves a json + two figures. Standalone; run with the osl conda python.

    ~/anaconda3/envs/osl/bin/python Analysis/noise_sweep_cast.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.envs.osl_env import OslEnv
from Analysis.osl2d import eval_dump, phase4_ablation
from Analysis.osl2d._io import load_traces
from Analysis.osl2d.probe import probe_weights_episode
from Analysis.osl2d.segment import LABELS, LABEL_TO_INT

# ---- config (absolute paths; current model = the 2.40M GRU) -----------------
RUN_DIR = Path("/home/hyunseo/Personal_Research/OSL/runs/ppo_gru_nb_20260531_113633")
CKPT_LABEL = "final"
NOISE_STAGE = 2
NOISE_STRENGTHS = [0.0, 0.1, 0.2, 0.3]
SEEDS = (0, 1)
EPISODES_PER_SEED = 8
MAX_STEPS = 1200
TOP_K = 16
DEVICE = "cpu"
ABLATION_EPS = 4                 # rollouts/seed for ablation success estimate
OUT = RUN_DIR / "analysis" / "noise_sweep"
OUT.mkdir(parents=True, exist_ok=True)


def _strength_tag(a: float) -> str:
    return f"n{int(round(a * 100)):02d}"          # 0.2 -> "n20"


def _dump_traces(strength: float, trace_label: str) -> None:
    """Dump stochastic traces for this noise level into a per-noise trace dir.

    Bypasses eval_dump.collect (which resolves the checkpoint from the label, so
    a per-noise label like 'final_n20' would fail). We always load the bare
    `final` checkpoint and only the *trace dir* is noise-specific.
    """
    from Analysis.osl2d.eval_dump import _env_config_from_ckpt, _resolve_ckpt
    from Analysis.osl2d.policy_adapter import Policy2DAdapter
    ckpt = _resolve_ckpt(RUN_DIR, CKPT_LABEL)
    adapter = Policy2DAdapter.from_checkpoint(ckpt, device=DEVICE)
    env_cfg = _env_config_from_ckpt(ckpt, NOISE_STAGE, strength)
    out_dir = RUN_DIR / "analysis" / "traces" / trace_label
    out_dir.mkdir(parents=True, exist_ok=True)
    gi = {k: [int(i) for i in v] for k, v in adapter.node_group_indices.items()}
    gi["state_size"] = int(adapter.n_nodes if adapter.feature_dim > 1 else adapter.state_size)
    gi["feature_dim"] = int(adapter.feature_dim)
    gi["backbone"] = adapter.backbone_kind
    (out_dir / "group_indices.json").write_text(json.dumps(gi, indent=2))
    for s in SEEDS:
        env = OslEnv(env_cfg)
        for e in range(EPISODES_PER_SEED):
            traj = eval_dump.rollout(adapter, env, 10_000 + int(s) * 1000 + e,
                                     MAX_STEPS, stochastic=True)
            np.savez_compressed(
                out_dir / f"eval_seed{s}_ep{e:03d}.npz",
                **traj, episode=e, seed=int(s),
                episode_id=int(s) * 10_000 + e, ckpt_label=trace_label,
                action_mode="stochastic")


def collect_and_topk(strength: float) -> dict:
    """Dump stochastic traces for this noise level (cached) and return
    per-behavior top-k neuron indices from a linear probe."""
    label = f"{CKPT_LABEL}_{_strength_tag(strength)}__stoch"
    trace_dir = RUN_DIR / "analysis" / "traces" / label
    if not (trace_dir.exists() and any(trace_dir.glob("eval_seed*_ep*.npz"))):
        _dump_traces(strength, label)
    traces = load_traces(RUN_DIR, [label])
    W, classes = probe_weights_episode(traces.h, traces.label.astype(int), traces.episode_id, seed=0)
    contrib = np.abs(W)
    if contrib.shape[0] == 1 and len(classes) == 2:
        contrib = np.vstack([contrib, contrib])
    tops = {}
    for row_i, cls in enumerate(classes):
        if row_i >= contrib.shape[0]:
            break
        name = LABELS[int(cls)] if int(cls) < len(LABELS) else str(cls)
        tops[name] = np.argsort(-contrib[row_i])[:TOP_K].astype(int).tolist()
    return {"label": label, "top": tops, "raw_success": traces_success(RUN_DIR, label)}


def _topk_from_traces(traces, label_name: str) -> list[int]:
    """Top-k neurons for one behavior label from a probe over the given traces."""
    W, classes = probe_weights_episode(
        traces.h, traces.label.astype(int), traces.episode_id, seed=0)
    if W is None:
        return []
    contrib = np.abs(W)
    if contrib.shape[0] == 1 and len(classes) == 2:
        contrib = np.vstack([contrib, contrib])
    for row_i, cls in enumerate(classes):
        if row_i >= contrib.shape[0]:
            break
        name = LABELS[int(cls)] if int(cls) < len(LABELS) else str(cls)
        if name == label_name:
            return np.argsort(-contrib[row_i])[:TOP_K].astype(int).tolist()
    return []


def topk_per_seed(strength: float, label_name: str) -> dict:
    """Per-seed top-k neuron sets for one behavior, for a noise level.

    Splits the (cached) stochastic traces by their stored seed and runs a
    separate probe per seed, yielding one top-k set per seed. Used to build a
    *distribution* of Jaccard overlaps (→ box plot) rather than a single scalar.
    Returns {seed: [neuron indices]}.
    """
    label = f"{CKPT_LABEL}_{_strength_tag(strength)}__stoch"
    trace_dir = RUN_DIR / "analysis" / "traces" / label
    if not (trace_dir.exists() and any(trace_dir.glob("eval_seed*_ep*.npz"))):
        _dump_traces(strength, label)
    out = {}
    for s in SEEDS:
        per_seed_files = sorted(trace_dir.glob(f"eval_seed{s}_ep*.npz"))
        if not per_seed_files:
            continue
        # load_traces over a temp single-seed view by reusing load_traces' loader
        # is awkward; instead pool this seed's npz directly.
        hs, labs, eids = [], [], []
        for f in per_seed_files:
            d = np.load(f, allow_pickle=True)
            hs.append(d["h"]); labs.append(d["label"])
            eids.append(np.full(len(d["h"]), int(d["episode_id"]), dtype=np.int64))
        if not hs:
            continue

        class _T:  # minimal duck-typed traces object for the probe
            pass
        t = _T()
        t.h = np.concatenate(hs, axis=0)
        t.label = np.concatenate(labs, axis=0)
        t.episode_id = np.concatenate(eids, axis=0)
        out[int(s)] = _topk_from_traces(t, label_name)
    return out


def traces_success(run_dir: Path, trace_label: str) -> float:
    s = []
    for f in (run_dir / "analysis" / "traces" / trace_label).glob("eval_seed*_ep*.npz"):
        d = np.load(f, allow_pickle=True)
        s.append(int(d["success"]))
    return float(np.mean(s)) if s else 0.0


def jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))


def ablate_cast(strength: float, cast_neurons: list[int]) -> dict:
    """Run live-env ablation of a given neuron set under this noise level,
    returning baseline & ablated success. Reuses phase4's rollout helpers."""
    from Analysis.osl2d.eval_dump import _env_config_from_ckpt, _resolve_ckpt
    from Analysis.osl2d.policy_adapter import Policy2DAdapter
    ckpt = _resolve_ckpt(RUN_DIR, CKPT_LABEL)
    adapter = Policy2DAdapter.from_checkpoint(ckpt, device=DEVICE)
    env = OslEnv(_env_config_from_ckpt(ckpt, NOISE_STAGE, strength))

    def run_set(patch_idx):
        runs = []
        for s in SEEDS:
            for e in range(ABLATION_EPS):
                runs.append(phase4_ablation._rollout(
                    env, adapter, 10_000 + int(s) * 1000 + e,
                    patch_indices=patch_idx, max_steps=MAX_STEPS, stochastic=True))
        return float(np.mean([r["success"] for r in runs]))

    base = run_set(None)
    ablated = run_set(np.asarray(cast_neurons, dtype=np.int64))
    return {"baseline_success": base, "ablated_success": ablated,
            "d_success": ablated - base}


def main():
    print(f"[sweep] model = {RUN_DIR.name}")
    # 1) per-noise top-k
    topk_by_noise = {}
    for a in NOISE_STRENGTHS:
        print(f"[sweep] collect+topk @ strength={a} ...")
        topk_by_noise[a] = collect_and_topk(a)

    clean_cast = topk_by_noise[NOISE_STRENGTHS[0]]["top"].get("CAST", [])

    # 2) overlap of CAST top-k vs clean
    overlap = {a: jaccard(clean_cast, topk_by_noise[a]["top"].get("CAST", []))
               for a in NOISE_STRENGTHS}

    # 3) ablation: fixed (clean cast neurons) and adaptive (own cast neurons)
    fixed, adaptive = {}, {}
    for a in NOISE_STRENGTHS:
        print(f"[sweep] ablation @ strength={a} ...")
        fixed[a] = ablate_cast(a, clean_cast)
        own = topk_by_noise[a]["top"].get("CAST", clean_cast)
        adaptive[a] = ablate_cast(a, own)

    results = {
        "run_dir": str(RUN_DIR), "noise_stage": NOISE_STAGE,
        "strengths": NOISE_STRENGTHS, "top_k": TOP_K,
        "clean_cast_neurons": clean_cast,
        "cast_topk_overlap_vs_clean": overlap,
        "cast_topk_per_noise": {a: topk_by_noise[a]["top"].get("CAST", []) for a in NOISE_STRENGTHS},
        "ablation_fixed": fixed,
        "ablation_adaptive": adaptive,
    }
    (OUT / "noise_sweep_cast.json").write_text(json.dumps(results, indent=2))
    print("[sweep] saved", OUT / "noise_sweep_cast.json")

    xs = NOISE_STRENGTHS
    # Figure 1: CAST ablation Δsuccess vs noise (fixed & adaptive)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(xs, [fixed[a]["d_success"] for a in xs], "o-", label="ablate FIXED (clean cast neurons)")
    ax.plot(xs, [adaptive[a]["d_success"] for a in xs], "s--", label="ablate ADAPTIVE (own cast neurons)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("noise strength (stage 2, dynamic plume)")
    ax.set_ylabel("Δ success when CAST neurons ablated")
    ax.set_title("Causal importance of CAST vs. environmental variability")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "cast_ablation_vs_noise.png", dpi=150)
    plt.close(fig)

    # Figure 2: CAST top-k overlap vs clean
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(xs, [overlap[a] for a in xs], "o-", color="purple")
    ax.set_ylim(0, 1)
    ax.set_xlabel("noise strength (stage 2)")
    ax.set_ylabel("Jaccard overlap of CAST top-k vs clean")
    ax.set_title("Is the CAST neuron module stable across environments?")
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "cast_topk_overlap.png", dpi=150)
    plt.close(fig)
    print("[sweep] figures →", OUT)

    # console summary
    print("\n=== SUMMARY ===")
    print(f"{'noise':>6} | {'overlap':>7} | {'Δsucc(fixed)':>12} | {'Δsucc(adapt)':>12} | {'base→abl(fixed)':>16}")
    for a in xs:
        f = fixed[a]
        print(f"{a:>6.1f} | {overlap[a]:>7.2f} | {f['d_success']:>+12.2f} | "
              f"{adaptive[a]['d_success']:>+12.2f} | {f['baseline_success']:.2f}→{f['ablated_success']:.2f}")


if __name__ == "__main__":
    main()
