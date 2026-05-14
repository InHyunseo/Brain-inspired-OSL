"""Plot CSV outputs from analyze_cartpole_ac.py.

Reads a demo/RL_practice run directory and writes compact analysis figures to
the run's plots/ directory.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Plot CartPole analysis CSV outputs.")
    p.add_argument("--run-dir", required=True, help="Path to a cartpole_ac_* run directory.")
    p.add_argument("--top-units", type=int, default=20)
    return p.parse_args()


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing required file: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def setup_matplotlib(plot_dir: Path):
    cache_dir = plot_dir / ".matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_baseline(rows: list[dict], plot_dir: Path, plt):
    seeds = [int(r["seed"]) for r in rows]
    steps = np.asarray([as_float(r, "steps") for r in rows], dtype=np.float32)
    success = np.asarray([as_float(r, "success") for r in rows], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(rows))
    colors = np.where(success > 0.5, "#2a9d8f", "#e76f51")
    ax.bar(x, steps, color=colors, alpha=0.85)
    ax.axhline(500, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylim(0, max(520, float(steps.max()) * 1.05))
    ax.set_xlabel("eval episode")
    ax.set_ylabel("steps")
    ax.set_title(
        f"Baseline final policy: mean steps={steps.mean():.1f}, "
        f"success={success.mean():.2f}"
    )
    if len(rows) <= 30:
        ax.set_xticks(x)
        ax.set_xticklabels(seeds, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("eval seed")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "analysis_baseline_summary.png", dpi=140)
    plt.close(fig)


def plot_probe(rows: list[dict], plot_dir: Path, plt):
    targets = [r["target"] for r in rows]
    r2 = np.asarray([as_float(r, "r2", float("nan")) for r in rows], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    bars = ax.bar(targets, r2, color=["#457b9d", "#f4a261"][: len(targets)], alpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R2")
    ax.set_title("Linear probe: hidden -> CartPole state")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, r2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(1.02, float(value) + 0.03),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(plot_dir / "analysis_probe_r2.png", dpi=140)
    plt.close(fig)


def plot_group_patches(rows: list[dict], plot_dir: Path, plt):
    grouped = defaultdict(list)
    for r in rows:
        target = r["target"]
        if target.startswith("unit_"):
            continue
        grouped[(target, r["mode"])].append(as_float(r, "delta_steps"))

    targets = ["all_hidden", "first_half", "second_half"]
    modes = ["zero", "mean", "flip"]
    x = np.arange(len(targets))
    width = 0.24

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = {"zero": "#264653", "mean": "#2a9d8f", "flip": "#e76f51"}
    for i, mode in enumerate(modes):
        means = []
        stds = []
        for target in targets:
            vals = np.asarray(grouped.get((target, mode), [0.0]), dtype=np.float32)
            means.append(float(vals.mean()))
            stds.append(float(vals.std()))
        ax.bar(
            x + (i - 1) * width,
            means,
            width,
            yerr=stds,
            label=mode,
            capsize=3,
            color=colors[mode],
            alpha=0.9,
        )
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel("delta steps vs baseline")
    ax.set_title("Hidden patch effect by group")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "analysis_patch_group_delta_steps.png", dpi=140)
    plt.close(fig)


def plot_unit_ablation(rows: list[dict], plot_dir: Path, plt, top_units: int):
    grouped = defaultdict(list)
    for r in rows:
        target = r["target"]
        if not target.startswith("unit_"):
            continue
        grouped[int(as_float(r, "unit"))].append(as_float(r, "delta_steps"))

    if not grouped:
        return

    unit_stats = []
    for unit, vals in grouped.items():
        arr = np.asarray(vals, dtype=np.float32)
        unit_stats.append((unit, float(arr.mean()), float(arr.std())))
    unit_stats.sort(key=lambda x: x[1])

    top = unit_stats[: max(1, int(top_units))]
    labels = [f"{unit}" for unit, _, _ in top]
    means = np.asarray([mean for _, mean, _ in top], dtype=np.float32)
    stds = np.asarray([std for _, _, std in top], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(top))
    ax.barh(y, means, xerr=stds, color="#6d597a", alpha=0.9, capsize=2)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("mean delta steps vs baseline")
    ax.set_ylabel("hidden unit")
    ax.set_title(f"Most disruptive unit-wise zero ablations (top {len(top)})")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "analysis_unit_ablation_top.png", dpi=140)
    plt.close(fig)

    units = np.asarray([unit for unit, _, _ in sorted(unit_stats)], dtype=np.int64)
    all_means = np.asarray([mean for _, mean, _ in sorted(unit_stats)], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(units, all_means, marker=".", lw=1.0, color="#6d597a")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("hidden unit")
    ax.set_ylabel("mean delta steps")
    ax.set_title("Unit-wise zero ablation scan")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "analysis_unit_ablation_scan.png", dpi=140)
    plt.close(fig)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    analysis_dir = run_dir / "analysis"
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib(plot_dir)

    baseline = read_csv(analysis_dir / "baseline_eval.csv")
    probe = read_csv(analysis_dir / "probe_results.csv")
    patch = read_csv(analysis_dir / "patch_results.csv")

    plot_baseline(baseline, plot_dir, plt)
    plot_probe(probe, plot_dir, plt)
    plot_group_patches(patch, plot_dir, plt)
    plot_unit_ablation(patch, plot_dir, plt, args.top_units)

    print(f"figures saved to {plot_dir}")


if __name__ == "__main__":
    main()
