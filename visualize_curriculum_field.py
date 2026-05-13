"""Visualize the bump-field curriculum.

Renders a per-phase static snapshot (PNG) and a temporal GIF that animates the
dynamic phases by stepping `OslEnv` with zero-velocity actions so the field
advances exactly as it does in training.

Usage:
    python visualize_curriculum_field.py
    python visualize_curriculum_field.py --out-dir runs/field_viz --frames 120
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle

# Allow running from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.envs.osl_env import EnvConfig, OslEnv


# Curriculum: (stage, strength, label).
# Stage 0 = clean, 1 = static bumps, 2 = dynamic bumps. Strength is the
# bump-field scaling alpha in [0, 1].
DEFAULT_PHASES = [
    (0, 0.0, "Phase 0: clean"),
    (1, 0.3, "Phase 1: static α=0.3"),
    (2, 0.3, "Phase 2: dynamic α=0.3"),
    (2, 0.6, "Phase 3: dynamic α=0.6"),
    (2, 1.0, "Phase 4: dynamic α=1.0"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="runs/field_viz")
    p.add_argument("--grid-mm", type=float, default=0.5,
                   help="Heatmap pixel size (mm).")
    p.add_argument("--frames", type=int, default=120,
                   help="Animation frames per dynamic phase.")
    p.add_argument("--interval-ms", type=int, default=80)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--arena-width-mm", type=float, default=80.0)
    p.add_argument("--arena-height-mm", type=float, default=120.0)
    p.add_argument("--source-x-mm", type=float, default=40.0)
    p.add_argument("--source-y-mm", type=float, default=100.0)
    p.add_argument("--gaussian-sigma-mm", type=float, default=30.0)
    p.add_argument("--success-radius-mm", type=float, default=7.5)
    return p.parse_args()


def env_kw(args):
    return dict(
        arena_width_mm=args.arena_width_mm,
        arena_height_mm=args.arena_height_mm,
        source_x_mm=args.source_x_mm,
        source_y_mm=args.source_y_mm,
        gaussian_sigma_mm=args.gaussian_sigma_mm,
        success_radius_mm=args.success_radius_mm,
    )


def make_env(args, stage, strength):
    cfg = EnvConfig.from_dict({
        **env_kw(args),
        "noise_stage": int(stage),
        "noise_strength": float(strength),
        "seed": args.seed,
    })
    return OslEnv(cfg)


def sample_grid(env, grid_mm):
    W = env.cfg.arena_width_mm
    H = env.cfg.arena_height_mm
    xs = np.arange(0.0, W + grid_mm, grid_mm)
    ys = np.arange(0.0, H + grid_mm, grid_mm)
    out = np.empty((len(ys), len(xs)), dtype=np.float32)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            out[j, i] = env.field.sample(float(x), float(y))
    return xs, ys, out


def decorate(ax, env, title, vmax_label="c"):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    sx, sy = env.cfg.source_x_mm, env.cfg.source_y_mm
    ax.plot([sx], [sy], marker="*", color="white", markersize=14,
            markeredgecolor="black", linewidth=0)
    ax.add_patch(Circle((sx, sy), env.cfg.success_radius_mm, fill=False,
                        edgecolor="white", linestyle="--", linewidth=1.0))


def render_snapshots(args, phases, out_path):
    n = len(phases)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 5.2), squeeze=False)
    for k, (stage, strength, label) in enumerate(phases):
        env = make_env(args, stage, strength)
        # Reset once so spawn / heading are realised; we don't visualize the
        # agent but reset() also re-seeds the bump field at this strength.
        env.reset(seed=args.seed + k)
        xs, ys, grid = sample_grid(env, args.grid_mm)
        ax = axes[0, k]
        im = ax.imshow(grid, origin="lower",
                       extent=[xs[0], xs[-1], ys[0], ys[-1]],
                       cmap="magma", vmin=0.0)
        decorate(ax, env, label)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="c")
    fig.suptitle("Bump-field curriculum — snapshots", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[snapshot] {out_path}")


def render_animation(args, phases, out_path):
    """Animate every phase side-by-side; static phases stay frozen, dynamic
    phases evolve via env.step() with zero-velocity actions so the bump
    AR(1) + drift dynamics fire exactly like in training."""
    envs = []
    for k, (stage, strength, _label) in enumerate(phases):
        env = make_env(args, stage, strength)
        env.reset(seed=args.seed + 100 + k)
        envs.append(env)

    n = len(phases)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 5.2), squeeze=False)
    ims = []
    for k, env in enumerate(envs):
        xs, ys, grid = sample_grid(env, args.grid_mm)
        ax = axes[0, k]
        # Cap vmax slightly above the initial peak to keep the colorbar stable
        # as dynamics evolve.
        vmax = max(1e-6, float(grid.max()) * 1.4)
        im = ax.imshow(grid, origin="lower",
                       extent=[xs[0], xs[-1], ys[0], ys[-1]],
                       cmap="magma", vmin=0.0, vmax=vmax)
        decorate(ax, env, phases[k][2])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="c")
        ims.append(im)

    # Zero-velocity action (raw_v=-1 maps to v=0) — agent stays put while the
    # field advances at the env's natural rate.
    zero_action = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

    def update(frame_idx):
        for k, env in enumerate(envs):
            stage, _strength, _ = phases[k]
            if stage >= 2:
                env.step(zero_action)
            _, _, g = sample_grid(env, args.grid_mm)
            ims[k].set_data(g)
        return ims

    anim = animation.FuncAnimation(
        fig, update, frames=args.frames, interval=args.interval_ms, blit=False)

    try:
        anim.save(out_path, writer=animation.PillowWriter(fps=int(round(1000 / args.interval_ms))))
        print(f"[gif] {out_path}")
    except Exception as e:
        print(f"[gif] save failed ({e}); falling back to MP4 if available")
        mp4_path = os.path.splitext(out_path)[0] + ".mp4"
        anim.save(mp4_path, writer=animation.FFMpegWriter(fps=int(round(1000 / args.interval_ms))))
        print(f"[mp4] {mp4_path}")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    phases = DEFAULT_PHASES
    render_snapshots(args, phases, os.path.join(args.out_dir, "snapshots.png"))
    render_animation(args, phases, os.path.join(args.out_dir, "curriculum.gif"))


if __name__ == "__main__":
    main()
