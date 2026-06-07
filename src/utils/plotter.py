"""Rollout GIF rendering utilities — bilateral sensor + head/body aware."""
from __future__ import annotations

import io
import os

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _plume_field(env, grid_mm: float = 0.5):
    """Sample the current odor field (base × current bump perturbation) on the
    same grid the curriculum viz uses (default 0.5 mm/pixel) so the two views
    look consistent."""
    cfg = env.cfg
    xs = np.arange(0.0, cfg.arena_width_mm + grid_mm, grid_mm)
    ys = np.arange(0.0, cfg.arena_height_mm + grid_mm, grid_mm)
    field = np.empty((len(ys), len(xs)), dtype=np.float32)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            field[j, i] = env.field.sample(float(x), float(y))
    return field, cfg.arena_width_mm, cfg.arena_height_mm


# vmax cache keyed by id(env): set on the first call for an env, reused after.
# Matches the curriculum viz behaviour (auto-scale to the first sampled field)
# while keeping the colour scale stable across frames of one rollout.
_RENDER_VMAX_CACHE: dict[int, float] = {}


def render_rollout_frame(env, traj_x, traj_y, cast_x, cast_y, step, title=None):
    """One matplotlib frame: plume + trajectory + high-cast events + source."""
    field, W, H = _plume_field(env)
    cfg = env.cfg

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    cache_key = id(env)
    if cache_key not in _RENDER_VMAX_CACHE:
        _RENDER_VMAX_CACHE[cache_key] = max(1e-6, float(field.max()) * 1.2)
    vmax = _RENDER_VMAX_CACHE[cache_key]
    ax.imshow(field, extent=[0.0, W, 0.0, H], origin="lower", cmap="magma",
              vmin=0.0, vmax=vmax)
    ax.plot(traj_x, traj_y, color="#50dcff", linewidth=2.0, alpha=0.85)
    if cast_x:
        ax.scatter(cast_x, cast_y, color="white", marker="*", s=140,
                   edgecolors="black", zorder=10)
    ax.scatter([cfg.source_x_mm], [cfg.source_y_mm], color="lime", marker="P",
               s=160, zorder=11)
    ax.add_patch(plt.Circle((cfg.source_x_mm, cfg.source_y_mm),
                            cfg.success_radius_mm, color="gray", fill=False))
    ax.set_xlim(0.0, W)
    ax.set_ylim(0.0, H)
    ax.set_aspect("equal")
    ax.tick_params(colors="white")
    ax.set_title(title or f"Step: {step}", color="white")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=80, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    buf.seek(0)
    return np.array(imageio.v2.imread(buf))


def _harmonize_frames(frames):
    """Crop all frames to the common (min) H×W so they stack into a GIF.

    `render_rollout_frame` uses `bbox_inches="tight"`, so frame pixel dimensions
    can drift by a row/column as the title text width changes across steps.
    imageio's stacking then fails with "all input arrays must have the same
    shape". Center-crop every frame to the smallest height/width to fix it.
    """
    if not frames:
        return frames
    min_h = min(f.shape[0] for f in frames)
    min_w = min(f.shape[1] for f in frames)
    if all(f.shape[0] == min_h and f.shape[1] == min_w for f in frames):
        return frames
    out = []
    for f in frames:
        h, w = f.shape[:2]
        top = (h - min_h) // 2
        left = (w - min_w) // 2
        out.append(f[top:top + min_h, left:left + min_w])
    return out


def save_gif(frames, path, fps=15):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, _harmonize_frames(frames), fps=fps, loop=0)
    print(f"[GIF] Saved to {path}")
