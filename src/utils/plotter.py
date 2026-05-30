"""Plotting + GIF rendering utilities."""
from __future__ import annotations

import csv
import io
import json
import os

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Training-curve plotting (RSAC episode-loop)
# ---------------------------------------------------------------------------


def _plot_data_dir(run_dir):
    return os.path.join(run_dir, "plot_data")


def _to_float_list(values):
    return [float(v) for v in values]


def _ema(arr, alpha):
    out = []
    m = arr[0] if arr else 0.0
    for x in arr:
        m = alpha * float(x) + (1.0 - alpha) * m
        out.append(float(m))
    return out


def _calc_ylim(values):
    if not values:
        return [-1.0, 1.0]
    vmin, vmax = float(np.min(values)), float(np.max(values))
    margin = max(1.0, abs(vmax) * 0.1) if np.isclose(vmin, vmax) else 0.1 * (vmax - vmin)
    return [vmin - margin, vmax + margin]


def save_training_plot_data(run_dir, ep_returns, ep_steps_to_goal, ema_alpha=0.05):
    data_dir = _plot_data_dir(run_dir)
    os.makedirs(data_dir, exist_ok=True)

    x_ret = list(range(1, len(ep_returns) + 1))
    x_step = list(range(1, len(ep_steps_to_goal) + 1))
    ret_raw = _to_float_list(ep_returns)
    step_raw = _to_float_list(ep_steps_to_goal)
    ret_ema = _to_float_list(_ema(ret_raw, ema_alpha))
    step_ema = _to_float_list(_ema(step_raw, ema_alpha))
    ret_ylim = _calc_ylim(ret_raw)
    step_ylim = [0.0, float(max(ep_steps_to_goal) + 5)] if ep_steps_to_goal else [0.0, 1.0]

    payload = {
        "ema_alpha": float(ema_alpha),
        "returns": {"x": x_ret, "raw": ret_raw, "ema": ret_ema, "ylabel": "Return", "ylim": ret_ylim},
        "steps_to_goal": {
            "x": x_step, "raw": step_raw, "ema": step_ema,
            "ylabel": "Steps to Source", "ylim": step_ylim,
        },
    }
    with open(os.path.join(data_dir, "training_metrics.json"), "w") as f:
        json.dump(payload, f, indent=2)

    with open(os.path.join(data_dir, "returns.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "return_raw", "return_ema"])
        for ep, raw, ema_v in zip(x_ret, ret_raw, ret_ema):
            w.writerow([ep, raw, ema_v])

    with open(os.path.join(data_dir, "steps_to_goal.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "step_raw", "step_ema"])
        for ep, raw, ema_v in zip(x_step, step_raw, step_ema):
            w.writerow([ep, raw, ema_v])


def save_raw_ema_png(run_dir, filename, x, y_raw=None, y_ema=None, ylabel="", ylim=None):
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    has_label = False
    if y_raw is not None:
        ax.plot(x, y_raw, alpha=0.25)
    if y_ema is not None:
        ax.plot(x, y_ema, linewidth=2.0, label=ylabel)
        has_label = True
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    if has_label:
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, filename), dpi=150)
    plt.close(fig)


def plot_training_pngs_from_data(run_dir):
    data_path = os.path.join(_plot_data_dir(run_dir), "training_metrics.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing training plot data: {data_path}")
    with open(data_path, "r") as f:
        data = json.load(f)

    for key, fname in (("returns", "returns.png"), ("steps_to_goal", "steps_to_goal.png")):
        d = data[key]
        save_raw_ema_png(
            run_dir, fname, d["x"],
            y_raw=d.get("raw"), y_ema=d.get("ema"),
            ylabel=d.get("ylabel", ""),
            ylim=tuple(d.get("ylim")) if d.get("ylim") is not None else None,
        )


# ---------------------------------------------------------------------------
# Rollout GIF rendering — bilateral sensor + head/body separation aware
# ---------------------------------------------------------------------------


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
