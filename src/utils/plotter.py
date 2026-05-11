"""Plotting + GIF rendering utilities shared across agents."""
import csv
import io
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio


# ---------------------------------------------------------------------------
# Training-curve plotting (RSAC / DRQN episode-loop)
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
        "steps_to_goal": {"x": x_step, "raw": step_raw, "ema": step_ema,
                          "ylabel": "Step to Source", "ylim": step_ylim},
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
# Rollout GIF rendering — ported from ipynb/PPO_framework.ipynb render_elite_gif
# ---------------------------------------------------------------------------


def render_rollout_frame(env, traj_x, traj_y, cast_x, cast_y, step, title=None):
    """Render one matplotlib frame matching the notebook's elite-GIF style.

    Works for both StaticEnv (uses analytic _conc grid) and DynamicEnv
    (uses cached _field_view). Returns an RGB numpy array.
    """
    L = float(env.L)

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    field = getattr(env, "_field_view", None)
    if field is None:
        res = 100
        xs = np.linspace(-L, L, res)
        ys = np.linspace(-L, L, res)
        X, Y = np.meshgrid(xs, ys)
        field = np.clip(np.asarray(env._conc(X, Y), dtype=np.float32), 0.0, 1.0)
    ax.imshow(field, extent=[-L, L, -L, L], origin="lower", cmap="inferno", vmin=0, vmax=1)

    ax.plot(traj_x, traj_y, color="#50dcff", linewidth=2.0, alpha=0.8)
    if cast_x:
        ax.scatter(cast_x, cast_y, color="white", marker="*", s=150,
                   edgecolors="black", zorder=10)
    ax.scatter(env.src_x, env.src_y, color="lime", marker="P", s=150, zorder=11)
    circle = plt.Circle((env.src_x, env.src_y), env.r_goal, color="gray", fill=False)
    ax.add_patch(circle)

    if title is None:
        title = f"Step: {step}"
    ax.set_title(title, color="white")
    ax.tick_params(colors="white")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=80, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    buf.seek(0)
    return np.array(imageio.v2.imread(buf))


def save_gif(frames, path, fps=15):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"[GIF] Saved to {path}")
