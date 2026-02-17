import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio  # <--- 필수 추가

def _plot_data_dir(run_dir):
    return os.path.join(run_dir, "plot_data")

def _to_float_list(values):
    return [float(v) for v in values]

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
        "returns": {
            "x": x_ret,
            "raw": ret_raw,
            "ema": ret_ema,
            "ylabel": "Return",
            "ylim": ret_ylim,
        },
        "steps_to_goal": {
            "x": x_step,
            "raw": step_raw,
            "ema": step_ema,
            "ylabel": "Step to Source",
            "ylim": step_ylim,
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

def plot_training_pngs_from_data(run_dir):
    data_path = os.path.join(_plot_data_dir(run_dir), "training_metrics.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing training plot data: {data_path}")
    with open(data_path, "r") as f:
        data = json.load(f)

    ret = data["returns"]
    save_raw_ema_png(
        run_dir,
        "returns.png",
        ret["x"],
        y_raw=ret.get("raw"),
        y_ema=ret.get("ema"),
        ylabel=ret.get("ylabel", "Return"),
        ylim=tuple(ret.get("ylim")) if ret.get("ylim") is not None else None,
    )

    step = data["steps_to_goal"]
    save_raw_ema_png(
        run_dir,
        "steps_to_goal.png",
        step["x"],
        y_raw=step.get("raw"),
        y_ema=step.get("ema"),
        ylabel=step.get("ylabel", "Step to Source"),
        ylim=tuple(step.get("ylim")) if step.get("ylim") is not None else None,
    )

def save_eval_plot_data(run_dir, env_config, trajectories, title):
    data_dir = _plot_data_dir(run_dir)
    os.makedirs(data_dir, exist_ok=True)
    payload = {
        "title": str(title),
        "env_config": {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v for k, v in env_config.items()},
        "trajectories": [
            {
                "return": float(t["return"]),
                "success": bool(t["success"]),
                "x": _to_float_list(t["x"]),
                "y": _to_float_list(t["y"]),
            }
            for t in trajectories
        ],
    }
    with open(os.path.join(data_dir, "eval_trajectories.json"), "w") as f:
        json.dump(payload, f, indent=2)

def plot_trajectory_png_from_data(run_dir, out_png=None):
    data_path = os.path.join(_plot_data_dir(run_dir), "eval_trajectories.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing eval plot data: {data_path}")
    with open(data_path, "r") as f:
        data = json.load(f)
    if out_png is None:
        out_png = os.path.join(run_dir, "plots", "trajectory.png")
    plot_trajs_png(data["env_config"], out_png, data["trajectories"], data["title"])

def replot_all_pngs(run_dir):
    plot_training_pngs_from_data(run_dir)
    plot_trajectory_png_from_data(run_dir)

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
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        margin = max(1.0, abs(vmax) * 0.1)
    else:
        margin = 0.1 * (vmax - vmin)
    return [vmin - margin, vmax + margin]

def save_raw_ema_png(run_dir, filename, x, y_raw=None, y_ema=None, ylabel="", ylim=None, avg_label=None):
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    has_label = False
    if y_raw is not None:
        ax.plot(x, y_raw, alpha=0.25)
    if y_ema is not None:
        ax.plot(x, y_ema, linewidth=2.0, label=f"{ylabel}")
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

def plot_trajs_png(env_config, out_png, trajs, title):
    L = env_config['L']
    src_x, src_y = env_config['src_x'], env_config['src_y']
    
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    
    # Heatmap (match environment concentration field with wind effect)
    res = 100
    xs = np.linspace(-L, L, res); ys = np.linspace(-L, L, res)
    X, Y = np.meshgrid(xs, ys)
    sigma_c = float(env_config.get('sigma_c', 1.0))
    bg_c = float(env_config.get('bg_c', 0.0))
    wind_x = float(env_config.get('wind_x', 0.0))
    wind_y = float(env_config.get('wind_y', 0.0))
    wind_mag = float(np.hypot(wind_x, wind_y))

    xr = X - src_x
    yr = Y - src_y
    if wind_mag <= 1e-6:
        r2 = xr * xr + yr * yr
        c = np.exp(-r2 / (2.0 * sigma_c * sigma_c))
    else:
        wx, wy = wind_x / wind_mag, wind_y / wind_mag
        t = xr * wx + yr * wy
        s = -xr * wy + yr * wx
        stretch = 1.0 + min(wind_mag, 2.0)
        sigma_s = sigma_c
        sigma_t = sigma_c * stretch
        sigma_up = sigma_c / stretch
        t_pos = np.maximum(0.0, t)
        t_neg = np.maximum(0.0, -t)
        c = np.exp(
            -(
                (s * s) / (2.0 * sigma_s**2)
                + (t_pos**2) / (2.0 * sigma_t**2)
                + (t_neg**2) / (2.0 * sigma_up**2)
            )
        )
    c = bg_c + (1.0 - bg_c) * c
    c = np.clip(c, 0.0, 1.0)
    im = ax.imshow(c, extent=[-L, L, -L, L], origin="lower", cmap="inferno", alpha=1.0, vmin=0.0, vmax=1.0)

    # Goal
    ax.plot(src_x, src_y, 'ko')
    circle = plt.Circle((src_x, src_y), env_config['r_goal'], color='gray', fill=False)
    ax.add_patch(circle)

    for t in trajs:
        ax.plot(t['x'], t['y'], alpha=0.6)
        ax.plot(t['x'][0], t['y'][0], 'x')
        ax.plot(t['x'][-1], t['y'][-1], 's')

    rets = [t['return'] for t in trajs]
    ax.set_title(f"{title}\nAvg Return: {np.mean(rets):.2f}")
    ax.title.set_color("white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(["0.0", "1.0"])
    cbar.ax.tick_params(colors="white")
    cbar.outline.set_edgecolor("white")
    
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

# ▼▼▼▼▼▼▼ [추가된 부분] ▼▼▼▼▼▼▼
def save_gif(frames, path, fps=30):
    """
    프레임 리스트를 받아서 GIF로 저장
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # loop=0 means infinite loop
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"[GIF] Saved to {path}")

def render_rollout_frame_png_style(env, title=None):
    L = float(env.L)
    src_x, src_y = float(env.src_x), float(env.src_y)

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    # Match trajectory.png heatmap style and scale.
    res = 100
    xs = np.linspace(-L, L, res)
    ys = np.linspace(-L, L, res)
    X, Y = np.meshgrid(xs, ys)
    c = np.asarray(env._conc(X, Y), dtype=np.float32)
    c = np.clip(c, 0.0, 1.0)
    im = ax.imshow(c, extent=[-L, L, -L, L], origin="lower", cmap="inferno", alpha=1.0, vmin=0.0, vmax=1.0)

    ax.plot(src_x, src_y, "ko")
    circle = plt.Circle((src_x, src_y), float(env.r_goal), color="gray", fill=False)
    ax.add_patch(circle)

    if len(env._trail) > 0:
        tx = [float(p[0]) for p in env._trail]
        ty = [float(p[1]) for p in env._trail]
        ax.plot(tx, ty, alpha=0.8, color="#50dcff")
        ax.plot(tx[0], ty[0], "x", color="white")
        ax.plot(tx[-1], ty[-1], "s", color="white", markersize=5)

    # Agent pose (oriented triangle) + current sensing ray/point + scan label.
    ax_x = float(env.x)
    ax_y = float(env.y)
    th = float(env.th)
    tri_len = 0.18
    p0 = (ax_x + tri_len * np.cos(th), ax_y + tri_len * np.sin(th))
    p1 = (ax_x + tri_len * np.cos(th + 2.5), ax_y + tri_len * np.sin(th + 2.5))
    p2 = (ax_x + tri_len * np.cos(th - 2.5), ax_y + tri_len * np.sin(th - 2.5))
    tri = plt.Polygon(
        [p0, p1, p2],
        closed=True,
        facecolor=(50 / 255.0, 100 / 255.0, 220 / 255.0),
        edgecolor="white",
        linewidth=0.8,
    )
    ax.add_patch(tri)

    if getattr(env, "_sense_pt", None) is not None:
        sx, sy = float(env._sense_pt[0]), float(env._sense_pt[1])
        ax.plot([ax_x, sx], [ax_y, sy], color=(220 / 255.0, 60 / 255.0, 60 / 255.0), linewidth=2)
        ax.plot(sx, sy, "o", color=(220 / 255.0, 60 / 255.0, 60 / 255.0), markersize=4)
        labels = ("F", "L", "R")
        scan_idx = int(getattr(env, "_render_scan_idx", 0))
        scan_idx = max(0, min(scan_idx, 2))
        ax.text(
            ax_x + 0.06 * L,
            ax_y + 0.06 * L,
            labels[scan_idx],
            color="white",
            fontsize=10,
            ha="left",
            va="bottom",
            bbox=dict(facecolor="black", edgecolor="none", alpha=0.4, pad=1.0),
        )

    if title:
        ax.set_title(title, color="white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(["0.0", "1.0"])
    cbar.ax.tick_params(colors="white")
    cbar.outline.set_edgecolor("white")

    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame = rgba[..., :3].copy()
    plt.close(fig)
    return frame
