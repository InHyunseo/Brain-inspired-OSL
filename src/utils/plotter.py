import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio  # <--- 필수 추가

def save_raw_ema_png(run_dir, filename, x, y_raw, y_ema, ylabel, ylim):
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, y_raw, alpha=0.25)
    ax.plot(x, y_ema, label=ylabel)
    ax.set_ylabel(ylabel); ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, filename), dpi=150)
    plt.close(fig)

def plot_trajs_png(env_config, out_png, trajs, title):
    L = env_config['L']
    src_x, src_y = env_config['src_x'], env_config['src_y']
    
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal")
    
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
    ax.imshow(c, extent=[-L, L, -L, L], origin="lower", cmap="inferno", alpha=0.5)

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
