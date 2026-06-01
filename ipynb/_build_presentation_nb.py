"""Generate ipynb/Presentation_assets.ipynb — one notebook that produces every
figure in presentation_assets/ (baseline + learned-policy). Run once:

    python ipynb/_build_presentation_nb.py
"""
import json
from pathlib import Path

NB = Path(__file__).parent / "Presentation_assets.ipynb"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


cells = []

cells.append(md("""# Presentation assets — one notebook, every slide figure

Builds **all** figures used in `presentation_assets/`, for both the **baseline**
(sensor-only chemotaxis, no model to load) and the **learned policy** (loaded
from a checkpoint):

| figure | source |
|---|---|
| `slide2_baseline_noise_sweep.png` | baseline — success vs noise |
| `baseline_s2_1p0.gif` | baseline — one trajectory GIF |
| `slide4_training_curves.png` | learned — success ratio vs steps |
| `slide6_active_sensing_overlap.png` | learned — active-sensing top-k overlap vs noise (**box plot**) |
| `slide6_jacobian_run.png` / `_active_sensing.png` | learned — eigenvalue clouds |
| `slide6_jacobian_oscillation.png` | learned — dominant-mode oscillation fraction (**box plot**) |

Set the checkpoint path in the **Config** cell, then *Run all*. Figures are saved
to `presentation_assets/` and shown inline.

**Plot format is unified by the kind of value:** success-rate / steps use a
colored mean **point+line with per-seed grey dots**; distributional quantities
(overlap, oscillation) use **box plots**. (`src/utils/baseline_plots.py`.)"""))

# ---- setup ----
cells.append(code("""# Setup — works locally and on Colab. Re-running is safe.
import os, sys, subprocess

if os.path.isdir('/content'):
    REPO_URL = 'https://github.com/InHyunseo/Brain-inspired-OSL.git'
    REPO_DIR = '/content/2d-osl'
    if not os.path.isdir(REPO_DIR):
        subprocess.check_call(['git', 'clone', REPO_URL, REPO_DIR])
else:
    REPO_DIR = os.path.abspath(os.getcwd())
    while not os.path.isdir(os.path.join(REPO_DIR, 'src')) and REPO_DIR != os.path.dirname(REPO_DIR):
        REPO_DIR = os.path.dirname(REPO_DIR)
os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
print('repo:', REPO_DIR, '\\ncwd :', os.getcwd())"""))

# ---- config ----
cells.append(md("""## Config
- `RUN_DIR` / `CKPT_LABEL` — the trained **PPO+GRU** run used for the learned-policy
  figures. On Colab, point `RUN_DIR` at wherever your run lives.
- `DEVICE` — `'auto'` uses CUDA when available (Colab T4/G4).
- `N_EPISODES` — episodes per condition (200 for stable stats; drop to ~40 for a
  quick rehearsal)."""))

cells.append(code("""# ===== CONFIG — edit these =====
RUN_DIR   = 'runs/ppo_gru_nb_20260531_113633'   # trained PPO+GRU run (learned-policy figs)
CKPT_LABEL = 'final'                            # checkpoint label under RUN_DIR
DEVICE     = 'auto'                             # 'auto' | 'cpu' | 'cuda'
N_EPISODES = 200                                # episodes / condition
SEEDS      = [0, 1, 2, 3]                       # seed groups (grey dots)

# Noise curriculum used everywhere below.
NOISE_SWEEP = [(0, 0.0), (1, 0.3), (1, 0.6), (1, 1.0), (2, 0.3), (2, 0.6), (2, 1.0)]
# Stage-2 strengths for the learned-policy overlap/oscillation sweep.
LEARNED_NOISE_STAGE = 2
LEARNED_NOISE_STRENGTHS = [0.0, 0.1, 0.2, 0.3]
TOP_K = 16
SAMPLES_PER_LABEL = 200          # Jacobian eigenvalue samples per behavior
EPISODES_PER_SEED_LEARNED = 8    # stochastic trace episodes/seed/noise level

import os, torch
import matplotlib.pyplot as plt
ASSETS = 'presentation_assets'
os.makedirs(ASSETS, exist_ok=True)
_dev = ('cuda' if torch.cuda.is_available() else 'cpu') if DEVICE == 'auto' else DEVICE
print('device:', _dev, '| run_dir:', RUN_DIR, '| n/condition:', N_EPISODES)

from src.utils.baseline_plots import (apply_style, point_line_with_seeds,
                                      boxplot_by_group, seed_subaggregates,
                                      ACCENT, BLUE, RED)
apply_style()"""))

# ============================== BASELINE ==============================
cells.append(md("""# Baseline (sensor-only chemotaxis — no model)

A purely computational controller: bilateral-gradient steering + stop/go +
active sensing (head sweep while stopped). No network, nothing to load."""))

cells.append(code("""import numpy as np
from src.envs.osl_env import EnvConfig, OslEnv

ENV_KW = dict(sensor_spacing_mm=0.15, episode_seconds=120.0,
              arena_width_mm=80.0, arena_height_mm=120.0,
              source_x_mm=40.0, source_y_mm=100.0,
              gaussian_sigma_mm=30.0, success_radius_mm=7.5)
STEER_GAIN = 80.0          # tanh(STEER_GAIN*asym): soft sign() on the bilateral asym
AS_HEAD_OMEGA = 1.0        # head-sweep amplitude while active-sensing
AS_HALF_PERIOD = 6         # steps before flipping sweep direction


class MinimalController:
    \"\"\"Bilateral gradient steering + stop/go + head-sweep active sensing.\"\"\"
    def __init__(self, steer_gain=STEER_GAIN, as_head_omega=AS_HEAD_OMEGA,
                 as_half_period=AS_HALF_PERIOD):
        self.steer_gain = steer_gain
        self.as_head_omega = as_head_omega
        self.as_half_period = as_half_period
        self.reset()

    def reset(self):
        self._sweep_dir = 1.0
        self._sweep_phase = 0

    def act(self, obs):
        c_left, c_right, dlog = float(obs[0]), float(obs[1]), float(obs[2])
        asym = (c_left - c_right) / (c_left + c_right + 1e-9)
        body_omega = float(np.tanh(self.steer_gain * asym))   # steer to stronger antenna
        if dlog >= 0.0:                                        # rising → go
            self._sweep_phase = 0
            return np.asarray([1.0, body_omega, 0.0], dtype=np.float32)
        # falling → stop and sweep the head (active sensing)
        self._sweep_phase += 1
        if self._sweep_phase >= self.as_half_period:
            self._sweep_phase = 0
            self._sweep_dir *= -1.0
        head_omega = float(np.clip(self._sweep_dir * self.as_head_omega, -1.0, 1.0))
        return np.asarray([-1.0, body_omega, head_omega], dtype=np.float32)


def make_env(stage, strength, seed):
    return OslEnv(EnvConfig.from_dict({**ENV_KW, 'noise_stage': int(stage),
                                       'noise_strength': float(strength), 'seed': seed}))


def run_episode(env, controller, seed, collect_traj=False, render_fn=None):
    obs, _ = env.reset(seed=seed)
    controller.reset()
    ret, success, as_steps = 0.0, False, 0
    tx, ty, ax_, ay_, frames = [], [], [], [], []
    for t in range(env.max_steps):
        action = controller.act(obs)
        if collect_traj:
            tx.append(env.x_mm); ty.append(env.y_mm)
        obs, r, term, trunc, info = env.step(action)
        ret += float(r)
        if info.get('event_is_high_cast_like'):
            as_steps += 1
            if collect_traj:
                ax_.append(env.x_mm); ay_.append(env.y_mm)
        if collect_traj and render_fn is not None:
            frames.append(render_fn(env, tx, ty, ax_, ay_, t,
                                    title=f'baseline seed={seed} step={t} AS={as_steps}'))
        if term or trunc:
            success = bool(info.get('success', False)); break
    return {'seed': seed, 'return': ret, 'success': success, 'steps': t + 1,
            'as_steps': as_steps, 'frames': frames if collect_traj else None}


controller = MinimalController()
_c = run_episode(make_env(0, 0.0, 20000), controller, 20000)
print('smoke (clean):', {k: _c[k] for k in ('success', 'steps', 'as_steps')})"""))

# ---- baseline evaluate + noise sweep figure ----
cells.append(md("""## Baseline — success ratio & steps-to-source vs noise
`slide2_baseline_noise_sweep.png`. Colored mean point+line; grey dots are
per-seed sub-aggregates. Clean field (stage 0) is **100%**."""))

cells.append(code("""import json
import matplotlib.pyplot as plt

def evaluate(stage, strength, ctrl=controller, n_episodes=N_EPISODES,
             seeds=tuple(SEEDS), seed_base=20000):
    per_seed = int(np.ceil(n_episodes / len(seeds)))
    succ, steps, eps = [], [], []
    for gi, sd in enumerate(seeds):
        for k in range(per_seed):
            seed = seed_base + gi * 100_000 + k
            r = run_episode(make_env(stage, strength, seed), ctrl, seed)
            succ.append(int(r['success'])); steps.append(r['steps']); eps.append(sd)
    succ = np.asarray(succ); steps = np.asarray(steps); eps = np.asarray(eps)
    seed_steps = []
    for sd in seeds:
        m = (eps == sd) & (succ == 1)
        seed_steps.append(float(steps[m].mean()) if m.any() else np.nan)
    return {
        'stage': stage, 'strength': strength, 'n': len(succ),
        'success_rate': float(succ.mean()),
        'mean_steps_success': float(steps[succ == 1].mean()) if (succ == 1).any() else np.nan,
        'seed_success_rate': seed_subaggregates(succ, eps, np.mean),
        'seed_mean_steps_success': np.asarray(seed_steps, dtype=float),
    }

rows = [evaluate(st, a) for st, a in NOISE_SWEEP]
clean = rows[0]
print(f"CLEAN success = {clean['success_rate']:.0%}  (n={clean['n']})")
assert clean['success_rate'] == 1.0, 'clean field must be solved every episode'

labels = [f"s{r['stage']}·α{r['strength']}" for r in rows]
x = np.arange(len(labels))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
point_line_with_seeds(ax[0], x, [r['seed_success_rate'] for r in rows], color=ACCENT)
ax[0].set_ylim(-0.02, 1.04); ax[0].set_ylabel('success rate'); ax[0].set_title('Success ratio')
point_line_with_seeds(ax[1], x, [r['seed_mean_steps_success'] for r in rows], color=ACCENT)
ax[1].set_ylabel('steps to source'); ax[1].set_title('Steps to source')
for a_ in ax:
    a_.set_xticks(x); a_.set_xticklabels(labels, rotation=30)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, 'slide2_baseline_noise_sweep.png'), dpi=150)
with open(os.path.join(ASSETS, 'baseline_noise_sweep.json'), 'w') as f:
    json.dump([{k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in r.items()}
               for r in rows], f, indent=2)
from IPython.display import display
display(fig); plt.close(fig)
print('saved slide2_baseline_noise_sweep.png')"""))

# ---- baseline GIF ----
cells.append(md("""## Baseline — trajectory GIF
`baseline_s2_1p0.gif`. A single trajectory at stage 2, α=1.0 (the hardest
condition shown in the deck)."""))

cells.append(code("""import os
from IPython.display import Image as DisplayImage, display
from src.utils.plotter import render_rollout_frame, save_gif

GIF_STAGE, GIF_STRENGTH = 2, 1.0
chosen = 20000
for i in range(300):                 # prefer a successful episode if one exists
    s = 20000 + i
    if run_episode(make_env(GIF_STAGE, GIF_STRENGTH, s), controller, s)['success']:
        chosen = s; break
res = run_episode(make_env(GIF_STAGE, GIF_STRENGTH, chosen), controller, chosen,
                  collect_traj=True, render_fn=render_rollout_frame)
print(f"seed={chosen} success={res['success']} steps={res['steps']} AS={res['as_steps']}")
gif_path = os.path.join(ASSETS, 'baseline_s2_1p0.gif')
save_gif(res['frames'], gif_path, fps=20)
display(DisplayImage(data=open(gif_path, 'rb').read(), format='gif'))
print('saved', gif_path)"""))

# ============================== LEARNED POLICY ==============================
cells.append(md("""# Learned policy (loaded from checkpoint)

Loads the trained PPO+GRU policy from `RUN_DIR/CKPT_LABEL` and builds the
mechanistic-interpretability figures. Needs the checkpoint + cached/dumped
stochastic traces (auto-dumped on first run)."""))

# training curve
cells.append(md("""## Learned — training curve
`slide4_training_curves.png`. Smoothed eval success-rate vs steps, from
`RUN_DIR/training_log.jsonl`."""))

cells.append(code("""import json, numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image as DisplayImage, display

log_path = os.path.join(RUN_DIR, 'training_log.jsonl')
rows_l = [json.loads(l) for l in open(log_path)]
ev = [(r['total_steps'], r['eval_success_rate']) for r in rows_l if 'eval_success_rate' in r]
ev_x = [s / 1e6 for s, _ in ev]; ev_y = [y for _, y in ev]

def _smooth(y, w=151):
    y = np.asarray(y, float)
    if len(y) < 3: return y
    w = min(w, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if w < 3: return y
    pad = w // 2
    return np.convolve(np.pad(y, pad, mode='edge'), np.ones(w) / w, mode='valid')

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(ev_x, _smooth(ev_y), color=BLUE, lw=3.0)
ax.set_xlabel('Steps (M)'); ax.set_ylim(-0.02, 1.02); ax.set_title('Success ratio')
for b in (1.5, 2.0):
    ax.axvline(b, color='gray', ls='--', lw=1.2)
for cx, name in ((0.75, 'phase 0'), (1.75, 'phase 1'), (2.25, 'phase 2')):
    ax.text(cx, 1.04, name, ha='center', va='bottom', color='dimgray')
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, 'slide4_training_curves.png'), dpi=150)
display(fig); plt.close(fig)
print('saved slide4_training_curves.png')"""))

# learned-policy GIF
cells.append(md("""## Learned — trajectory GIF
`slide1_4_trajectory.gif`. The trained policy rolled out deterministically and
rendered the same way as the baseline GIF (head-cast events marked)."""))

cells.append(code("""import os
import numpy as np
from IPython.display import Image as DisplayImage, display
from src.utils.plotter import render_rollout_frame, save_gif
from Analysis.osl2d._io import adapter_for_ckpt

LEARNED_GIF_STAGE, LEARNED_GIF_STRENGTH = 2, 0.3   # condition for the policy GIF

_gif_adapter = adapter_for_ckpt(RUN_DIR, CKPT_LABEL, device=_dev)

def policy_episode(seed, collect_traj=False):
    env = make_env(LEARNED_GIF_STAGE, LEARNED_GIF_STRENGTH, seed)
    obs, _ = env.reset(seed=seed)
    h = _gif_adapter.initial_state()
    ret, casts, success = 0.0, 0, False
    tx, ty, cx, cy, frames = [], [], [], [], []
    for t in range(env.max_steps):
        action, h = _gif_adapter.step_patched(obs, h)     # deterministic action
        if collect_traj:
            tx.append(env.x_mm); ty.append(env.y_mm)
        obs, r, term, trunc, info = env.step(action)
        ret += float(r)
        if info.get('event_is_high_cast_like'):
            casts += 1
            if collect_traj:
                cx.append(env.x_mm); cy.append(env.y_mm)
        if collect_traj:
            frames.append(render_rollout_frame(env, tx, ty, cx, cy, t,
                                               title=f'policy seed={seed} step={t} casts={casts}'))
        if term or trunc:
            success = bool(info.get('success', False)); break
    return {'seed': seed, 'success': success, 'casts': casts, 'steps': t + 1,
            'frames': frames if collect_traj else None}

# Prefer a successful episode that contains a cast (nicer for the slide).
chosen = 20000
for i in range(300):
    s = 20000 + i
    rr = policy_episode(s)
    if rr['success'] and rr['casts'] >= 1:
        chosen = s; break
res = policy_episode(chosen, collect_traj=True)
print(f"seed={chosen} success={res['success']} casts={res['casts']} steps={res['steps']}")
gif_path = os.path.join(ASSETS, 'slide1_4_trajectory.gif')
save_gif(res['frames'], gif_path, fps=20)
display(DisplayImage(data=open(gif_path, 'rb').read(), format='gif'))
print('saved', gif_path)"""))

# overlap + oscillation via noise_sweep_cast
cells.append(md("""## Learned — active-sensing neuron overlap & oscillation
- `slide6_active_sensing_overlap.png` — how stable the active-sensing top-k
  neuron set is as noise grows (**box plot** of per-seed Jaccard overlap vs the
  clean set).
- `slide6_jacobian_run.png` / `_active_sensing.png` — hidden-Jacobian dominant
  eigenvalue clouds (off the real axis ⇒ oscillation).
- `slide6_jacobian_oscillation.png` — dominant-mode oscillation fraction per
  behavior (**box plot** over Jacobian samples)."""))

cells.append(code("""from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image as DisplayImage, display

import Analysis.noise_sweep_cast as nsc
nsc.RUN_DIR = Path(RUN_DIR)
nsc.CKPT_LABEL = CKPT_LABEL
nsc.NOISE_STAGE = LEARNED_NOISE_STAGE
nsc.NOISE_STRENGTHS = LEARNED_NOISE_STRENGTHS
nsc.SEEDS = tuple(SEEDS)
nsc.EPISODES_PER_SEED = EPISODES_PER_SEED_LEARNED
nsc.MAX_STEPS = 1200
nsc.TOP_K = TOP_K
nsc.DEVICE = _dev

# ---- overlap box plot: per-seed Jaccard of active-sensing top-k vs clean ----
AS_NAME = 'ACTIVE_SENSING'
clean_per_seed = nsc.topk_per_seed(LEARNED_NOISE_STRENGTHS[0], AS_NAME)
overlap_groups = []
for a in LEARNED_NOISE_STRENGTHS:
    per_seed = nsc.topk_per_seed(a, AS_NAME)
    vals = [nsc.jaccard(clean_per_seed.get(s, []), per_seed.get(s, []))
            for s in nsc.SEEDS if s in per_seed]
    overlap_groups.append(np.asarray(vals, dtype=float))

fig, ax = plt.subplots(figsize=(6, 4.5))
boxplot_by_group(ax, overlap_groups, [f'{a:.1f}' for a in LEARNED_NOISE_STRENGTHS])
ax.set_xlabel(f'Noise (phase {LEARNED_NOISE_STAGE})'); ax.set_ylim(-0.02, 1.04)
ax.set_title('Neuron overlap')
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, 'slide6_active_sensing_overlap.png'), dpi=150)
display(fig); plt.close(fig)
print('overlap medians:', [round(float(np.median(g)), 2) if len(g) else None for g in overlap_groups])"""))

cells.append(code("""# ---- Jacobian eigenvalue clouds + oscillation box plot ----
from Analysis.osl2d._io import adapter_for_ckpt, load_traces
from Analysis.osl2d.jacobian import jacobian_at
from Analysis.osl2d.segment import LABEL_TO_INT

clean_label = f"{CKPT_LABEL}_{nsc._strength_tag(0.0)}__stoch"
# ensure clean traces exist
nsc.topk_per_seed(0.0, AS_NAME)
traces = load_traces(RUN_DIR, [clean_label])
adapter = adapter_for_ckpt(RUN_DIR, clean_label, device=_dev)
rng = np.random.default_rng(0)
theta = np.linspace(0, 2 * np.pi, 360)

def eigs_for(name):
    lab = LABEL_TO_INT[name]
    idx = np.where(traces.label == lab)[0]
    if len(idx) == 0:
        return np.array([]), np.array([]), np.array([])
    pick = rng.choice(idx, size=min(SAMPLES_PER_LABEL, len(idx)), replace=False)
    allw, dom, osc = [], [], []
    for t in pick:
        w = np.linalg.eigvals(jacobian_at(adapter, traces.obs[t], traces.h[t]))
        d = w[np.argmax(np.abs(w))]
        allw.append(w); dom.append(d); osc.append(float(abs(d.imag) > 1e-6))
    return np.concatenate(allw), np.asarray(dom), np.asarray(osc)

osc_samples = {}
for key, title, fn in [('RUN', 'Run', 'slide6_jacobian_run.png'),
                       ('ACTIVE_SENSING', 'Active sensing', 'slide6_jacobian_active_sensing.png')]:
    E_all, E_dom, osc = eigs_for(key)
    osc_samples[title] = osc
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    if len(E_all):
        ax.scatter(E_all.real, E_all.imag, s=6, alpha=0.25, color='0.7', label='all modes')
        ax.scatter(E_dom.real, E_dom.imag, s=22, alpha=0.8, color=RED,
                   edgecolors='none', label='dominant mode')
    ax.plot(np.cos(theta), np.sin(theta), color='black', lw=0.6, alpha=0.4)
    ax.axhline(0, color='k', lw=0.3, alpha=0.4); ax.axvline(0, color='k', lw=0.3, alpha=0.4)
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect('equal')
    ax.set_title(title); ax.set_xlabel('Re'); ax.set_ylabel('Im')
    ax.legend(loc='upper left', markerscale=1.3)
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, fn), dpi=150)
    display(fig); plt.close(fig)

# oscillation box plot: per-sample oscillation indicator (0/1) per behavior.
fig, ax = plt.subplots(figsize=(6, 4.5))
groups = [osc_samples.get('Run', np.array([])), osc_samples.get('Active sensing', np.array([]))]
boxplot_by_group(ax, groups, ['Run', 'Active sensing'], colors=[BLUE, RED])
ax.set_ylim(-0.05, 1.05); ax.set_ylabel('oscillatory (per-sample)')
ax.set_title('Oscillation')
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, 'slide6_jacobian_oscillation.png'), dpi=150)
display(fig); plt.close(fig)
print('oscillation fraction:', {k: round(float(v.mean()), 2) if len(v) else None
                                for k, v in osc_samples.items()})"""))

cells.append(code("""# ---- summary: list everything written ----
import glob
print('presentation_assets/ now contains:')
for p in sorted(glob.glob(os.path.join(ASSETS, '*'))):
    print('  ', os.path.basename(p))"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", NB, "with", len(cells), "cells")
