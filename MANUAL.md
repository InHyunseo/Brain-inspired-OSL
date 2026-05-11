# Manual

## Environment
- Ubuntu 22.04 (WSL2 OK), Python 3.10+
- No sb3/sb3-contrib dependency anymore — pure PyTorch + numpy + gymnasium

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Required assets
The connectome CSVs must live at:
```
assets/connectome/weights.csv
assets/connectome/metadata.csv
```
Override locations with `--weights-csv` / `--metadata-csv`.

## Notebooks (recommended for interactive runs)

All three notebooks share the same Colab-friendly pattern: first cell clones the repo + installs deps + `cd`s in, then a single hyperparameter block at the top of the training cell drives everything.

| Notebook | Track | Env |
|---|---|---|
| `ipynb/PPO_framework.ipynb` | PPO (custom on-policy) | new bilateral `OslEnv` |
| `ipynb/RSAC_framework.ipynb` | RSAC (off-policy episode loop) | new bilateral `OslEnv` |
| `ipynb/DRQN_framework.ipynb` | DRQN/DQN (legacy demo) | old `StaticEnv`/`DynamicEnv` from `demo/DRQN/` |

Edit `PHASES`, `ENV_KW`, `PPO_KW` / `AGENT_KW` at the top of the relevant cell to tune. The `.py` CLIs below are the same code paths — useful for headless / batch runs.

## Run Commands

### PPO (default, 4-phase noise curriculum)
```bash
python3 train.py --agent-type ppo
```
Phases (default): `[(noise_stage=0, strength=0.0, 1.5M), (1, 0.3, 0.5M), (1, 0.6, 0.5M), (2, 1.0, 1.0M)]`.
Stage 0 = clean Gaussian, stage 1 = static white noise, stage 2 = temporally correlated AR(1) noise (advanced each step).
Outputs: `runs/ppo_main_*/`. Final checkpoint at `ckpt_final.pt`; per-phase checkpoints under `checkpoints/step_*.pt`.

#### Short debug run
```bash
python3 train.py --agent-type ppo \
  --curriculum-phases '[[0,0.0,50000]]' --num-envs 4 --no-parallel-envs
```

### RSAC (single env)
```bash
python3 train.py --agent-type rsac \
  --rsac-actor-backbone connectome \
  --total-episodes 20000 \
  --rsac-noise-stage 0 --rsac-noise-strength 0.0
```
Backbone choices: `connectome` (default), `gru`, `mlp`.

### End-to-end (train + eval)
```bash
python3 main.py --agent-type ppo
```

### Eval only
```bash
python3 eval.py --run-dir runs/ppo_main_YYYYMMDD_HHMMSS \
  --eval-noise-stage 1 --eval-noise-strength 0.5 --eval-episodes 50
```
PPO eval loads `ckpt_final.pt`, rolls deterministic episodes, and renders the best-return episode as a GIF.
RSAC loads `best.pt` (fallback `final.pt`).

### Replot
```bash
python3 replot.py --run-dir runs/<agent>_main_YYYYMMDD_HHMMSS
```

## CLI Flags

### Common
- `--agent-type {ppo, rsac}` (default `ppo`)
- `--seed`, `--force-cpu`, `--out-dir`, `--run-name`

### Env
- `--sensor-spacing-mm` (0.15)
- `--episode-seconds` (120.0)
- `--arena-width-mm`, `--arena-height-mm` (80, 120)
- `--source-x-mm`, `--source-y-mm` (40, 100)
- `--gaussian-sigma-mm` (30.0)
- `--success-radius-mm` (7.5)

### Connectome
- `--weights-csv` (default `assets/connectome/weights.csv`)
- `--metadata-csv` (default `assets/connectome/metadata.csv`)
- `--message-passing-steps` (6)
- `--latent-dim` (32)

### PPO
- `--curriculum-phases` (JSON list of `[noise_stage, noise_strength, timesteps]`)
- `--num-envs` (16), `--rollout-steps` (128), `--update-epochs` (4), `--minibatch-envs` (4)
- `--gamma` (0.99), `--gae-lambda` (0.95), `--clip-epsilon` (0.2)
- `--entropy-coef` (0.005), `--value-loss-coef` (0.5)
- `--actor-lr` (3e-4), `--critic-lr` (1e-3), `--log-std-init` (-0.5)
- `--actor-max-grad-norm` (0.5), `--critic-max-grad-norm` (0.5)
- `--parallel-envs` / `--no-parallel-envs` (default on; subprocess fork)
- `--log-every-updates` (1), `--eval-interval-updates` (10), `--checkpoint-every-timesteps` (100000)

### RSAC
- `--rsac-actor-backbone {connectome, gru, mlp}` (default `connectome`)
- `--total-episodes` (20000)
- `--rnn-hidden` (147), `--lr-actor / --lr-critic / --lr-alpha` (3e-4)
- `--gamma 0.99`, `--tau 0.005`
- `--batch-size 128`, `--seq-len 16`, `--buffer-size 150000`, `--learning-starts 5000`
- `--rsac-noise-stage`, `--rsac-noise-strength` (env noise during training)

### Eval
- `--eval-episodes` / `--episodes` (alias)
- `--seed-base` (20000), `--ckpt`
- `--save-gif` / `--no-save-gif`
- `--eval-noise-stage` (2), `--eval-noise-strength` (1.0)

## Env API

### Observation (5,)
```
[c_left, c_right, prev_v_norm, prev_body_omega_norm, prev_head_omega_norm]
```
- `c_left`, `c_right`: instantaneous concentration at the two head sensors (raw, not log)
- Last three: efference copy of the previous step's normalized motor command

### Action (3,) in [-1, 1]
```
[v, body_omega, head_omega]
```
- `v` mapped to `[0, v_max_mm_s]` (1.2 mm/s)
- `body_omega` mapped to ±`body_omega_max_deg_s` (120°/s)
- `head_omega` mapped to ±`head_omega_max_deg_s` (240°/s) — head rotates independently of body, sensor heading = body_heading + head_relative_angle

### Reward
```
reward_goal      = +10.0 on success (within success_radius)
reward_log       = 0.1 * d/dt log(c_avg)        # following the gradient is rewarded
reward_time      = -0.005 per step
reward_wall      = -2.0 on wall contact (terminal)
reward_spin      = -0.02 if event flagged as spin_like
```

### Termination
- `terminated` if success or wall contact
- `truncated` if `step_count >= episode_seconds / dt` (1200 steps at default dt=0.1)

### Info dict (selected keys)
- `distance_to_source_mm`, `bearing_to_source_rad`
- `sensor_left`, `sensor_right`, `sensor_avg`
- `x_mm`, `y_mm`, `heading_rad`, `head_relative_angle_rad`
- `v_mm_s`, `body_omega_rad_s`, `head_omega_rad_s`
- `event_is_run / is_stop / is_low_sweep / is_high_cast_like / is_turn_like / is_spin_like`
- `success`, `is_success`, `wall_contact`, `termination_reason` (`success`/`wall`/`time_limit`/`running`)

## Output Structure (PPO)
```
runs/ppo_main_YYYYMMDD_HHMMSS/
├── checkpoints/step_*.pt          # periodic snapshots (every 100k steps)
├── ckpt_final.pt                  # final checkpoint
├── plots/best_agent.gif
├── training_log.jsonl             # per-update metrics
├── summary.json
└── config.json
```

## Output Structure (RSAC)
```
runs/rsac_main_YYYYMMDD_HHMMSS/
├── checkpoints/{best,first,final}.pt
├── plots/{returns.png, steps_to_goal.png, best_agent.gif}
├── plot_data/{training_metrics.json, returns.csv, steps_to_goal.csv}
└── config.json
```

## Notes
- Pre-refactor `runs/` checkpoints (`final.zip` + `final_vnorm.pkl`) are **not** loadable. Re-train from scratch.
- The connectome CSVs were ported from the bySY subrepo (`bySY/larva_connectome_output/Total_Submatrix_*.csv`). The shape (389×389 weights, 387 base neurons + 2 padding) is preserved.
- `head_omega` is fully continuous — there is no discrete cast macro anymore. The `event_is_high_cast_like` flag is purely diagnostic (used for GIF cast markers, not for policy gating).
