# OSL — Odor Source Localization with a Larva-Inspired RL Agent

A bio-inspired reinforcement-learning study of **odor source localization (OSL)**:
reach an unseen odor source from local concentration alone, using two ideas
borrowed from *Drosophila* larva chemotaxis —

- **active sensing** — an independent head axis (`head_omega`) so head-casting can
  emerge as a learned behavior, and
- a **connectome backbone** — the real ~387-node larva connectivity graph used as
  the policy's recurrent actor (vs. a plain GRU as the capacity reference).

The task is a POMDP: only the current bilateral concentration is observed, so
direction must be integrated over time. Plumes range from a clean Gaussian field
to a bump-field "turbulence" via a single curriculum scalar α ∈ [0, 1].

## What was done / results

- **Hand-built baseline (no network).** Bilateral gradient steering solves the
  clean Gaussian field every time (100%, ~455 steps) and degrades gracefully
  under noise (~46–50% at the hardest α=1.0). It is the bar any learned policy
  must clear, and no RL agent here beats it on the clean Gaussian.
- **PPO + GRU policy.** Learns the clean field (~100% eval success). Crucially,
  **cast / active-sensing rises with noise** — the same trend the baseline shows
  by hand-coded rule, here emerging from learning alone. Note: the policy only
  solves the task with **stochastic** rollouts; deterministic (mean) actions give
  ~0%, so all analysis uses stochastic rollouts.
- **Connectome backbone (negative result).** Swapping the GRU for the real larva
  connectome (~17.5K learnable edge scalars, ~1/30 the GRU's parameters) fails to
  train end-to-end — the biological sparsity that saves parameters appears to
  cost trainability/capacity. Scaling it up does train; the limit was capacity,
  not the wiring.
- **Mechanistic analysis** (`analysis/osl2d/`) on the GRU hidden state: behavior
  labeling → linear probing → Jacobian eigenmodes → causal ablation. Active
  sensing shows oscillatory dynamics where RUN does not, and the neurons carrying
  it are reassigned as the noise level changes.

## Folder structure

```
OSL/
├── README.md            # this file — project overview, results, run commands
├── requirements.txt     # pure PyTorch + numpy + gymnasium + matplotlib (no sb3)
├── train.py eval.py main.py   # PPO entry points (train / eval / both)
├── visualize_curriculum_field.py   # render per-phase odor-field PNG + GIF
├── assets/connectome/   # weights.csv, metadata.csv (the larva connectivity graph)
├── src/                 # the library — env, models, PPO agent, baseline, utils
│   └── README.md        # subpackage map + import graph
├── analysis/            # mechanistic analysis pipeline on trained policies
│   ├── README.md        # pipeline overview
│   ├── methodology.md   # the 3-phase methodology
│   ├── METHODS_MATH.md  # per-phase equations
│   └── osl2d/           # collect → label → probe → Jacobian → ablation
├── notebooks/           # Colab-friendly end-to-end notebooks
│   └── README.md
└── demo/                # legacy RL practice / analysis-plumbing sanity checks
    └── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Connectome CSVs must live at `assets/connectome/weights.csv` and
`assets/connectome/metadata.csv` (override with `--weights-csv` / `--metadata-csv`).

## Run

```bash
# Train (PPO; default noise curriculum). Outputs land in runs/ppo_main_*/
python3 train.py --backbone gru          # GRU backbone (the working reference)
python3 train.py --backbone connectome   # connectome backbone

# Short smoke run
python3 train.py --curriculum-phases '[[0,0.0,50000]]' --num-envs 4 --no-parallel-envs

# Eval a finished run (deterministic rollouts + best-episode GIF)
python3 eval.py --run-dir runs/ppo_main_YYYYMMDD_HHMMSS \
  --eval-noise-stage 1 --eval-noise-strength 0.5 --eval-episodes 50

# Train + eval back-to-back
python3 main.py
```

Key flags: `--backbone {gru, connectome}`, `--gru-hidden` (421),
`--curriculum-phases` (JSON list of `[noise_stage, noise_strength, timesteps]`),
`--message-passing-steps` (6), `--latent-dim` (32), `--seed`, `--force-cpu`.
The env is a 6-D observation `[c_left, c_right, dlog, prev_v, prev_body_ω,
prev_head_ω]` and a 3-D action `[v, body_ω, head_ω]` in [-1, 1]; see
[`src/README.md`](src/README.md) for the full env API.

## Analysis

After a run finishes, the mechanistic pipeline reads its checkpoint + dumped
traces; see [`analysis/README.md`](analysis/README.md).
