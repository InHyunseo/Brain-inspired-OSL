# src — OSL library

The reusable library behind the `train.py` / `eval.py` / `main.py` entry points.
PPO only; selectable connectome or GRU actor backbone.

## Subpackages

- **`envs/`** — the environment.
  - `osl_env.py` — `OslEnv` + `EnvConfig`. 6-D observation
    `[c_left, c_right, dlog, prev_v, prev_body_ω, prev_head_ω]`, 3-D action
    `[v, body_ω, head_ω]` in [-1, 1]. Head and body rotate on independent axes.
  - `odor_field.py` — `GaussianOdorField` + bump-field perturbation; one scalar
    α ∈ [0, 1] scales every bump parameter (clean → turbulent).
  - `geometry.py` — angle wrap + bilateral sensor positions.
  - `events.py` — `classify_event(...)` → run / cast / turn / spin flags
    (diagnostic labels, not policy gating).
  - `parallel_runner.py` — `ParallelRunner` (subprocess) / `VectorRunner`
    (in-process); both expose `set_noise_stage(stage, strength)` for the curriculum.
- **`models/`** — networks.
  - `connectome.py` — `Connectome` actor: sparse 387-node larva graph + sensor
    input nodes + MBON output nodes; 6 message-passing steps per env step.
  - `gru_backbone.py` — `GRUBackbone`: a plain GRU mirroring the connectome
    interface (the capacity reference). Swap with `--backbone {connectome, gru}`.
  - `policy.py` — `Policy`: actor (connectome or GRU) + a recurrent value head.
- **`agents/`** — `ppo_agent.py`: custom on-policy PPO (`PPOTrainer` + `PPOConfig`
  + `RolloutBuffer`), GAE, sequence updates, persistent rollout state so a
  curriculum can be advanced phase by phase.
- **`baselines/`** — `chemotaxis.py`: the non-learning bilateral-gradient
  controller (`BilateralChemotaxis`) used as the upper reference on clean fields.
- **`utils/`** — `config.py` (CLI parser), `factory.py` (env + PPO trainer
  factories), `seed.py` (RNG), `plotter.py` (rollout frame + GIF rendering).

## Import graph

```
train.py / eval.py / main.py
    └─ utils.config, utils.factory, utils.seed
            └─ agents.ppo_agent, models.policy
                    └─ envs.osl_env  ←  envs.{events, geometry, odor_field}
                            └─ models.{connectome, gru_backbone}, envs.parallel_runner
eval.py → utils.plotter   (best-episode GIF)
```
