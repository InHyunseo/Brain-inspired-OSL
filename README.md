# Odor Source Localization (Larva Connectome RL)

2D OSL with bilateral (stereo) sensors, independent head/body rotation axes, and
a 387-node larva connectome actor. PPO is the primary track; RSAC keeps three
backbone choices (`connectome` / `gru` / `mlp`) for ablations.

## Project Goals
- Bilateral plume sensing with biologically realistic 0.15 mm sensor spacing
- Independent head-axis control (head casting via continuous `head_omega`)
- Real larva connectome as the actor (CSV-driven message passing, 17k learnable edges)
- Efference copy in observation so the policy can disentangle self-motion from plume change

## Code Structure

### Environment (`src/envs/`)
- `osl_env.py` — `OslEnv` + `EnvConfig`. Observation `[c_left, c_right, prev_v, prev_body_omega, prev_head_omega]` (5,). Action `Box([v, body_omega, head_omega])` in [-1, 1] (3,). Head and body rotate on independent axes; sensor positions track `heading + head_relative_angle`.
- `geometry.py` — `wrap_angle`, `sensor_positions(x, y, heading, spacing)`.
- `odor_field.py` — `GaussianOdorField` with 3 noise stages: 0=clean, 1=static white noise, 2=temporally correlated AR(1) field.
- `events.py` — `classify_event(...)` → run / stop / low_sweep / high_cast_like / turn_like / spin_like flags.
- `parallel_runner.py` — `ParallelRunner` (subprocess fork pipe) and `VectorRunner` (in-process). Both expose `set_noise_stage(stage, strength)` for in-place curriculum advancement.

### Networks (`src/models/`)
- `connectome.py` — `Connectome` branch. Loads a sparse 387-node connectivity matrix + ORN/MBON metadata, augments with 2 sensor input nodes (left/right ORN fan-out) and `latent_dim` output nodes (MBON fan-in). Each env step runs `message_passing_steps=6` synchronous tanh updates with sensor re-injection. Edge weights are learnable scalars (~17k).
- `policy.py` — `Policy`. Actor = `Connectome` (consumes `obs[:, 0:2]`) → latent concat with efference copy `obs[:, 2:5]` → Linear → tanh-Gaussian over the 3-D action. Critic = stateless 2-layer MLP over the full 5-D obs. `evaluate_actions_sequence(obs[T,B,D], mask, action, state0)` for PPO updates.
- `networks.py` — RSAC backbones: `GRUActor`, `MLPActor`, `ConnectomeActor` (thin wrapper around `Connectome` for online sampling) + `QCritic` (GRU twin-critic component). All produce 3-D Gaussian actions matching the env.

### Agents (`src/agents/`)
- `ppo_agent.py` — `PPOTrainer` + `PPOConfig` + `RolloutBuffer`. Custom on-policy PPO with separate actor/critic optimizers (3e-4 / 1e-3), GAE, sequence-based update via `evaluate_actions_sequence`. Persistent rollout state across phases — `train(phase_timesteps=...)` can be called repeatedly to advance a curriculum.
- `rsac_agent.py` — `RSACAgent`, recurrent SAC. Backbone selection via `--rsac-actor-backbone {connectome, gru, mlp}`. Pure 3-D Gaussian (no Bernoulli cast — cast is now continuous head_omega).

## Connectome (`Connectome` branch)
- 387 base neurons (sensory / PN / LN / KC / MBON) + 2 sensor input nodes + `latent_dim` (32) output nodes
- Augmented connectivity: original edges (where weight > 0) ∪ left-sensor → left ORN ∪ right-sensor → right ORN ∪ MBON → output
- Per env step: 6 inner iterations of `inject_sensors → aggregate_messages + bias → tanh → re-inject_sensors`
- ~17k learnable edge scalars + 423 node biases

## Entry Points
- `train.py` — `--agent-type {ppo, rsac}`.
- `eval.py` — same flag, `--run-dir` required, deterministic rollouts + best-episode GIF.
- `main.py` — train + eval back-to-back.
- `replot.py` — regenerate PNG plots from `plot_data/training_metrics.json` (RSAC).
- `ipynb/PPO_framework.ipynb` — Colab-ready PPO end-to-end (clone → smoke → curriculum train → curves → eval+GIF).
- `ipynb/RSAC_framework.ipynb` — same style, RSAC episode loop with `connectome` / `gru` / `mlp` backbone toggle.
- `ipynb/DRQN_framework.ipynb` — legacy DRQN/DQN demo on the old single-sensor env (`demo/DRQN/`).

## Removed (legacy)
- `osl_env_2d.py` (single-sensor + 4-step cast lock)
- 5-population dense connectome (`_ConnectomeCell`, `ConnectomeExtractor`, `_HybridHead` with Bernoulli cast)
- DRQN agent (discrete cast no longer exists)
- sb3 / sb3-contrib dependency

## Reference
Run commands and CLI flags: `MANUAL.md`. Migration plan: `~/.claude/plans/b-transient-ripple.md`.
