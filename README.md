# Odor Source Localization (Larva Connectome RL)

Bilateral-sensor odor source localization with an independent head/body rotation
axis and a 387-node *Drosophila* larva connectome as the actor backbone.
Both PPO and SAC are first-class trainers and share the same env, parallel
runner, curriculum loop, connectome model, logging, checkpoint, and eval code —
only the optimisation algorithm differs.

## Project Goals

- Bilateral plume sensing with biologically realistic 0.15 mm sensor spacing
- Independent head-axis control (head casting via continuous `head_omega`)
- Real larva connectome as the actor (CSV-driven message passing, ~17k learnable edges)
- Efference copy in observation so the policy can disentangle self-motion from plume change
- Bump-field odor noise model that interpolates from clean Gaussian to
  hydrodynamic-feeling turbulence via a single α ∈ [0, 1] curriculum scalar

## Code Structure

### Environment (`src/envs/`)

- `osl_env.py` — `OslEnv` + `EnvConfig`. Observation `[c_left, c_right, prev_v, prev_body_omega, prev_head_omega]` (5,). Action `Box([v, body_omega, head_omega])` in [-1, 1] (3,). Head and body rotate on independent axes; sensor positions track `heading + head_relative_angle`. Spawn policy: rejection-sampled annulus around the source (cue zone ∩ `[r_min, r_max]`); heading initialised toward the source with Gaussian error.
- `geometry.py` — `wrap_angle`, `sensor_positions(x, y, heading, spacing)`.
- `odor_field.py` — `GaussianOdorField` with a bump-field perturbation model: many independent local Gaussian bumps with signed amplitudes, drifting and AR(1)-modulated. Stage 0 = clean. Stage 1 = static bumps (frozen at reset). Stage 2 = dynamic bumps (advance per env step). One scalar α scales every bump parameter (count, amplitude, drift speed, lifecycle rate, respawn probability). Per-bump sigma is capped near the source so the global plume gradient is preserved at full α.
- `events.py` — `classify_event(...)` → run / stop / low_sweep / high_cast_like / turn_like / spin_like flags.
- `parallel_runner.py` — `ParallelRunner` (subprocess fork pipe) and `VectorRunner` (in-process). Both expose `set_noise_stage(stage, strength)` for in-place curriculum advancement.

### Networks (`src/models/`)

- `connectome.py` — `Connectome` branch. Loads a sparse 387-node connectivity matrix + ORN/MBON metadata, augments with 2 sensor input nodes (left/right ORN fan-out) and `latent_dim` output nodes (MBON fan-in). Each env step runs `message_passing_steps=6` synchronous tanh updates with sensor re-injection. Edge weights are learnable scalars (~17k).
- `policy.py` — `Policy` (PPO). Actor = `Connectome` (consumes `obs[:, 0:2]`) → latent concat with efference copy `obs[:, 2:5]` → Linear → tanh-Gaussian over the 3-D action. Critic = stateless 2-layer MLP `V(s)`. `evaluate_actions_sequence(...)` for PPO sequence updates.
- `networks.py` — legacy SAC backbones (`GRUActor`, `MLPActor`, `ConnectomeActor` + `QCritic`) kept for reference.

### Agents (`src/agents/`)

- `ppo_agent.py` — `PPOTrainer` + `PPOConfig` + `RolloutBuffer`. Custom on-policy PPO with separate actor/critic optimizers, GAE, sequence-based update via `evaluate_actions_sequence`. Persistent rollout state across phases — `train(phase_timesteps=...)` can be called repeatedly to advance a curriculum.
- `sac_agent.py` — `SACTrainer` + `SACConfig` + `SACPolicy` + `ReplayBuffer`. Off-policy SAC with twin Q critics + Polyak-averaged targets + auto-tuned entropy temperature α. Reuses `parallel_runner`, the connectome actor, and the same curriculum surface as PPO (`trainer.runner.set_noise_stage(...)` then `trainer.train(phase_timesteps=...)`).

## Connectome (`Connectome` branch)

- 387 base neurons (sensory / PN / LN / KC / MBON) + 2 sensor input nodes + `latent_dim` (32) output nodes
- Augmented connectivity: original edges (where weight > 0) ∪ left-sensor → left ORN ∪ right-sensor → right ORN ∪ MBON → output
- Per env step: 6 inner iterations of `inject_sensors → aggregate_messages + bias → tanh → re-inject_sensors`
- ~17k learnable edge scalars + 423 node biases

## Entry Points

- `train.py` — `--agent-type {ppo, sac}`. Curriculum from `--curriculum-phases` JSON.
- `eval.py` — same flag, `--run-dir` required, deterministic rollouts + best-episode GIF.
- `main.py` — train + eval back-to-back.
- `visualize_curriculum_field.py` — render per-phase odor-field snapshot PNG + dynamic GIF (env stepping at rest).
- `ipynb/PPO_framework.ipynb` — Colab / local end-to-end PPO (clone → smoke → curriculum train → curves → curriculum-field viz → elite-seed eval + GIF).
- `ipynb/SAC_framework.ipynb` — same surface as the PPO notebook with the SAC trainer + auto-α / Q-mean curves.
- `ipynb/DRQN_framework.ipynb` — legacy DRQN/DQN demo on the old single-sensor env (`demo/DRQN/`).

## Reference

Run commands and CLI flags: `MANUAL.md`. Methodology and analysis plan: `analysis.md`.
