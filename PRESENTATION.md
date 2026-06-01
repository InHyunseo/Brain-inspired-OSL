# Odor Source Localization with RL: Active Sensing in a GRU Policy

## Task

- **Odor Source Localization (OSL):** reach an unseen source from local odor concentration only. Gaussian plume (σ = 30 mm), spawn 55–70 mm from the source, success within 7.5 mm, episode ≤ 1200 steps.
- **POMDP:** only the current concentration is observed; direction must be integrated over time.
- **Observation (6-D):** left sensor, right sensor, `dlog` (temporal change of log-concentration), and 3 efference-copy channels (previous forward speed, body-ω, head-ω).
- **Action (3-D, continuous):** forward speed, body angular velocity, **head angular velocity** — the channel that lets active sensing / casting emerge.
- Behavior labels are derived post-hoc from the action/kinematics.

## Baseline: hand-built bilateral chemotaxis (no network)

A classical controller: steer toward the stronger sensor (`body_ω = tanh(gain · (c_L − c_R)/(c_L + c_R))`), drive forward while concentration is rising, stop while it falls. Head never moves (no active sensing). No learning.

Success vs noise (n = 200 episodes / condition):

| condition | stage 0, α0.0 | stage 1, α0.3 | α0.6 | α1.0 | stage 2, α0.3 | α0.6 | α1.0 |
|---|---|---|---|---|---|---|---|
| success | **100%** | 100% | 91% | 46% | 100% | 96% | 50% |
| steps (success) | 455 | 482 | 573 | 686 | 595 | 828 | 1021 |

- **Solves the clean field on every episode** (100%, ~455 steps).
- **Degrades under turbulence:** down to ~46–50% at the hardest noise (α1.0), with steps-to-source roughly doubling.
- Fixed rule, no learning — the same gradient-following in every environment.

## Policy: GRU recurrent network

- `obs(6) → GRU(hidden 421) → action(3) + value head`. Hidden state integrates direction over time (required for the POMDP).
- Trained with **PPO** (on-policy), success-radius curriculum (20 mm → 7.5 mm) then a noise curriculum. ~542K parameters.
- Eval success reaches **100%** on the clean field by ~2.4M steps.
- **Only solves the task with stochastic actions** — deterministic (mean) rollouts give ~0% success. The weak signal makes exploration noise functionally required, so **all analysis below uses stochastic rollouts.**

## Analysis pipeline

`label → decode (top-k neurons) → dynamics (Jacobian) → causal ablation`, on hidden-state traces.

- **Labels (3-way):** RUN (locomotion, a trivial control), **ACTIVE_SENSING** (head-sweep casting — the behavior of interest), OTHER (everything else; the old TURN label was a ~75% catch-all and STOP/SPIN were <0.2%).
- **Decoding:** logistic probe `h_t → behavior`; neuron contribution = `|W[behavior, neuron]|`; top-k (= 16) neurons per behavior define that behavior's module.
- **Dynamics:** local Jacobian `J_t = ∂h_{t+1}/∂h_t`; its dominant eigenvalue (largest |λ|) is oscillatory (Im ≠ 0) or pure-decay (Im = 0).
- **Ablation:** zero a behavior's top-k neurons during a live rollout; the success drop measures causal contribution.

## Findings: active sensing

- **Its dynamics oscillate; RUN's do not.** Dominant-eigenvalue oscillation fraction: **active sensing 0.50 vs RUN 0.03**. The head-sweep rhythm appears as a neural oscillation.
- **The neurons implementing it are reassigned as the environment changes.** Active-sensing top-k overlap with the clean set, across stage-2 noise 0.0 → 0.1 → 0.2 → 0.3: **1.00 → 0.68 → 0.28 → 0.10**. Same behavior, different neurons per environment.
- **The behavior is linearly decodable above chance** (probe accuracy 0.76 vs shuffle 0.75 — weak but present), with **partial** hidden-state separation (slightly negative silhouette).

Caveats: the Jacobian is a local linearization read off the dominant mode; hidden-state clusters overlap (separation is partial, not clean).

## Connectome attempt (negative result)

- **Goal:** replace the GRU with the real *Drosophila* larva connectome as the recurrent policy. Built from `weights.csv` (389×389 connectivity, ~15.5K nonzero edges) + `metadata.csv` (per-node cell type/side → ORN sensor inputs, MBON outputs). Each env step injects the 2 sensor readings into ORN nodes, runs 6 message-passing steps over the fixed sparse graph (learnable per-edge scalar = synapse strength), reads the latent from MBON nodes.
- **Result:** training failed — ~0% success, distance never decreases. Also tried gradient-free evolution strategies on a spiking version → still ~0%.
- **Likely cause:** ~1/30 the parameters of the GRU (~17.5K) and sparse 6-hop recurrence → weak/slow gradients, low capacity. The biological constraint that saves parameters appears to cost trainability.

## Figures (`presentation_assets/`)

```
slide1_4_trajectory.gif            policy trajectory (plume + agent reaching source)
slide2_baseline_noise_sweep.png    baseline success & steps vs noise (mean ± per-seed)
baseline_s2_1p0.gif                baseline trajectory at the hardest noise
slide4_training_curves.png         eval success ratio vs training steps
slide6_active_sensing_overlap.png  active-sensing top-k overlap vs clean (box plot)
slide6_jacobian_run.png            RUN dominant eigenvalues (on real axis = decay)
slide6_jacobian_active_sensing.png active-sensing eigenvalues (off-axis = oscillation)
slide6_jacobian_oscillation.png    oscillation fraction: RUN 0.03 vs active sensing 0.50
```

All figures are regenerated by `ipynb/Presentation_assets.ipynb` (set `RUN_DIR`/`CKPT_LABEL` at the top, Run all). `RUN_DIR = runs/ppo_gru_nb_20260531_113633`.
