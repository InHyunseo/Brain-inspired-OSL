# Odor Source Localization with RL: Active Sensing in a GRU Policy, and a Connectome Attempt

> 7-minute talk. **Outline only** (bullets are slide content, not a script).
> Slides are written in **English**; delivered orally in **Korean**.
> Flow: behavior = agent action → OSL task → classical baseline → GRU policy → training results → analysis tools → findings (active sensing) → connectome attempt → next steps.
>
> **PNG/GIF column**: each slide lists the exact asset file to place on it. Paths are
> relative to the repo root. `RUN = runs/ppo_gru_nb_20260531_113633`,
> `BASE = runs/baseline_chemotaxis`.

---

## Slide 1 — Behavior = agent action; the OSL task

**Asset:** `RUN/plots/agent_seed20000.gif` (one frame = plume heatmap + agent trajectory reaching source)

- **Behavior is defined as the agent's action.** 3-D continuous action: forward speed, body angular velocity, **head angular velocity** (the channel that lets *active sensing / casting* emerge). Discrete behavior labels are derived from these post-hoc.
- **Task = Odor Source Localization (OSL):** reach an unseen source from local concentration only. Gaussian plume (σ=30 mm). Spawn 55–70 mm away, success within 7.5 mm, episode ≤ 1200 steps.
- **Partially observed (POMDP):** only current concentration is visible; *direction* must be integrated over time.
- **Observation (6-D):** L sensor, R sensor, **dlog** (temporal change of log-concentration — the "am I going up-gradient?" cue), + 3 efference-copy channels.

---

## Slide 2 — Baseline first: a hand-built chemotaxis controller

**Asset:** `BASE/noise_sweep.png` (success rate vs noise for the classical FSM)

- Before any learning: a **classical bilateral-chemotaxis FSM** (no network). Steer toward the stronger sensor at fixed gain; switch to head-sweep ("cast") mode when the signal weakens. Textbook biological strategy, hand-coded.
- **Solves the clean task: 100% success** (mean 456 steps); holds under mild noise (98–100% at strength 0.3).
- **Brittle to turbulence:** drops to **~42–47%** at the hardest noise (strength 1.0).
- **Its casting is a fixed heuristic** — same rule regardless of environment.
- **Motivation for RL:** the hand-built controller works but is rigid and degrades under noise. Can a *learned* policy be more robust — and discover *for itself* a strategy (e.g. *when* to cast)? (We revisit this in Slide 7: the RL policy adapts its casting to the environment; the FSM cannot.)

---

## Slide 3 — Policy: a GRU recurrent network

**Asset:** block diagram (make in slides) — `obs(6) → GRU(hidden 421) → action(3) + value head`

- **GRU recurrent layer** carries the hidden state that integrates direction over time (required for POMDP).
- Trained with **PPO** (on-policy); success-radius curriculum (20 mm → 7.5 mm), then a noise curriculum.
- ~542K parameters.

---

## Slide 4 — Training results: the GRU learns

**Assets:**
- `RUN/plots/agent_seed20000.gif` (best-seed trajectory)
- training curves PNG — *regenerate in `ipynb` cell 9* (not yet saved to disk)

- Clean field: **success ~74–88%**, distance 55–70 mm → **7.5 mm** target reached; entropy ~0.36 stable; converged ~2.4M steps.
- Holds under mild noise; degrades as dynamic-plume noise grows (Slide 6).
- **Key caveat:** the policy only solves the task with **stochastic actions** — deterministic (mean action) gives ~0% success. Exploration noise is functionally required by the weak signal → **all analysis below uses stochastic rollouts.**

---

## Slide 5 — Analysis tools: how we open the policy

**Asset:** 4-step pipeline arrow (make in slides): **label → decode (top-k) → dynamics → causal ablation**

- **Behavior labeling:** classify each timestep. We keep **3 labels** — RUN (locomotion, trivial control), **ACTIVE_SENSING** (head-sweep "casting", the behavior of interest), OTHER (everything else). (Old TURN was a ~75% catch-all; STOP/SPIN were <0.2% and unreliable.)
- **Linear decoding + top-k neurons:** logistic probe `h_t → behavior`; neuron contribution = `|W[behavior, neuron]|`; take **top-k neurons per behavior** = that behavior's module.
- **Dynamics:** local Jacobian `J_t = ∂h_{t+1}/∂h_t`; its **dominant eigenvalue** (largest |λ|) tells whether the governing mode oscillates (Im≠0) or just decays (Im=0).
- **Causal ablation:** zero a behavior's top-k neurons during live rollout; success drop = causal contribution. (Ran it, but the deltas were ambiguous at high noise — not shown.)

---

## Slide 6 — Findings: active sensing

> We focus on the one behavior that is both well-defined and interesting: **active sensing** (stop + head-sweep to read the gradient). RUN is a trivial control; everything else is OTHER.

**Assets (in `presentation_assets/`):**
- `slide6_active_sensing_overlap.png` — active-sensing top-k overlap vs clean (drops toward ~0.10)
- `slide6_jacobian_run.png` + `slide6_jacobian_active_sensing.png` — eigenvalue clouds; grey = all modes, red = the dominant mode per timestep
- `slide6_jacobian_oscillation.png` — dominant-mode oscillation fraction, Run 0.03 vs Active sensing 0.50

- **Active sensing rises sharply with noise:** frequency **0.7% → 0.6% → 1.9% → 5.9%** as dynamic-plume noise grows 0.0 → 0.3 (**≈8×** clean→noisy). Unstable signal → more active sensing. Biologically expected.
- **The neurons implementing it are reassigned with the environment:** active-sensing top-k overlap vs clean = **1.00 → 0.68 → 0.28 → 0.10**. Same behavior, different neurons per environment.
- **Its dynamics oscillate, RUN's don't:** the dominant Jacobian eigenvalue is oscillatory (Im≠0) for **0.50** of active-sensing timesteps vs **0.03** for RUN — the head-sweep rhythm shows up as a neural oscillation. Linear probe also decodes behavior above chance (**0.77 vs 0.71**).

**Honest caveats (state briefly):**
- Hidden-state separation is *partial* (probe 0.77, slightly negative silhouette).
- Jacobian is a **local linearization** read off the **dominant** mode — standard, and conservative for our claim, but not the full nonlinear picture.

---

## Slide 7 — Connectome attempt: structure & why it failed

**Assets:**
- connectome graph diagram (make in slides): 389-node larva connectivity + ORN inputs / MBON outputs
- param-count comparison (make in slides): **GRU 542K vs connectome ~17.5K (1/30)**

- **Goal:** replace the generic GRU with the *real larva connectome* as the recurrent policy — a biologically constrained network. **(Built directly from two CSVs, not a generic GNN.)**
  - `weights.csv`: **389×389** connectivity (real synapse counts; ~15.5K nonzero edges).
  - `metadata.csv`: per-node cell-type/side (`is_orn`, `is_mbon`, …) → designates **ORN sensor inputs** and **MBON outputs**.
  - Per env step: inject 2 sensor readings into ORN nodes, run **6 message-passing steps** over the fixed sparse graph (learnable per-edge scalar = synapse strength), read latent from MBON nodes.
- **Result: training failed** — ~0% success, distance never decreases.
- **Robust to the learning algorithm:** also tried gradient-free **evolution strategies** on a spiking version → still ~0% success.
- **Likely cause:** ~**1/30 the parameters**, sparse 6-hop recurrence → weak/slow gradients, low capacity. The biological constraint that saves parameters appears to cost trainability.

---

## Slide 8 — Next steps (2 weeks) & open questions

**Asset:** none (bullet list; invite feedback)

- **Contrast with the baseline (callback to Slide 2):** the FSM's casting is a *fixed* rule; the RL policy *adapts* its active sensing to the environment (≈8× rise with noise). That adaptivity is the payoff of learning.
- **Next 2 weeks:**
  - Consolidate and report the **3D results**.
  - Diagnose the connectome failure from an ML angle (init/weight-scaling for the sparse 6-hop recurrence is the first suspect; capacity vs trainability).
- **Feedback welcome** — especially on training biologically-constrained networks.

---

## Appendix A — Confirmed numbers (cheat sheet)

```
[Task]  Gaussian plume σ=30mm, spawn 55–70mm, success 7.5mm, ≤1200 steps, POMDP
[Obs 6-D]  L-sensor, R-sensor, dlog, fwd-speed, body-ω, head-ω
[Action 3-D]  forward, body-ω, head-ω (active sensing / cast)

[Baseline FSM]  hand-coded bilateral chemotaxis (steer + cast mode), no network
   success vs noise:  clean 100% (456 steps) → 0.3: ~98–100% → 1.0: ~42–47%
   forcing more casting only hurts (fixed rule): 47% → 13% → 3% as cast threshold raised

[GRU]  hidden 421, ~542K params, PPO, success-radius curriculum 20→7.5mm
       clean success ~74–88%, reaches 7.5mm, entropy ~0.36 stable, ~2.4M steps
       ★ deterministic eval = 0% success → ALL analysis uses stochastic rollouts

[Labels]  3-way: RUN / ACTIVE_SENSING (head-sweep "casting") / OTHER
          (old TURN ~75% catch-all; STOP/SPIN <0.2% — merged into OTHER)
[Decoding]  linear probe acc 0.77 (chance 0.71); top-k(=16) neurons per behavior
[Dynamics]  active sensing = oscillatory hidden dynamics; RUN ≈ pure decay
[Noise sweep (stage2 × 0.0/0.1/0.2/0.3), stochastic eval]
   ★ active-sensing FREQUENCY: 0.7% → 0.6% → 1.9% → 5.9%  (≈8× rise with noise)
   ★ active-sensing top-k overlap vs clean: 1.00 → 0.45 → 0.33 → 0.14 (reassigned)
   episode success rate: 0.84 / 0.84 / 0.81 / 0.54

[Connectome]  weights.csv 389×389 (~15.5K edges) + metadata.csv (is_orn/is_mbon)
              6 message-passing steps/env-step, per-edge scalar = synapse strength
              ~17.5K params (1/30 of GRU) → training FAILED (0%), incl. ES on SNN version
```

---

## Appendix B — Asset inventory (final, in `presentation_assets/`)

```
presentation_assets/
├── slide1_4_trajectory.gif             # Slide 1 & 4 — best-seed trajectory (plume + agent)
├── slide2_baseline_noise_sweep.png     # Slide 2 — baseline success vs noise (blue)
├── slide4_training_curves.png          # Slide 4 — Success ratio vs steps, phase 0/1/2 markers
├── slide6_active_sensing_overlap.png   # Slide 6 — neuron overlap vs clean (1.0 → 0.10)
├── slide6_jacobian_run.png             # Slide 6 — RUN eigenvalues (dominant on real axis = decay)
├── slide6_jacobian_active_sensing.png  # Slide 6 — AS eigenvalues (dominant off-axis = oscillation)
└── slide6_jacobian_oscillation.png     # Slide 6 — oscillation fraction 0.03 vs 0.50
```

All seven are produced by `ipynb/PPO_GRU_framework.ipynb` (training-curve cell + analysis cell).

> Draw in the slide tool (not auto-generated):
> - Slide 3: GRU block diagram `obs(6) → GRU(421) → action(3)+value`.
> - Slide 5: 4-step pipeline arrow `label → decode → dynamics → ablation`.
> - Slide 7: connectome graph + GRU-vs-connectome param-count bars.

---

## Appendix C — Pre-talk self-check

- [ ] All seven PNG/GIF present in `presentation_assets/` (re-run the notebook's plot cells if stale).
- [ ] Confirm overlap / jacobian figures are the **3-label (active-sensing)** versions.
- [ ] Draw the three slide-tool diagrams (Slides 3, 5, 7).
- [ ] (If asked) connectome init fix: try weight-norm / init scaling on the sparse 6-hop recurrence.

---

## Appendix D — Tone

- Lead with **method** ("how we opened the policy"), let findings follow.
- State only what's **confirmed**. The connectome is an **honest negative result + concrete next step** (init normalization); explicitly invite feedback.
- Every figure title describes **what is plotted** (English), not a conclusion.
- Slides in English; deliver in Korean.
