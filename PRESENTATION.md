# Odor Source Localization with RL: Behavior Modules in a GRU Policy, and a Connectome Attempt

> 7-minute talk. Slides + speaker notes + backup.
> Flow: behavior = agent action → OSL task → GRU policy → training results → analysis tools (top-k, ablation, dynamics) → findings → connectome attempt & next steps.

---

## Slide 1 — Behavior as agent action; the OSL task

### Show
- A plume heatmap with an agent trajectory reaching the source (one frame of the best-seed gif).
- The action space, framed as "behavior."

### Content
- **Behavior is defined as the agent's action.** The 3-D continuous action is: forward speed, body angular velocity, **head angular velocity (casting)**. From these we later label discrete behaviors (RUN / TURN / CAST / SPIN / STOP).
- **Task = Odor Source Localization (OSL)**: reach an unseen odor source using only local concentration. Gaussian plume (σ=30mm, models slow fluid/solid diffusion). Spawn 55–70mm away, success within 7.5mm, episode ≤1200 steps.
- **Partially observed (POMDP)**: only current concentration is visible; direction must be integrated over time in the hidden state.
- **Observation (6-D)**: left/right sensor, **dlog** (temporal change of log-concentration — the "am I going up-gradient?" cue), and 3 efference-copy channels.

### Speaker notes
> "We frame behavior as the agent's action — forward, body turn, and an independent head turn that lets *casting* emerge. The task is odor source localization: reach an invisible source from local concentration only. It's partially observed — the agent sees concentration, not direction, so it must integrate over time. The key signal is dlog, the temporal change in concentration."

---

## Slide 2 — Policy: a GRU recurrent network

### Show
- Simple block diagram: obs(6) → GRU(hidden 421) → action(3), value head.

### Content
- **GRU recurrent layer** carries the hidden state that integrates direction over time (needed for POMDP).
- Trained with **PPO** (on-policy), success-radius curriculum (20mm → 7.5mm), then a noise curriculum.
- ~542K parameters.

### Speaker notes
> "The policy is a GRU recurrent network. The recurrence is essential — it's how the agent remembers direction in a partially observed task. We train it with PPO, gradually tightening the success radius and then adding plume noise."

---

## Slide 3 — Training results: the GRU learns

### Show
- **Training curves** (success rate, distance-to-source vs steps) — clean convergence.
- **Best-seed trajectory gif** (`plots/agent_seed20000.gif`).
- A small table of success vs noise stage/strength.

### Content
- Clean field: **success ~74–88%**, distance 55–70mm → **7.5mm** (target reached); stable entropy (~0.36), converged by ~2.4M steps.
- Holds under mild noise; degrades as dynamic-plume noise grows (see analysis).
- **Important caveat**: this policy only solves the task with **stochastic actions** — deterministic (mean action) gives ~0% success. Exploration noise is functionally required by the weak odor signal. (All evaluation/analysis below is stochastic.)

### Speaker notes
> "The GRU learns it. From 55–70mm out it reaches the 7.5mm target, ~74–88% success on the clean field. One important detail we found: the policy only works when it acts stochastically — evaluate it deterministically and it fails completely. The exploration randomness is part of how it solves a weak-signal task, so all our analysis uses stochastic rollouts."

---

## Slide 4 — Analysis tools: how we open the policy (brief, principled)

### Show
- A 4-step pipeline arrow: label → decode (top-k) → dynamics → causal ablation.

### Content (keep short, just name the methods we'll use)
- **Behavior labeling**: classify each timestep into RUN/TURN/CAST/SPIN/STOP from kinematics.
- **Linear decoding + top-k neurons**: fit a logistic-regression probe `h_t → behavior`; neuron contribution = `|W[behavior, neuron]|`; take the **top-k neurons per behavior** as that behavior's module.
- **Dynamics**: local Jacobian `J_t = ∂h_{t+1}/∂h_t`; eigenvalues → oscillatory vs decaying modes.
- **Causal ablation**: zero a behavior's top-k neurons during live rollout; the drop in success rate = that module's causal contribution. (Correlation → intervention.)

### Speaker notes
> "To open the policy we use four tools. We label behaviors, then **decode** them from the hidden state with a linear probe — the probe weights rank which neurons code each behavior, giving a **top-k neuron set per behavior**. We look at the **dynamics** via the Jacobian's eigenvalues, and finally we **causally test** each module by zeroing its top-k neurons and measuring the success drop. Decoding finds candidates; ablation proves causation."

---

## Slide 5 — Findings

### Show
- **Per-behavior top-k overlap vs noise** (overlap_all_behaviors.png) — drops toward ~0.1.
- **Per-behavior ablation Δsuccess vs noise** (per_behavior_ablation.png).
- **top-k neuron UMAP per noise** (topk_umap_by_noise.png) — behavior clusters.

### Content (confirmed)
- **The hidden state encodes behavior**: linear probe ~77% (chance 71%), with clearly separable modules (TURN/SPIN/CAST high F1).
- **Each behavior has a distinct top-k neuron set** — functional modules exist.
- **Distinct dynamics per module**: RUN ≈ pure decay (oscillatory ~3%), CAST/STOP strongly oscillatory (25–58%) — the network implements casting's rhythm as a neural oscillation.
- **Causal ablation**: zeroing a module's top-k neurons selectively breaks that function. CAST/STOP/RUN are the strong causal drivers; TURN is comparatively redundant.
- **★ Environment-dependent reassignment ★**: as dynamic-plume noise grows (0.0→0.3), the top-k neurons coding each behavior are **almost completely reassigned** — CAST overlap vs clean drops **1.0 → 0.39 → 0.19 → 0.07**. The same behavior is carried by different neurons in different environments. (UMAP checks whether behavior clusters persist despite the reassignment.)

### Content (to verify — do not assert yet)
- Whether CAST *increases* with noise: in current stochastic traces CAST frequency stays ~75% (slightly down), so "cast rises with noise" is **not yet supported** — flagged for re-check before the talk.

### Speaker notes
> "Four findings. One: behavior is linearly decodable from the hidden state, and each behavior has its own top-k neuron set — real functional modules. Two: those modules have different dynamics — straight runs are non-oscillatory, casting and stopping are oscillatory; the network literally implements casting's rhythm as an oscillation. Three: ablation confirms causation — zeroing a module's neurons breaks that function, and casting, running, stopping are the causal drivers. Four, the most striking: when the plume becomes more variable, the neurons coding each behavior are almost entirely reassigned — clean-condition cast neurons overlap only 7% with the high-noise ones. The same function, carried by different neurons depending on the environment."

---

## Slide 6 — Connectome attempt: structure, and why it failed

### Show
- The connectome graph: 389-node larva connectivity (from CSVs) + sensor/MBON I/O nodes.
- A parameter-count comparison: GRU 542K vs connectome ~17.5K.

### Content
- **Goal**: replace the generic GRU with the *real larva connectome* as the policy's recurrent structure — a biologically constrained network.
- **How it's built (not a generic GNN — built directly from two CSVs):**
  - `weights.csv`: a **389×389 connectivity matrix** (real synapse counts; ~15.5K nonzero edges).
  - `metadata.csv`: per-node cell-type/side metadata (`is_orn`, `is_left_orn`, `is_mbon`, …) → designates **sensor input nodes (ORN)** and **output nodes (MBON fan-in)**.
  - At each env step, inject the 2 sensor readings into the ORN nodes, run **6 synchronous message-passing steps** over the fixed sparse graph (learnable per-edge scalar = synapse strength), read the policy latent from the MBON nodes.
- **Result: training failed** — 0% success, distance never decreases.
- **Likely cause**: ~**1/30 the parameters** of the GRU (17.5K vs 542K), plus a sparse, 6-hop-recurrent structure → weak/slow gradients and low capacity. The biological constraint that saves parameters appears to cost trainability.

### Speaker notes
> "Finally, the connectome attempt. Instead of a generic GRU, we wanted the *real* larva wiring as the policy. It's built directly from two CSVs — a 389×389 connectivity matrix of real synapse counts, and per-neuron metadata that tags which nodes are sensory inputs and which are outputs. Each step we inject the sensors, run six message-passing rounds over this fixed sparse graph with learnable synapse strengths, and read out the action.
> It failed to learn — zero success. The likely reason is scale: about one-thirtieth the parameters of the GRU, in a sparse six-hop recurrent graph, gives weak gradients and limited capacity. The biological constraint that saves parameters seems to cost trainability."

---

## Slide 7 — Next steps (2 weeks) & open questions

### Show
- Short bullet list; invite feedback.

### Content
- **Consolidate 3D results** (the 3D version of this pipeline) and report them.
- **Diagnose the connectome failure from an RL/ML angle**, not just "too few params":
  - Candidate: **initial-weight normalization / scaling** — sparse 6-hop tanh recurrence may vanish or saturate at init (we already saw a residual-accumulation divergence when scaling node features; init scaling is the natural next suspect).
  - Other angles: gradient flow through 6 unrolled hops, learning-rate/optimizer per-structure, capacity vs trainability trade-off.
- **Feedback welcome** — especially on making the biologically-constrained network trainable.

### Speaker notes
> "Over the next two weeks I'll consolidate the 3D results and dig into *why* the connectome failed — from an RL/ML angle, not just 'too few parameters.' My current suspicion is initialization: a sparse six-hop tanh recurrence can vanish or saturate at init, so weight normalization or init scaling is the first thing I want to try. I'd really welcome feedback on getting the biologically-constrained network to train. Thank you."

---

## Appendix A — Confirmed numbers (cheat sheet)

```
[Task]  Gaussian plume σ=30mm, spawn 55–70mm, success 7.5mm, ≤1200 steps, POMDP
[Obs 6-D] L-sensor, R-sensor, dlog, fwd-speed, body-ω, head-ω
[Action 3-D] forward, body-ω, head-ω(cast)

[GRU]  hidden 421, ~542K params, PPO, success-radius curriculum 20→7.5mm
       clean success ~74–88%, reaches 7.5mm, entropy ~0.36 stable, ~2.4M steps
       ★ deterministic eval = 0% success → ALL analysis uses stochastic rollouts

[Decoding]  linear probe acc 0.77 (chance 0.71); per-behavior top-k(=16) neuron sets
[Dynamics]  |λ|max 0.93–0.99; oscillatory: RUN 3% / TURN 8% / CAST 25% / STOP 58%
[Ablation]  zeroing top-k → CAST/STOP/RUN strong Δ−; TURN ~redundant (stochastic eval)
[Noise sweep (stage2 × 0.0/0.1/0.2/0.3)]
   CAST top-k overlap vs clean: 1.00 → 0.39 → 0.19 → 0.07  (all behaviors drop similarly)
   success: 0.83 / 0.79 / 0.79 / 0.56
   CAST frequency: ~75% → ~72% (NOT increasing — flagged, verify before talk)

[Connectome]  weights.csv 389×389 (~15.5K edges, real synapse counts) + metadata.csv
              (cell-type/side, is_orn/is_mbon) → ORN inputs, MBON outputs
              6 message-passing steps/env-step, per-edge scalar = synapse strength
              ~17.5K params (1/30 of GRU) → training FAILED (0% success)
```

---

## Appendix B — Figure locations

```
runs/ppo_gru_nb_20260531_113633/
├── plots/agent_seed20000.gif                 # best-seed trajectory (Slide 1/3)
└── analysis/
    ├── (training_curves_clean.png)           # success/distance vs steps (Slide 3) — regen in notebook cell 9
    ├── overlap_all_behaviors.png             # per-behavior top-k overlap vs noise (Slide 5)
    ├── per_behavior_ablation.png             # per-behavior ablation Δsuccess vs noise (Slide 5)
    ├── topk_umap_by_noise.png                # top-k neuron UMAP per noise (Slide 5)
    ├── phase2c_neuron_heatmap.png            # neuron contribution heatmap (Slide 4/5)
    └── phase3a_eigvals_final.png             # Jacobian eigenvalues (dynamics, Slide 5)
```

> Sweep figures are produced by `ipynb/Noise_Sweep_Analysis.ipynb` (run top-to-bottom; traces cached).

---

## Appendix C — Open self-check before the talk

- [ ] **CAST-vs-noise frequency**: confirm whether casting actually rises in any sub-region (e.g. near source) before claiming it. Current global trace says no.
- [ ] **Sample size**: noise 0.2–0.3 ablation/overlap numbers are still a bit noisy — bump `EPISODES_PER_SEED`/`ABLATION_EPS` for clean curves.
- [ ] **UMAP**: confirm behavior clusters persist across noise (supports "reassignment, not collapse").
- [ ] **Connectome init fix**: try weight-normalization / init scaling on the sparse 6-hop recurrence; re-test trainability.

---

## Appendix D — Tone

- Lead with the **method** ("how we opened the policy"), let findings follow.
- State only what's confirmed; keep CAST-frequency claim flagged until verified.
- Frame the connectome as an **honest negative result + concrete next step** (init normalization), and explicitly invite feedback.
- Titles on every figure describe **what is plotted** (English), not conclusions.
