# Poster Plan — OSL: Larva-Inspired RL for Odor-Source Localization (2D + 3D)

> **Spec.** A0 portrait (841 × 1189 mm) · English · academic conference poster
> **Structure.** **Two independent tracks**, side by side, each method-forward and
> self-contained: **Track A — 2D** (In Hyun Seo; PPO, custom sim) and **Track B — 3D**
> (teammate; RSAC, ROS2+Gazebo+GADEN). A thin shared band across the bottom delivers
> the **cross-validated common conclusion**.
> **Takeaway (single sentence, top-of-poster).**
> *We test the same question — can a **larva-connectome actor** localize an odor
> source as well as a GRU? — in two **independent** RL pipelines (2D/PPO and
> 3D/RSAC). GRU learns it in both; the connectome backbone is **unstable in both** —
> cross-validated evidence for a shared capacity/trainability bottleneck, not an
> artifact of one setup.*

This is a **layout + copy blueprint** — finalized copy + a slot for every figure.
**Figure slots are placeholders until final assets are exported.** Hand to
LaTeX `beamerposter`/`tikzposter`, Canva, Figma, or PowerPoint to render.

**Conventions.**

- **Backbone color** (shared across both tracks): GRU = teal · connectome = orange ·
  hand-baseline (2D only) = grey.

- **Track frame:** Track A (2D) = one frame hue, Track B (3D) = another. The two
  tracks are visually separated columns; only the bottom band is shared.

- **Terminology (2D):** say **"active sensing"** (independent head axis sweeping),
  not "cast".

---

## 0. Design grid — two tracks + shared band

A0 portrait. A full-width header and a short shared intro; then the page splits into
**two vertical tracks** (2D left, 3D right), each running its own
Setup → Method → Results top-to-bottom; a shared conclusion band spans the bottom.

```text
┌───────────────────────────────────────────────────────────────────┐
│  HEADER  (title · authors[A=2D / B=3D] · affiliation · QR)          │ ~8%
├───────────────────────────────────────────────────────────────────┤
│  SHARED INTRO  (the common question: connectome vs GRU actor, OSL)  │ ~6%
├─────────────────────────────────┬─────────────────────────────────┤
│  TRACK A — 2D OSL  (PPO)         │  TRACK B — 3D OSL  (RSAC)        │
│  In Hyun Seo                     │  [teammate]                     │
│                                  │                                 │
│  A1. Environment (2D field)      │  B1. Environment (3D plume,     │
│  A2. Obs / Action                │       ROS2+Gazebo+GADEN)        │
│  A3. Method: PPO + curriculum    │  B2. Obs (odor/pose/wind/detect)│ ~64%
│       + actor (conn ∥ GRU)       │  B3. Method: RSAC + actors      │
│  A4. Results: success / steps /  │       (GRU ∥ connectome+ctxMLP) │
│       active-sensing vs α        │  B4. Results: GRU (success/step/│
│  A5. Jacobian eigenmode (probe)  │       traj) — stable            │
│                                  │  B5. Results: connectome —      │
│                                  │       partial / unstable        │
├─────────────────────────────────┴─────────────────────────────────┤
│  SHARED CONCLUSION BAND                                              │
│  Cross-validated: GRU works in both; connectome unstable in both →   │ ~14%
│  capacity/trainability bottleneck + per-track future work.           │
├───────────────────────────────────────────────────────────────────┤
│  FOOTER  (refs · repo · contact · funding)                          │
└───────────────────────────────────────────────────────────────────┘
```

Visual weight: the **two result blocks** (A4/A5 and B4/B5) anchor each track. A
reader scanning only the header + bottom band still gets the cross-validated message.

### PowerPoint scaffold notes

- **Canvas:** A0 portrait in PowerPoint custom size, 33.11 × 46.81 in
  (841 × 1189 mm). Export final as PDF for printing.
- **Style target:** match `poster_example.pdf`: white background, dark-blue
  section bars, thin black card outlines, large figure-first result blocks, and a
  full-width shared conclusion band.
- **Placeholder workflow:** keep every gray dashed box as an editable figure slot.
  Replace with final PNG/PDF assets only after the 2D and 3D result figures are
  frozen; preserve the slot labels (`A4a`, `B4c`, etc.) until the final pass.
- **Generated scaffold:** `output/poster/osl_2d_3d_poster_scaffold.pptx`
  with preview `output/poster/osl_2d_3d_poster_scaffold_preview.png`.

---

## HEADER + SHARED INTRO

- **Title:** *Larva-Connectome vs GRU Actors for Odor-Source Localization:
  A Cross-Validation Across 2D (PPO) and 3D (RSAC) RL Pipelines*

- **Authors:** In Hyun Seo† , [teammate]‡ — Hanyang University. †2D/PPO · ‡3D/RSAC.
  *(fill advisor / lab)*

- **QR** → repo. **Subtitle** = the Takeaway sentence.
- **SHARED INTRO copy:** "A *Drosophila* larva localizes odor from local
  concentration over a compact connectome-scale circuit. We ask whether that real
  wiring, used as an RL **actor backbone**, can match a plain GRU — and we test it twice, in
  **two independently built pipelines** (different algorithm, different simulator,
  different observation space). Agreement across both is the evidence."

- **FIGURE 0 (placeholder)** — larva chemotaxis + connectome cartoon. *Hand-made, shared.*

---

## TRACK A — 2D OSL (PPO, custom sim) · In Hyun Seo

### A1. Environment

*Copy:* "2D arena, local bilateral odor cues + self-motion history, but **no map or
source location** → a POMDP (direction integrated over time). Plumes interpolate
clean Gaussian → bump-field turbulence via one scalar α∈[0,1]. Body and head rotate
on **independent axes**, so head-sweeping **active sensing** can emerge rather than
being hand-coded."

- **Arena:** 80 × 120 mm, source (40, 100), Gaussian σ = 30 mm, success radius 7.5 mm,
  120 s episodes (Δt = 0.1 s), sensor spacing 0.15 mm.

- **FIGURE A1 (placeholder)** — odor field across α + bilateral-sensor/independent-head schematic.
  *Asset: `visualize_curriculum_field.py`.*

### A2. Observation / Action

- **Obs (6-D):** `[c_left, c_right, dlog, prev_v, prev_body_ω, prev_head_ω]`
- **Act (3-D):** `[v, body_ω, head_ω] ∈ [−1,1]` (tanh-squashed Gaussian)
- **Reward = biological energy budget:** sparse goal +20 dominates metabolic motion
  costs (time −0.005/step, run −0.01·(v/vmax)², body-turn −0.005·ω², **head-sweep
  −0.02·ω², ×2 when stopped — the costliest action**, spin −0.05, wall −2 terminal),
  plus dlog(c)/dt shaping (k=0.05, clip ±0.5) and concentration k=0.02.

### A3. Method — PPO + curriculum + selectable actor

*Copy:* "Custom on-policy **PPO** (GAE, sequence updates) over **16 vectorized envs**,
trained through a noise **curriculum**."

- **Actor (swappable `--backbone`):** **connectome** — real connectome graph
  augmented with 2 sensor nodes + 32 readout nodes (**423 total nodes** in the current
  implementation), **6 message-passing steps/step**, tanh, **D=8 feature/node**; vs
  **GRU** hidden=421 (**state-size parity**, the capacity reference). Separate critic;
  reported runs use a stateless **MLP critic (64,64)**, with a recurrent critic
  selectable for ablations.

- **Curriculum (5 phases `[stage, α, steps]`):** (0,0.0,1.0M)→(1,0.3,0.5M)→
  (2,0.3,0.5M)→(2,0.6,0.5M)→(2,1.0,1.0M); clean→static→dynamic bumps.
  **Persistent rollout state** advances phases without restart.

- **PPO / regularization:** γ=0.99, λ=0.95, clip 0.2, ent 0.005; **adv-norm,
  value-clip, grad-clip 0.5, target-KL early-stop @0.02**; Adam (actor 3e-4 / critic 1e-3).

- **FIGURE A3 (placeholder)** — actor-critic architecture (connectome 6-hop unroll ∥ GRU)
  + 16-env curriculum schematic. *Asset: hand-made from `src/models/*` + `visualize_curriculum_field.py`.*

### A4. Results — metrics vs α

*Copy:* "GRU learns the clean field (~100%) and degrades gracefully. **Active-sensing
ratio rises with turbulence** — emergent, matching the hand-baseline's hand-coded trend."

- **FIGURE A4a (placeholder)** — success ratio + steps-to-source vs α (GRU / connectome / baseline).
  *Asset: `analysis/ppo_gru_noise_sweep_eval.py`.*

- **FIGURE A4b (placeholder) — HEADLINE** — active-sensing ratio vs α, learned vs baseline, both rising.
  *Asset: `analysis/noise_sweep_cast.py` (relabel "active-sensing ratio").*

- **FIGURE A4c (placeholder)** — trained **multi-seed trajectory overlay** (GRU, plus
  optional hand-baseline reference), active-sensing events marked.
  *Asset: `notebooks/ppo_gru.ipynb` trajectory PNG cell / `notebooks/baseline.ipynb`.*

### A5. Jacobian eigenmode probe

*Copy:* "Linearize the hidden dynamics (autograd) per step; segment by behavior.
**Active-sensing segments carry oscillatory modes (complex λ, |λ|≈1) that RUN
lacks** — active sensing is implemented as a hidden-state oscillator."

- $J_t = \partial f/\partial h\big|_{h_t}$, $\{\lambda_k\}=\mathrm{eig}(J_t)$;
  $f_{\rm osc}=|\arg\lambda^\star|/(2\pi\Delta t)$ vs the active-sensing PSD peak.

- **FIGURE A5 (placeholder)** — eigenvalue spectrum RUN vs ACTIVE-SENSING, unit circle.
  *Asset: `analysis/osl2d/phase3a_jacobian.py`.*

---

## TRACK B — 3D OSL (RSAC, ROS2+Gazebo+GADEN) · [teammate]

### B1. Environment

*Copy:* "A **3D gas-plume** simulation: odor source at (1.0, 3.0, 0.7), door/outlet
geometry, **+x wind with small y fluctuation**. The robot must localize the source
within a goal radius."

- **Sim stack:** ROS2 + Gazebo + **GADEN** (gas-dispersion). Higher physical fidelity
  than the 2D custom sim.

- **FIGURE B1 (placeholder)** — 3D plume + arena (source, outlet, wind direction).
  *Asset: teammate's environment renderer.*

### B2. Observation

*Copy:* "Richer observation than 2D: the robot receives **odor, pose, wind, and
detection** information."

- **Obs:** odor concentration + robot pose + wind + detection flag.
- *(confirm exact dims/units with teammate)*
- **Task:** localize source within goal radius. Max 300 steps × 0.5 s.

### B3. Method — RSAC + two actors

*Copy:* "Off-policy **RSAC** (recurrent SAC). Two actor structures compared, same as
the 2D question: a **GRU actor** vs a **connectome-based actor**."

- **GRU actor:** receives the full observation; learns temporal dependencies via
  recurrent hidden state; integrates odor / position / wind / detection.

- **Connectome actor:** uses the **connectome graph as the odor-processing pathway**;
  additional context (pose, wind, detection) integrated through a **context MLP**;
  adds **gradual state update + odor-change sensitivity** for more stable temporal behavior.

- **Reward:** odor cue + distance progress + goal hold + action penalty (stability).

| Item | Setting |
|------|---------|
| Algorithm | RSAC (recurrent SAC) |
| Sim | ROS2 + Gazebo + GADEN |
| Task | 3D odor source localization |
| Max steps | 300 (× 0.5 s/step) |
| Reward | odor cue + distance progress + goal hold + action penalty |
| Actors | GRU ∥ connectome + context MLP |

- **FIGURE B3 (placeholder)** — two-actor diagram: GRU (full-obs recurrent) ∥
  connectome (odor pathway) + context MLP (pose/wind/detect). *Asset: hand-made from teammate spec.*

### B4. Results — GRU (stable)

*Copy:* "The **GRU policy learns stably**: success rate rises and step-to-goal falls
over training; the eval trajectory shows odor-guided approach to the source region."

- **FIGURE B4a (placeholder)** — GRU success-rate curve.
- **FIGURE B4b (placeholder)** — GRU step-to-goal curve.
- **FIGURE B4c (placeholder)** — GRU eval trajectory (approaches source). *Asset: teammate.*

### B5. Results — Connectome (partial / unstable)

*Copy:* "The **connectome policy learns only partially and stays unstable**. Failed
trajectories often **drift downstream away from the source**, even when starting near
the plume."

- **FIGURE B5a (placeholder)** — connectome success-rate curve (lower / noisier).
- **FIGURE B5b (placeholder)** — connectome step-to-goal curve.
- **FIGURE B5c (placeholder)** — connectome eval trajectory (downstream drift / failure). *Asset: teammate.*

---

## SHARED CONCLUSION BAND  (spans both tracks)

*Copy (the cross-validated punchline):*

> **Same result, two independent pipelines.** A **GRU actor** learns odor-source
> localization in **both** 2D (PPO, custom sim) and 3D (RSAC, ROS2+Gazebo+GADEN).
> The **larva-connectome actor underperforms in both** — failing to train end-to-end
> in 2D, and only partially/​unstably learning (downstream drift) in 3D. Because the
> two setups share **no algorithm, simulator, or observation space**, the agreement
> points to a shared **backbone-level bottleneck**: the real connectome is a
> parameter-efficient but **capacity/trainability-limited** scaffold, not an artifact
> of any one pipeline.

- **2D adds:** active sensing **emerges and scales with turbulence** (costliest action,
  yet used); Jacobian ties it to **oscillatory hidden-state dynamics**.

- **Future:** stronger context integration + stable temporal memory + better reward
  design for connectome actors (3D); shuffled-edge / scaled-connectome controls,
  causal ablation (2D); shared analysis protocol across both.

---

## FOOTER

- **Key refs** (3–5): larva chemotaxis / active sensing; larva connectome; PPO+GAE; SAC. *→ fill.*
- **Repo:** github.com/…  · **Contact:** inhsroy@hanyang.ac.kr · **Lab/funding** *(fill)*

---

## Figure asset checklist (all PLACEHOLDER)

| # | Figure | Track | Source | Status |
|---|--------|-------|--------|--------|
| 0 | Larva + connectome cartoon | shared | hand-made | ❌ make |
| A1 | 2D field across α + sensor/head schematic | A | `visualize_curriculum_field.py` + hand-made | ⏳ |
| A3 | 2D actor-critic arch + curriculum | A | hand-made from `src/models/*` | ❌ make |
| A4a | Success / steps vs α | A | `analysis/ppo_gru_noise_sweep_eval.py` | ⏳ need run |
| A4b | **Active-sensing ratio vs α (headline)** | A | `analysis/noise_sweep_cast.py` | ⏳ need run |
| A4c | Trained multi-seed trajectory overlay | A | `notebooks/ppo_gru.ipynb` / `notebooks/baseline.ipynb` | ⏳ need run |
| A5 | Jacobian spectrum RUN vs active-sensing | A | `analysis/osl2d/phase3a_jacobian.py` | ⏳ need run |
| B1 | 3D plume + arena (source/outlet/wind) | B | teammate renderer | ⏳ from teammate |
| B3 | 3D two-actor diagram (GRU ∥ conn+ctxMLP) | B | hand-made from spec | ❌ make |
| B4a–c | GRU success / step / trajectory | B | teammate | ⏳ from teammate |
| B5a–c | Connectome success / step / trajectory | B | teammate | ⏳ from teammate |

**Status:** final poster-ready figures are not all in place yet. Track-A figures
need one 2D training run + the noise-sweep eval + the Jacobian phase. Track-B
figures come from the teammate (request the 6 result plots B4/B5 + the env render
+ obs dims).

## Open items to confirm

- [ ] **3D obs exact dims/units** (odor + pose + wind + detection) — for B2.
- [ ] **2D connectome result framing in A4a:** show the negative/unstable curve, or GRU-only? (band already states it.)
- [ ] Connectome param count vs GRU in **each** pipeline (supports the "capacity limit" claim).
- [ ] Author order / advisor / lab / funding.
- [ ] Get the 6 Track-B result figures + env render from teammate at final resolution.
