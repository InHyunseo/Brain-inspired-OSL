# analysis — Mechanistic analysis of trained OSL policies

Explains a trained policy's OSL behavior at the **neural-circuit level** —
labeling → linear probing → Jacobian eigenmodes → causal ablation — all computed
on the actor's hidden-state trace. Backbone-agnostic (GRU or connectome).

- **`methodology.md`** — the 3-phase methodology (objective, scope, falsifiable
  success criteria).
- **`METHODS_MATH.md`** — per-phase equations.
- **`osl2d/`** — the implementation.

## Pipeline (`osl2d/`)

`run_all.py` orchestrates the whole chain from a finished run's checkpoint:

```
collect → P1 (label) → P2a (latent viz) → P2b (probe) → P2c (neuron) → P3a (Jacobian) → P3b (fixed-point) → P4 (ablation)
```

```bash
python -m analysis.osl2d.run_all --run-dir runs/ppo_main_YYYYMMDD_HHMMSS \
  --checkpoints ckpt_final --stochastic
```

Every phase consumes the single trace dumped by `eval_dump.py`
(`(obs, hidden, action, event-flags, success)` per step) and writes JSON +
figures under the run's `analysis/` dir. All rollouts are **stochastic** (the
policy scores ~0% deterministic, so sampling is required — see the project README).

Key modules: `policy_adapter.py` (unified `forward`/`step` over the trained
`Policy`), `segment.py` (RUN / ACTIVE_SENSING / OTHER labels from event flags),
`probe.py`, `jacobian.py`, `_io.py`.

## Top-level scripts

Targeted analyses of the trained **GRU** policy across a noise sweep (used for the
key findings); they import `analysis.osl2d`:

- `ppo_gru_noise_sweep_eval.py` — success / steps / cast vs. noise.
- `noise_sweep_cast.py` — cast-neuron stability + causal ablation across noise.
- `ppo_gru_active_sensing_topk_pca.py` — top-k active-sensing neurons + PCA.
