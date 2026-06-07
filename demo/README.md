# demo — legacy practice & sanity-check sandbox

Early RL practice and analysis-plumbing sanity checks. **Not part of the main
pipeline** — nothing in `src/` or `analysis/` imports from here. Kept as a
learning record.

- **`DRQN/`** — discrete DRQN/DQN on an old single-sensor OSL env (predecessor of
  the current `src/envs/osl_env.py`).
- **`PassiveSensing/`** — DQN on a simpler passive-sensing task variant.
- **`RL_practice/`** — CartPole actor-critic + the frame/probe/patch analysis
  plumbing, used to validate the analysis tooling before applying it to OSL.

Each subfolder has its own README.
