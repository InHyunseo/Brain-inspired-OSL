# notebooks — end-to-end Colab notebooks

Colab-friendly notebooks that train + evaluate + (optionally) analyze in one go.
Each follows the same pattern: a first cell clones the repo / installs deps /
`cd`s into the root, then a single hyperparameter block at the top of the
training cell drives everything. The `.py` entry points in the repo root are the
same code paths for headless runs.

- **`ppo_gru.ipynb`** — PPO with the **GRU** backbone (the working reference) +
  the `analysis.osl2d` mechanistic pipeline on the trained policy.
- **`ppo_connectome.ipynb`** — PPO with the **connectome** backbone + the same
  analysis pipeline.
- **`baseline.ipynb`** — the non-learning bilateral-chemotaxis baseline
  (`src/baselines/chemotaxis.py`) on the 2D OSL task.
