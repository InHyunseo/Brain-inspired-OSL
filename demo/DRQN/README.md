# DRQN Demo (Legacy)

Discrete-action recurrent / feed-forward Q-learning baseline for the **legacy** single-sensor 2D OSL env (Gaussian / turbulent plume + 4-step cast macro). Kept here as a historical reference before the biological refactor (bilateral sensor + larva connectome PPO/RSAC under `src/`).

Why it lives in `demo/` instead of `src/`:
- The discrete action set `{RUN, CAST, TURN_L, TURN_R}` only makes sense with the old `Box([v, omega, cast])` env that has the 4-step cast lock. The new `OslEnv` (continuous `[v, body_omega, head_omega]`, no cast macro) breaks the discrete adapter.
- Self-contained — no dependency on the rest of the repo. Safe to delete or freeze.

## Layout
```
demo/DRQN/
├── osl_env_2d.py   # legacy StaticEnv / DynamicEnv (single sensor + cast)
├── qnet.py         # GRU (DRQN) / MLP (DQN) Q-net
├── drqn_agent.py   # discrete agent + to_env_action adapter
├── buffer.py       # episode replay buffer (int64 actions)
└── __init__.py
```

## Run from a script

```python
import torch, numpy as np
from collections import deque
from demo.DRQN import StaticEnv, DynamicEnv, DRQNAgent, EpisodeReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_env(kind):
    if kind == "static": return StaticEnv()
    return DynamicEnv(noise_coef=float(kind.split("_")[1]))

phases = [("static", 20_000), ("dynamic_0.3", 10_000), ("dynamic_0.6", 10_000), ("dynamic_1.0", 20_000)]

env = make_env(phases[0][0])
agent = DRQNAgent(
    obs_dim=env.observation_space.shape[0],
    action_low=env.action_space.low, action_high=env.action_space.high,
    device=device, rnn_hidden=147, lr=1e-4, gamma=0.99, recurrent=True,
)
buffer = EpisodeReplayBuffer(cap_steps=150_000)

global_ep, success_window = 0, deque(maxlen=100)
for phase_kind, phase_episodes in phases:
    env = make_env(phase_kind)
    for _ in range(phase_episodes):
        global_ep += 1
        obs, _ = env.reset(seed=42 + global_ep)
        h, traj, ep_ret, success = None, [], 0.0, False
        while True:
            eps = max(0.05, 1.0 - global_ep / 4000)
            a, h = agent.get_action(obs, h, epsilon=eps)
            next_obs, r, term, trunc, info = env.step(agent.to_env_action(a))
            traj.append((obs, a, float(r), next_obs, float(term)))
            obs = next_obs; ep_ret += r
            if info.get("is_success"): success = True
            if term or trunc: break
        buffer.add_episode(traj)
        success_window.append(float(success))
        if len(buffer) >= 5000:
            agent.update(buffer.sample(batch_size=128, seq_len=16))
            if global_ep % 20 == 0: agent.sync_target()
```

For a Colab-runnable end-to-end example see `ipynb/DRQN_framework.ipynb`.
