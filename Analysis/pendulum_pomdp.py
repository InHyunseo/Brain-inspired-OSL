"""Velocity-masked Pendulum-v1.

Pendulum-v1's native observation is `[cos(theta), sin(theta), theta_dot]`.
We hide `theta_dot` so the policy must integrate it through hidden state — this
makes the task a POMDP where memory is required, which (a) gives nontrivial
hidden-state dynamics to analyze and (b) creates two clear behavior modes
(swing-up vs balance) suitable for Phase 2 segment-conditioned analysis.

The true `theta` and `theta_dot` are exposed via `info` for ground-truth
labeling in Phase 3a linear probing, but never reach the policy.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np


class VelocityMaskedPendulum(gym.Wrapper):
    """Pendulum-v1 wrapper masking angular velocity.

    Obs: [cos(theta), sin(theta)]  (2,)
    Info adds: angle (theta in [-pi, pi]), angvel (theta_dot).
    """

    def __init__(self, render_mode: str | None = None):
        env = gym.make("Pendulum-v1", render_mode=render_mode)
        super().__init__(env)
        low = np.array([-1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _mask(self, obs):
        return np.asarray(obs[:2], dtype=np.float32)

    def _label(self):
        # Pendulum-v1 internal state: env.unwrapped.state = [theta, theta_dot]
        theta, theta_dot = self.env.unwrapped.state
        theta_wrapped = float(((theta + np.pi) % (2 * np.pi)) - np.pi)
        return {"angle": theta_wrapped, "angvel": float(theta_dot)}

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        info.update(self._label())
        return self._mask(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info.update(self._label())
        return self._mask(obs), reward, terminated, truncated, info


def make_env(seed: int = 0, render_mode: str | None = None):
    def thunk():
        env = VelocityMaskedPendulum(render_mode=render_mode)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    return thunk
