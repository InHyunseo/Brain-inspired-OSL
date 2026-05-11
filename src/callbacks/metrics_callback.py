"""Custom TensorBoard metrics callback for sb3 RecurrentPPO.

Ported from ipynb/PPO_framework.ipynb `CustomMetricsCallback`. Tracks
success_rate, cast_ratio, ep_rew_mean, ep_len_mean over a rolling window
of the last 100 episodes.
"""
from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class MetricsCallback(BaseCallback):
    def __init__(self, window=100, verbose=0):
        super().__init__(verbose)
        self.success_buffer = deque(maxlen=window)
        self.cast_ratio_buffer = deque(maxlen=window)
        self.ep_rew_buffer = deque(maxlen=window)
        self.ep_len_buffer = deque(maxlen=window)

    def _on_step(self) -> bool:
        for idx, done in enumerate(self.locals["dones"]):
            if not done:
                continue
            info = self.locals["infos"][idx]
            if "final_info" in info:
                info = info["final_info"]

            self.success_buffer.append(float(info.get("is_success", False)))
            self.cast_ratio_buffer.append(float(info.get("cast_ratio", 0.0)))

            if "episode" in info:
                self.ep_rew_buffer.append(float(info["episode"]["r"]))
                self.ep_len_buffer.append(float(info["episode"]["l"]))
        return True

    def _on_rollout_end(self) -> None:
        if self.success_buffer:
            self.logger.record("rollout/custom_success_rate", float(np.mean(self.success_buffer)))
            self.logger.record("rollout/custom_cast_ratio", float(np.mean(self.cast_ratio_buffer)))
        if self.ep_rew_buffer:
            self.logger.record("rollout/custom_ep_rew_mean", float(np.mean(self.ep_rew_buffer)))
            self.logger.record("rollout/custom_ep_len_mean", float(np.mean(self.ep_len_buffer)))
