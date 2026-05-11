"""Legacy DRQN demo — single-sensor 2D OSL env + discrete {RUN, CAST, TURN_L, TURN_R}.

Self-contained (no dep on src/). Kept as a historical reference / baseline before the
biological refactor (bilateral sensor + larva connectome PPO). See `README.md`.
"""
from demo.DRQN.osl_env_2d import StaticEnv, DynamicEnv
from demo.DRQN.qnet import QNet
from demo.DRQN.drqn_agent import (
    DRQNAgent,
    A_RUN, A_CAST, A_TURN_L, A_TURN_R, N_ACTIONS,
)
from demo.DRQN.buffer import EpisodeReplayBuffer

__all__ = [
    "StaticEnv", "DynamicEnv",
    "QNet", "DRQNAgent",
    "A_RUN", "A_CAST", "A_TURN_L", "A_TURN_R", "N_ACTIONS",
    "EpisodeReplayBuffer",
]
