from src.agents.drqn_agent import DRQNAgent
from src.agents.rsac_agent import RSACAgent

__all__ = ["DRQNAgent", "RSACAgent", "PPOAgent"]


def __getattr__(name):
    # Lazy-import PPOAgent so envs/RSAC/DRQN keep working without sb3 installed.
    if name == "PPOAgent":
        from src.agents.ppo_agent import PPOAgent
        return PPOAgent
    raise AttributeError(name)
