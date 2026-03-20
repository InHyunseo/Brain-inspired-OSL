import gymnasium as gym
from gymnasium.envs.registration import register

from src.agents.dqn_agent import DQNAgent
from src.agents.drqn_agent import DRQNAgent
from src.agents.rsac_agent import RSACAgent


def make_env(env_id, **kwargs):
    if env_id not in gym.envs.registry:
        if str(env_id).endswith("-v4"):
            ep = "src.envs.odor_env_v4:OdorHoldEnvV4"
        else:
            ep = "src.envs.odor_env_v3:OdorHoldEnv"
        register(id=env_id, entry_point=ep, kwargs=kwargs)
    return gym.make(env_id, **kwargs)


def make_agent(args, env, device):
    if args.agent_type == "dqn":
        return DQNAgent(
            env.observation_space.shape[0],
            env.action_space.n,
            device,
            hidden=args.dqn_hidden,
            lr=args.lr,
        )

    elif args.agent_type == "drqn":
        return DRQNAgent(
            env.observation_space.shape[0],
            env.action_space.n,
            device,
            rnn_hidden=args.rnn_hidden,
            lr=args.lr,
        )

    else:  # rsac
        return RSACAgent(
            env.observation_space.shape[0],
            env.action_space.shape[0],
            env.action_space.low,
            env.action_space.high,
            device,
            rnn_hidden=args.rnn_hidden,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            lr_alpha=args.lr_alpha,
            gamma=args.gamma,
            tau=args.tau,
            actor_backbone=getattr(args, "rsac_actor_backbone", "gru"),
            connectome_steps=getattr(args, "connectome_steps", 4),
            connectome_hidden=getattr(args, "connectome_hidden", 180),
        )