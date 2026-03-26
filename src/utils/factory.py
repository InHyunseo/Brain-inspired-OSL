import gymnasium as gym
from gymnasium.envs.registration import register

from src.agents.dqn_agent import DQNAgent
from src.agents.drqn_agent import DRQNAgent
from src.agents.rsac_agent import RSACAgent


def build_env_kwargs(args):
    return {
        "src_x": args.src_x,
        "src_y": args.src_y,
        "wind_x": args.wind_x,
        "sigma_c": args.sigma_c,
        "reward_mode": args.reward_mode,
        "bio_reward_scale": args.bio_reward_scale,
        "cast_penalty": args.cast_penalty,
        "turn_penalty": args.turn_penalty,
        "b_hold": args.b_hold,
        "goal_hold_steps": args.goal_hold_steps,
        "terminate_on_hold": args.terminate_on_hold,
    }


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

    elif args.agent_type == "rsac":
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

    else:
        raise ValueError(f"Unsupported agent_type: {args.agent_type}")

def make_agent_from_conf(conf, env, device, overrides=None):
    class Obj:
        pass

    args = Obj()
    args.agent_type = conf.get("agent_type", "drqn")
    args.dqn_hidden = conf.get("dqn_hidden", 256)
    args.rnn_hidden = conf.get("rnn_hidden", 147)
    args.lr = conf.get("lr", 1e-4)

    args.lr_actor = conf.get("lr_actor", 3e-4)
    args.lr_critic = conf.get("lr_critic", 3e-4)
    args.lr_alpha = conf.get("lr_alpha", 3e-4)
    args.gamma = conf.get("gamma", 0.99)
    args.tau = conf.get("tau", 0.005)
    args.rsac_actor_backbone = conf.get("rsac_actor_backbone", "gru")
    args.connectome_steps = conf.get("connectome_steps", 4)
    args.connectome_hidden = conf.get("connectome_hidden", 180)

    if overrides is not None:
        for key in vars(overrides):
            value = getattr(overrides, key)
            if value is not None:
                setattr(args, key, value)

    return make_agent(args, env, device)