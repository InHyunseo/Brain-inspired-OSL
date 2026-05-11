"""Env + agent factories shared by ppo / rsac / drqn."""
from src.envs.osl_env_2d import StaticEnv, DynamicEnv


# env_kind strings used everywhere (CLI, phase JSON, factory):
#   "static"        -> StaticEnv
#   "dynamic_0.3"   -> DynamicEnv(noise_coef=0.3)
#   "dynamic_0.6"   -> DynamicEnv(noise_coef=0.6)
#   "dynamic_1.0"   -> DynamicEnv(noise_coef=1.0)
#   "dynamic:<c>"   -> DynamicEnv(noise_coef=float(<c>))     (arbitrary coef)
_FIXED_DYNAMIC = {"dynamic_0.3": 0.3, "dynamic_0.6": 0.6, "dynamic_1.0": 1.0}


def parse_env_kind(env_kind):
    """Return (env_cls, kwargs) for the given env_kind string."""
    if env_kind == "static":
        return StaticEnv, {}
    if env_kind in _FIXED_DYNAMIC:
        return DynamicEnv, {"noise_coef": _FIXED_DYNAMIC[env_kind]}
    if env_kind.startswith("dynamic:"):
        return DynamicEnv, {"noise_coef": float(env_kind.split(":", 1)[1])}
    raise ValueError(f"Unknown env_kind: {env_kind}")


def make_env(env_kind):
    cls, kwargs = parse_env_kind(env_kind)
    return cls(**kwargs)


def make_env_fn(env_kind, monitor=False):
    """Return a thunk that creates a fresh env. Wraps in sb3 Monitor if requested."""
    cls, kwargs = parse_env_kind(env_kind)

    if monitor:
        from stable_baselines3.common.monitor import Monitor

        def _fn():
            return Monitor(cls(**kwargs))
    else:
        def _fn():
            return cls(**kwargs)
    return _fn


def make_agent(args, env, device):
    """Build an agent from parsed CLI args.

    For PPO, `env` is unused (PPOAgent constructs its own VecEnv); pass None.
    """
    if args.agent_type == "ppo":
        from src.agents.ppo_agent import PPOAgent
        return PPOAgent(
            features_dim=args.features_dim,
            lr=args.ppo_lr,
            batch_size=args.ppo_batch_size,
            n_steps=args.ppo_n_steps,
            ent_coef=args.ppo_ent_coef,
            tb_log_dir=getattr(args, "tb_log", None),
            seed=args.seed,
        )

    if args.agent_type == "rsac":
        from src.agents.rsac_agent import RSACAgent
        return RSACAgent(
            obs_dim=env.observation_space.shape[0],
            act_dim=env.action_space.shape[0],
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            device=device,
            rnn_hidden=args.rnn_hidden,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            lr_alpha=args.lr_alpha,
            gamma=args.gamma,
            tau=args.tau,
            actor_backbone=args.rsac_actor_backbone,
            connectome_steps=args.connectome_steps,
            connectome_hidden=args.connectome_hidden,
        )

    if args.agent_type == "drqn":
        from src.agents.drqn_agent import DRQNAgent
        return DRQNAgent(
            obs_dim=env.observation_space.shape[0],
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            device=device,
            rnn_hidden=args.rnn_hidden,
            lr=args.lr,
            gamma=args.gamma,
            recurrent=args.drqn_recurrent,
        )

    raise ValueError(f"Unsupported agent_type: {args.agent_type}")
