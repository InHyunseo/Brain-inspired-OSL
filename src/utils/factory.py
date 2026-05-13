"""Env + agent factories shared by ppo / rsac entry points."""
from __future__ import annotations

from typing import Any

import torch

from src.envs.osl_env import EnvConfig, OslEnv


def make_env_config_dict(args, *, noise_stage: int = 0, noise_strength: float = 0.0) -> dict[str, Any]:
    """Build the env config dict from CLI args (used by PPO trainer + parallel runners)."""
    return {
        "sensor_spacing_mm": args.sensor_spacing_mm,
        "episode_seconds": args.episode_seconds,
        "arena_width_mm": args.arena_width_mm,
        "arena_height_mm": args.arena_height_mm,
        "source_x_mm": args.source_x_mm,
        "source_y_mm": args.source_y_mm,
        "gaussian_sigma_mm": args.gaussian_sigma_mm,
        "success_radius_mm": args.success_radius_mm,
        "noise_stage": noise_stage,
        "noise_strength": noise_strength,
        "seed": args.seed,
        # Reward shaping
        "reward_goal": args.reward_goal,
        "reward_log_k": args.reward_log_k,
        "reward_log_clip": args.reward_log_clip,
        "reward_conc_k": args.reward_conc_k,
        "reward_time_penalty": args.reward_time_penalty,
        "reward_run_cost": args.reward_run_cost,
        "reward_body_turn_cost": args.reward_body_turn_cost,
        "reward_head_cast_cost": args.reward_head_cast_cost,
        "reward_head_cast_stopped_mult": args.reward_head_cast_stopped_mult,
        "reward_spin_penalty": args.reward_spin_penalty,
        "wall_penalty": args.wall_penalty,
    }


def make_env(args, *, seed: int | None = None,
             noise_stage: int = 0, noise_strength: float = 0.0) -> OslEnv:
    """Construct a single OslEnv. Used for RSAC training and eval."""
    cfg_dict = make_env_config_dict(args, noise_stage=noise_stage, noise_strength=noise_strength)
    if seed is not None:
        cfg_dict["seed"] = seed
    return OslEnv(EnvConfig.from_dict(cfg_dict))


def make_ppo_trainer(args, run_dir):
    """Construct a PPOTrainer. Curriculum is driven externally via runner.set_noise_stage."""
    from src.agents.ppo_agent import PPOConfig, PPOTrainer

    env_cfg = make_env_config_dict(args, noise_stage=0, noise_strength=0.0)
    cfg = PPOConfig(
        rollout_steps=args.rollout_steps,
        num_envs=args.num_envs,
        parallel_envs=args.parallel_envs,
        update_epochs=args.update_epochs,
        minibatch_envs=args.minibatch_envs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_loss_coef=args.value_loss_coef,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        actor_max_grad_norm=args.actor_max_grad_norm,
        critic_max_grad_norm=args.critic_max_grad_norm,
        log_std_init=args.log_std_init,
        latent_dim=args.latent_dim,
        message_passing_steps=args.message_passing_steps,
        weights_csv=args.weights_csv,
        metadata_csv=args.metadata_csv,
        eval_interval_updates=args.eval_interval_updates,
        eval_episodes=args.eval_episodes_during_train,
        log_every_updates=args.log_every_updates,
        checkpoint_every_timesteps=args.checkpoint_every_timesteps,
        seed=args.seed,
        device="cpu" if args.force_cpu else "auto",
    )
    return PPOTrainer(env_cfg, cfg, run_dir=run_dir)


def make_rsac_agent(args, env, device: torch.device):
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
        connectome_latent_dim=args.latent_dim,
        connectome_message_passing_steps=args.message_passing_steps,
        connectome_weights_csv=args.weights_csv,
        connectome_metadata_csv=args.metadata_csv,
    )
