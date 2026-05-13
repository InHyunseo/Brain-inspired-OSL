"""Unified eval entry. Branches on the agent_type recorded in run_dir/config.json.

PPO : load Policy from ckpt_final.pt, run deterministic rollouts, render GIF.
SAC : load SACPolicy, run deterministic rollouts, render GIF of best episode.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from src.utils.config import build_parser
from src.utils.factory import make_env
from src.utils.plotter import render_rollout_frame, save_gif
from src.utils.seed import set_global_seed


def _device(args):
    if args.force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def _load_conf(run_dir):
    path = os.path.join(run_dir, "config.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing config.json in {run_dir}")
    with open(path, "r") as f:
        return json.load(f)


def _apply_conf_overrides(args, conf):
    """Restore train-time settings from config.json, but keep eval-time overrides."""
    keep = {
        "out_dir", "run_dir", "ckpt", "episodes", "eval_episodes",
        "save_gif", "force_cpu", "seed_base", "agent_type",
        "eval_noise_stage", "eval_noise_strength",
    }
    for k, v in conf.items():
        if k in keep:
            continue
        if hasattr(args, k):
            setattr(args, k, v)
    args.agent_type = conf.get("agent_type", args.agent_type)
    return args


def _build_eval_env(args):
    return make_env(
        args,
        seed=args.seed_base,
        noise_stage=args.eval_noise_stage,
        noise_strength=args.eval_noise_strength,
    )


# ---------------------------------------------------------------------------
# PPO eval
# ---------------------------------------------------------------------------


def eval_ppo(args, run_dir):
    from src.agents.ppo_agent import PPOConfig
    from src.models.policy import Policy

    ckpt_path = args.ckpt or os.path.join(run_dir, "ckpt_final.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(run_dir, "checkpoints", "ckpt_final.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No PPO checkpoint at {ckpt_path}.")
    print(f"[eval] loading {ckpt_path}")

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = PPOConfig(**payload["agent_config"])
    device = _device(args)
    policy = Policy(
        weights_csv=cfg.weights_csv,
        metadata_csv=cfg.metadata_csv,
        latent_dim=cfg.latent_dim,
        message_passing_steps=cfg.message_passing_steps,
        log_std_init=cfg.log_std_init,
    ).to(device)
    policy.load_state_dict(payload["policy_state_dict"])
    policy.eval()

    env = _build_eval_env(args)
    rets, succ = [], []
    best_seed, best_ret = None, -float("inf")

    for i in range(args.eval_episodes):
        seed = args.seed_base + i
        obs, _ = env.reset(seed=seed)
        actor_state, critic_state = policy.initial_states(1, device)
        mask = torch.zeros(1, 1, device=device)
        ep_ret, success = 0.0, False
        while True:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, next_actor_state, next_critic_state = policy.act(
                    obs_t, actor_state, critic_state, mask, deterministic=True
                )
            obs, r, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
            ep_ret += float(r)
            done = bool(terminated or truncated)
            if done:
                success = bool(info.get("success", False))
            mask.fill_(0.0 if done else 1.0)
            actor_state = next_actor_state * mask
            critic_state = next_critic_state * mask
            if done:
                break

        rets.append(ep_ret)
        succ.append(float(success))
        if ep_ret > best_ret:
            best_ret, best_seed = ep_ret, seed

    print(f"[eval] success_rate={np.mean(succ):.3f}  avg_return={np.mean(rets):.2f}  episodes={len(rets)}")

    if args.save_gif and best_seed is not None:
        gif_path = os.path.join(run_dir, "plots", "best_agent.gif")
        _render_ppo_gif(policy, device, args, best_seed, gif_path)


def _render_ppo_gif(policy, device, args, seed, gif_path):
    env = _build_eval_env(args)
    obs, _ = env.reset(seed=seed)
    actor_state, critic_state = policy.initial_states(1, device)
    mask = torch.zeros(1, 1, device=device)

    frames, traj_x, traj_y, cast_x, cast_y = [], [], [], [], []
    print(f"[eval] rendering GIF for seed {seed}")
    for t in range(env.max_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, _, _, next_actor_state, next_critic_state = policy.act(
                obs_t, actor_state, critic_state, mask, deterministic=True
            )
        traj_x.append(env.x_mm)
        traj_y.append(env.y_mm)

        obs, _, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
        if info.get("event_is_high_cast_like"):
            cast_x.append(env.x_mm)
            cast_y.append(env.y_mm)

        frames.append(render_rollout_frame(
            env, traj_x, traj_y, cast_x, cast_y, t,
            title=f"PPO seed={seed} step={t}",
        ))

        done = bool(terminated or truncated)
        mask.fill_(0.0 if done else 1.0)
        actor_state = next_actor_state * mask
        critic_state = next_critic_state * mask
        if done:
            break

    save_gif(frames, gif_path, fps=15)


# ---------------------------------------------------------------------------
# SAC eval
# ---------------------------------------------------------------------------


def eval_sac(args, run_dir):
    from src.agents.sac_agent import SACConfig, SACPolicy

    ckpt_path = args.ckpt or os.path.join(run_dir, "ckpt_final.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(run_dir, "checkpoints", "ckpt_final.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No SAC checkpoint at {ckpt_path}.")
    print(f"[eval] loading {ckpt_path}")

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = SACConfig(**payload["agent_config"])
    device = _device(args)
    policy = SACPolicy(
        weights_csv=cfg.weights_csv,
        metadata_csv=cfg.metadata_csv,
        latent_dim=cfg.latent_dim,
        message_passing_steps=cfg.message_passing_steps,
        critic_hidden=cfg.critic_hidden,
        log_std_init=cfg.log_std_init,
        log_std_min=cfg.log_std_min,
        log_std_max=cfg.log_std_max,
    ).to(device)
    policy.load_state_dict(payload["policy_state_dict"])
    policy.eval()

    env = _build_eval_env(args)
    rets, succ = [], []
    best_seed, best_ret = None, -float("inf")

    for i in range(args.eval_episodes):
        seed = args.seed_base + i
        obs, _ = env.reset(seed=seed)
        actor_state, critic_state = policy.initial_states(1, device)
        mask = torch.zeros(1, 1, device=device)
        ep_ret, success = 0.0, False
        while True:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, next_actor_state, next_critic_state = policy.act(
                    obs_t, actor_state, critic_state, mask, deterministic=True
                )
            obs, r, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
            ep_ret += float(r)
            done = bool(terminated or truncated)
            if done:
                success = bool(info.get("success", False))
            mask.fill_(0.0 if done else 1.0)
            actor_state = next_actor_state * mask
            critic_state = next_critic_state * mask
            if done:
                break

        rets.append(ep_ret)
        succ.append(float(success))
        if ep_ret > best_ret:
            best_ret, best_seed = ep_ret, seed

    print(f"[eval] success_rate={np.mean(succ):.3f}  avg_return={np.mean(rets):.2f}  episodes={len(rets)}")

    if args.save_gif and best_seed is not None:
        gif_path = os.path.join(run_dir, "plots", "best_agent.gif")
        _render_sac_gif(policy, device, args, best_seed, gif_path)


def _render_sac_gif(policy, device, args, seed, gif_path):
    env = _build_eval_env(args)
    obs, _ = env.reset(seed=seed)
    actor_state, critic_state = policy.initial_states(1, device)
    mask = torch.zeros(1, 1, device=device)

    frames, traj_x, traj_y, cast_x, cast_y = [], [], [], [], []
    print(f"[eval] rendering GIF for seed {seed}")
    for t in range(env.max_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, _, next_actor_state, next_critic_state = policy.act(
                obs_t, actor_state, critic_state, mask, deterministic=True
            )
        traj_x.append(env.x_mm)
        traj_y.append(env.y_mm)
        obs, _, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
        if info.get("event_is_high_cast_like"):
            cast_x.append(env.x_mm)
            cast_y.append(env.y_mm)
        frames.append(render_rollout_frame(
            env, traj_x, traj_y, cast_x, cast_y, t,
            title=f"SAC seed={seed} step={t}",
        ))
        done = bool(terminated or truncated)
        mask.fill_(0.0 if done else 1.0)
        actor_state = next_actor_state * mask
        critic_state = next_critic_state * mask
        if done:
            break

    save_gif(frames, gif_path, fps=15)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.run_dir is None:
        raise ValueError("--run-dir is required for evaluation.")
    if args.episodes is not None:
        args.eval_episodes = args.episodes

    conf = _load_conf(args.run_dir)
    args = _apply_conf_overrides(args, conf)
    set_global_seed(args.seed)

    if args.agent_type == "ppo":
        eval_ppo(args, args.run_dir)
    elif args.agent_type == "sac":
        eval_sac(args, args.run_dir)
    else:
        raise ValueError(f"Unsupported agent_type: {args.agent_type}")


if __name__ == "__main__":
    main()
