"""Unified eval entry. Branches on the agent_type recorded in run_dir/config.json.

PPO  : load model + VecNormalize, run elite-seed search, render notebook-style GIF.
RSAC/DRQN : load checkpoint, run deterministic rollouts, render GIF of the
            best-return episode using the shared plotter.
"""
import json
import os

import numpy as np
import torch

from src.utils.config import build_parser
from src.utils.factory import make_agent, make_env
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
    """Restore train-time settings from config.json, but keep CLI-provided eval
    overrides (--episodes, --noise-coef, --save-gif, --ckpt)."""
    keep = {"out_dir", "run_dir", "ckpt", "episodes", "eval_episodes",
            "save_gif", "force_cpu", "noise_coef", "seed_base", "agent_type"}
    for k, v in conf.items():
        if k in keep:
            continue
        if hasattr(args, k):
            setattr(args, k, v)
    args.agent_type = conf.get("agent_type", args.agent_type)
    return args


# ---------------------------------------------------------------------------
# PPO eval
# ---------------------------------------------------------------------------


def eval_ppo(args, run_dir):
    from src.agents.ppo_agent import PPOAgent
    model_path = args.ckpt or os.path.join(run_dir, "checkpoints", "final.zip")
    vnorm_path = os.path.join(run_dir, "checkpoints", "final_vnorm.pkl")
    env_kind = f"dynamic:{args.noise_coef}"

    agent = PPOAgent(features_dim=args.features_dim, seed=args.seed)
    agent.load(model_path, vecnorm_path=vnorm_path, env_kind=env_kind, n_envs=1)

    elites = _find_elite_seeds(agent, args)
    if not elites:
        print("[eval] no elite seeds found.")
        return

    if args.save_gif:
        best_seed = elites[0]
        gif_path = os.path.join(run_dir, "plots", "best_agent.gif")
        _render_ppo_gif(agent, best_seed, env_kind, gif_path)


def _find_elite_seeds(agent, args, n_to_find=3, min_casts=1, max_casts=300):
    elites = []
    vec = agent.vec_env
    print(f"[eval] searching elite seeds (success + casts in [{min_casts}, {max_casts}])")
    for trial in range(min(args.eval_episodes * 5, 500)):
        seed = args.seed_base + trial
        vec.seed(seed)
        obs = vec.reset()

        lstm_states, ep_starts = None, np.ones((1,), dtype=bool)
        for _ in range(300):
            action, lstm_states = agent.predict(
                obs, state=lstm_states, episode_start=ep_starts, deterministic=True
            )
            obs, _, done, infos = vec.step(action)
            ep_starts = done
            if done[0]:
                info = infos[0]
                casts = info.get("casts", 0)
                if info.get("is_success") and min_casts <= casts <= max_casts:
                    elites.append(seed)
                    print(f"[eval] seed {seed}: success (casts={casts})")
                break
        if len(elites) >= n_to_find:
            break
    return elites


def _render_ppo_gif(agent, seed, env_kind, gif_path):
    vec = agent.vec_env
    vec.seed(seed)
    obs = vec.reset()
    raw_env = vec.envs[0].unwrapped

    lstm_states, ep_starts = None, np.ones((1,), dtype=bool)
    frames, traj_x, traj_y, cx, cy = [], [], [], [], []
    print(f"[eval] rendering GIF for seed {seed}")
    for t in range(300):
        action, lstm_states = agent.predict(
            obs, state=lstm_states, episode_start=ep_starts, deterministic=True
        )
        curr_x, curr_y = raw_env.x, raw_env.y
        if int(np.rint(action[0, 2])) == 1 or raw_env.in_cast:
            cx.append(curr_x); cy.append(curr_y)
        traj_x.append(curr_x); traj_y.append(curr_y)

        frames.append(render_rollout_frame(
            raw_env, traj_x, traj_y, cx, cy, t,
            title=f"PPO seed={seed} step={t}",
        ))

        obs, _, done, _ = vec.step(action)
        ep_starts = done
        if done[0]:
            break

    save_gif(frames, gif_path, fps=15)


# ---------------------------------------------------------------------------
# RSAC / DRQN eval
# ---------------------------------------------------------------------------


def eval_episode_loop(args, run_dir):
    env_kind = args.rsac_env_kind if args.agent_type == "rsac" else args.drqn_env_kind
    env = make_env(env_kind)
    device = _device(args)
    agent = make_agent(args, env, device)

    ckpt = args.ckpt or os.path.join(run_dir, "checkpoints", "best.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(run_dir, "checkpoints", "final.pt")
    agent.load(ckpt)
    print(f"[eval] loaded {ckpt}")

    rets, succ = [], []
    best_seed, best_ret = None, -float("inf")

    for i in range(args.eval_episodes):
        seed = args.seed_base + i
        obs, _ = env.reset(seed=seed)
        h = None
        ep_ret, success = 0.0, False
        while True:
            if args.agent_type == "rsac":
                action, h = agent.get_action_deterministic(obs, h)
                env_action = action
            else:
                a_idx, h = agent.get_action_deterministic(obs, h)
                env_action = agent.to_env_action(a_idx)
            obs, r, terminated, truncated, info = env.step(env_action)
            ep_ret += float(r)
            if info.get("is_success"):
                success = True
            if terminated or truncated:
                break

        rets.append(ep_ret)
        succ.append(float(success))
        if ep_ret > best_ret:
            best_ret, best_seed = ep_ret, seed

    print(f"[eval] success_rate={np.mean(succ):.3f}  avg_return={np.mean(rets):.2f}")

    if args.save_gif and best_seed is not None:
        _render_episode_loop_gif(agent, env, best_seed, args, run_dir)


def _render_episode_loop_gif(agent, env, seed, args, run_dir):
    obs, _ = env.reset(seed=seed)
    h = None
    frames, traj_x, traj_y, cx, cy = [], [], [], [], []
    for t in range(300):
        if args.agent_type == "rsac":
            action, h = agent.get_action_deterministic(obs, h)
            env_action = action
            cast_taken = int(np.rint(action[2])) == 1
        else:
            a_idx, h = agent.get_action_deterministic(obs, h)
            env_action = agent.to_env_action(a_idx)
            cast_taken = (a_idx == 1)

        traj_x.append(env.x); traj_y.append(env.y)
        if cast_taken or getattr(env, "in_cast", False):
            cx.append(env.x); cy.append(env.y)

        frames.append(render_rollout_frame(
            env, traj_x, traj_y, cx, cy, t,
            title=f"{args.agent_type.upper()} seed={seed} step={t}",
        ))

        obs, _, terminated, truncated, _ = env.step(env_action)
        if terminated or truncated:
            break

    save_gif(frames, os.path.join(run_dir, "plots", "best_agent.gif"), fps=15)


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
    else:
        eval_episode_loop(args, args.run_dir)


if __name__ == "__main__":
    main()
