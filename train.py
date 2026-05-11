"""Unified training entry. Branches on --agent-type {ppo, rsac, drqn}.

PPO  : sb3 RecurrentPPO + 4-phase static->dynamic curriculum (16 SubprocVecEnv).
RSAC : single-env episode loop with hybrid actor.
DRQN : single-env episode loop with discrete action adapter.

Outputs land under runs/{agent}_{run_name}_{timestamp}/.
"""
import json
import os
import shutil
import time

import numpy as np
import torch

from src.utils.buffer import EpisodeReplayBuffer
from src.utils.config import build_parser, parse_phases
from src.utils.factory import make_agent, make_env
from src.utils.plotter import plot_training_pngs_from_data, save_training_plot_data
from src.utils.seed import set_global_seed


def _make_run_dir(args):
    if args.run_dir is not None:
        os.makedirs(args.run_dir, exist_ok=True)
        return args.run_dir
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    name = args.run_name or "main"
    run_dir = os.path.join(args.out_dir, f"{args.agent_type}_{name}_{timestamp}")
    for sub in ("checkpoints", "plots", "plot_data"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    return run_dir


def _dump_config(run_dir, args):
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)


def _device(args):
    if args.force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


# ---------------------------------------------------------------------------
# PPO trainer
# ---------------------------------------------------------------------------


def train_ppo(args, run_dir):
    from src.callbacks import MetricsCallback

    tb_log = args.tb_log or os.path.join(run_dir, "tb")
    os.makedirs(tb_log, exist_ok=True)
    args.tb_log = tb_log

    agent = make_agent(args, env=None, device=_device(args))

    phases = parse_phases(args)
    if not phases:
        raise ValueError("No phases specified for PPO training.")

    callback = MetricsCallback()
    prev_vecnorm = None
    for i, (env_kind, total_steps) in enumerate(phases):
        print(f"\n=== PPO Phase {i}: {env_kind} ({total_steps} steps) ===")
        agent.learn_phase(
            env_kind=env_kind,
            total_timesteps=total_steps,
            n_envs=args.n_envs,
            callback=callback,
            vecnorm_load_path=prev_vecnorm,
            reset_num_timesteps=(i == 0),
        )
        model_path = os.path.join(run_dir, "checkpoints", f"phase{i}_{env_kind}.zip")
        vnorm_path = os.path.join(run_dir, "checkpoints", f"phase{i}_{env_kind}_vnorm.pkl")
        agent.save(model_path, vecnorm_path=vnorm_path)
        prev_vecnorm = vnorm_path

    # Aliases for downstream eval / replot scripts.
    last_idx = len(phases) - 1
    last_kind = phases[-1][0]
    src_model = os.path.join(run_dir, "checkpoints", f"phase{last_idx}_{last_kind}.zip")
    src_vnorm = os.path.join(run_dir, "checkpoints", f"phase{last_idx}_{last_kind}_vnorm.pkl")
    if os.path.exists(src_model):
        shutil.copyfile(src_model, os.path.join(run_dir, "checkpoints", "final.zip"))
        shutil.copyfile(src_vnorm, os.path.join(run_dir, "checkpoints", "final_vnorm.pkl"))


# ---------------------------------------------------------------------------
# Episode-loop trainer for RSAC / DRQN
# ---------------------------------------------------------------------------


def _linear_eps(step, args):
    frac = min(1.0, step / max(1, args.eps_decay_steps))
    return args.eps_start + frac * (args.eps_end - args.eps_start)


def train_episode_loop(args, run_dir):
    env_kind = args.rsac_env_kind if args.agent_type == "rsac" else args.drqn_env_kind
    env = make_env(env_kind)
    device = _device(args)
    agent = make_agent(args, env, device)
    buffer = EpisodeReplayBuffer(cap_steps=args.buffer_size)

    best_return = -float("inf")
    ep_returns, ep_steps_to_goal = [], []

    for ep in range(1, args.total_episodes + 1):
        obs, _ = env.reset(seed=args.seed + ep)
        h = None
        traj = []
        ep_ret = 0.0
        steps_in_ep = 0
        success_step = None

        while True:
            if args.agent_type == "rsac":
                action, h = agent.get_action(obs, h)
                env_action = action
            else:
                eps = _linear_eps(ep, args)
                a_idx, h = agent.get_action(obs, h, epsilon=eps)
                action = a_idx
                env_action = agent.to_env_action(a_idx)

            next_obs, reward, terminated, truncated, info = env.step(env_action)
            done = bool(terminated or truncated)
            terminal = float(terminated)
            traj.append((obs, action, float(reward), next_obs, terminal))

            obs = next_obs
            ep_ret += float(reward)
            steps_in_ep += 1
            if info.get("is_success") and success_step is None:
                success_step = steps_in_ep
            if done:
                break

        buffer.add_episode(traj)
        ep_returns.append(ep_ret)
        ep_steps_to_goal.append(success_step if success_step is not None else steps_in_ep)

        if len(buffer) >= args.learning_starts:
            sampler = buffer.sample_continuous if args.agent_type == "rsac" else buffer.sample
            batch = sampler(args.batch_size, args.seq_len)
            agent.update(batch)
            if args.agent_type == "drqn" and ep % args.target_update_every == 0:
                agent.sync_target()

        if ep_ret > best_return:
            best_return = ep_ret
            agent.save(os.path.join(run_dir, "checkpoints", "best.pt"))
        if ep == 100:
            agent.save(os.path.join(run_dir, "checkpoints", "first.pt"))

        if ep % args.log_every == 0:
            recent = ep_returns[-args.log_every:]
            print(f"[ep {ep:5d}] return={ep_ret:7.2f}  recent_mean={np.mean(recent):7.2f}"
                  f"  steps={steps_in_ep}  buffer={len(buffer)}")

    agent.save(os.path.join(run_dir, "checkpoints", "final.pt"))
    save_training_plot_data(run_dir, ep_returns, ep_steps_to_goal)
    plot_training_pngs_from_data(run_dir)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.episodes is not None:
        args.eval_episodes = args.episodes

    set_global_seed(args.seed)
    run_dir = _make_run_dir(args)
    _dump_config(run_dir, args)
    print(f"[run_dir] {run_dir}")

    if args.agent_type == "ppo":
        train_ppo(args, run_dir)
    else:
        train_episode_loop(args, run_dir)

    print(f"\n[done] artifacts saved to {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
