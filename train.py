"""Unified training entry. Branches on --agent-type {ppo, rsac}.

PPO  : custom on-policy trainer with separate actor (larva connectome) /
       critic (MLP). Curriculum advances noise stages between phases by
       broadcasting set_noise_stage to the parallel env runner.
RSAC : single-env episode loop with selectable backbone (connectome / gru / mlp)
       for biological-fidelity baselines.

Outputs land under runs/{agent}_{run_name}_{timestamp}/.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from src.utils.buffer import EpisodeReplayBuffer
from src.utils.config import build_parser, parse_curriculum_phases
from src.utils.factory import make_env, make_ppo_trainer, make_rsac_agent
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
    phases = parse_curriculum_phases(args)
    if not phases:
        raise ValueError("No curriculum phases specified.")

    trainer = make_ppo_trainer(args, run_dir=run_dir)
    summary = {}
    try:
        for i, (stage, strength, steps) in enumerate(phases):
            print(f"\n=== PPO Phase {i}: noise_stage={stage} strength={strength} steps={steps} ===")
            trainer.runner.set_noise_stage(stage, strength)
            summary = trainer.train(phase_timesteps=steps)
        trainer.save_final(summary)
        print(f"\n[final summary]\n{json.dumps(summary, indent=2)}")
    finally:
        trainer.close()


# ---------------------------------------------------------------------------
# RSAC episode-loop trainer
# ---------------------------------------------------------------------------


def train_rsac(args, run_dir):
    env = make_env(
        args,
        seed=args.seed,
        noise_stage=args.rsac_noise_stage,
        noise_strength=args.rsac_noise_strength,
    )
    device = _device(args)
    agent = make_rsac_agent(args, env, device)
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
            action, h = agent.get_action(obs, h)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            terminal = float(terminated)
            traj.append((obs, action, float(reward), next_obs, terminal))

            obs = next_obs
            ep_ret += float(reward)
            steps_in_ep += 1
            if info.get("success") and success_step is None:
                success_step = steps_in_ep
            if done:
                break

        buffer.add_episode(traj)
        ep_returns.append(ep_ret)
        ep_steps_to_goal.append(success_step if success_step is not None else steps_in_ep)

        if len(buffer) >= args.learning_starts:
            batch = buffer.sample(args.batch_size, args.seq_len)
            agent.update(batch)

        if ep_ret > best_return:
            best_return = ep_ret
            agent.save(os.path.join(run_dir, "checkpoints", "best.pt"))
        if ep == 100:
            agent.save(os.path.join(run_dir, "checkpoints", "first.pt"))

        if ep % args.log_every == 0:
            recent = ep_returns[-args.log_every :]
            print(
                f"[ep {ep:5d}] return={ep_ret:7.2f}  recent_mean={np.mean(recent):7.2f}"
                f"  steps={steps_in_ep}  buffer={len(buffer)}"
            )

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
    elif args.agent_type == "rsac":
        train_rsac(args, run_dir)
    else:
        raise ValueError(f"Unsupported agent_type: {args.agent_type}")

    print(f"\n[done] artifacts saved to {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
