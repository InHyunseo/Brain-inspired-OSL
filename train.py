"""Unified training entry. Branches on --agent-type {ppo, sac}.

Both agents share the same parallel env runner, curriculum loop, connectome
backbone, logging, checkpointing, and eval. The only difference is the
optimisation algorithm: PPO (on-policy, clipped surrogate) vs SAC (off-policy,
twin Q + auto-α).

Outputs land under runs/{agent}_{run_name}_{timestamp}/.
"""
from __future__ import annotations

import json
import os
import time

import torch

from src.utils.config import build_parser, parse_curriculum_phases
from src.utils.factory import make_ppo_trainer, make_sac_trainer
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


def _train_curriculum(trainer, phases, agent_label: str) -> None:
    """Drive curriculum via runner.set_noise_stage. Shared by PPO and SAC."""
    summary = {}
    try:
        for i, (stage, strength, steps) in enumerate(phases):
            print(f"\n=== {agent_label} Phase {i}: noise_stage={stage} strength={strength} steps={steps} ===")
            trainer.runner.set_noise_stage(stage, strength)
            summary = trainer.train(phase_timesteps=steps)
        trainer.save_final(summary)
        print(f"\n[final summary]\n{json.dumps(summary, indent=2)}")
    finally:
        trainer.close()


def train_ppo(args, run_dir):
    phases = parse_curriculum_phases(args)
    if not phases:
        raise ValueError("No curriculum phases specified.")
    trainer = make_ppo_trainer(args, run_dir=run_dir)
    _train_curriculum(trainer, phases, agent_label="PPO")


def train_sac(args, run_dir):
    phases = parse_curriculum_phases(args)
    if not phases:
        raise ValueError("No curriculum phases specified.")
    trainer = make_sac_trainer(args, run_dir=run_dir)
    _train_curriculum(trainer, phases, agent_label="SAC")


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
    elif args.agent_type == "sac":
        train_sac(args, run_dir)
    else:
        raise ValueError(f"Unsupported agent_type: {args.agent_type}")

    print(f"\n[done] artifacts saved to {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
