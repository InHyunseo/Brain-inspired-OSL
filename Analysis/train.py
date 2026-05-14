"""Train RecurrentCfCPolicy on POMDP Pendulum via sb3-contrib RecurrentPPO.

After training, extract weights into a standalone `NCPCore` state_dict and
save as `policy.pt` so the phase scripts (which depend only on `NCPCore`)
work unchanged. Also dump `group_indices.json` for fast access.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from sb3_contrib import RecurrentPPO

from Analysis.ncp_core import NCPCore, NCPCoreConfig
from Analysis.pendulum_pomdp import make_env
from Analysis.recurrent_cfc_policy import RecurrentCfCPolicy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", default="ncp_pendulum")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--n_steps", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--n_epochs", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent_coef", type=float, default=0.0)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--units", type=int, default=32)
    p.add_argument("--output_size", type=int, default=8)
    p.add_argument("--sparsity", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(__file__).parent / "runs" / args.run_id / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env_fns = [make_env(seed=args.seed * 100 + i) for i in range(args.num_envs)]
    env = DummyVecEnv(env_fns)

    policy_kwargs = dict(
        units=args.units,
        output_size=args.output_size,
        sparsity_level=args.sparsity,
        wiring_seed=12345,
    )

    model = RecurrentPPO(
        policy=RecurrentCfCPolicy,
        env=env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        policy_kwargs=policy_kwargs,
        verbose=args.verbose,
        device=args.device,
        seed=args.seed,
        tensorboard_log=str(out_dir / "tb"),
    )

    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    # Save full sb3 model for completeness / resume.
    model.save(str(out_dir / "sb3_model.zip"))

    # Extract NCPCore-compatible state and save.
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    cfg = NCPCoreConfig(**model.policy.export_ncp_core_config(obs_dim, action_dim))
    core_sd = model.policy.export_ncp_core_state()
    torch.save({"state_dict": core_sd, "config": asdict(cfg)}, out_dir / "policy.pt")

    # Smoke-check that the extracted core actually loads cleanly.
    core = NCPCore(cfg)
    missing, unexpected = core.load_state_dict(core_sd, strict=False)
    if missing:
        print(f"[warn] missing keys when loading NCPCore: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected}")

    # Save group indices for fast phase-script access.
    gi = {k: v.tolist() for k, v in core.group_indices.items()}
    gi["state_size"] = core.state_size
    with (out_dir / "group_indices.json").open("w") as f:
        json.dump(gi, f, indent=2)

    print(f"saved → {out_dir / 'policy.pt'}")


if __name__ == "__main__":
    main()
