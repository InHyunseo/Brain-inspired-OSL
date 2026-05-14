"""Dump per-episode (obs, h, action, reward, angle, angvel) traces to npz.

Output:
    runs/{run_id}/seed_{s}/eval_ep{e:03d}.npz
    runs/{run_id}/seed_{s}/group_indices.json   (one per seed; identical content)

Phase 1-3 scripts consume these npz files only — they don't touch the policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from Analysis.ncp_core import NCPCore, NCPCoreConfig, action_to_env
from Analysis.pendulum_pomdp import VelocityMaskedPendulum


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", default="ncp_pendulum")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def load_policy(ckpt_path: Path, device) -> NCPCore:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = NCPCoreConfig(**blob["config"])
    policy = NCPCore(cfg).to(device)
    policy.load_state_dict(blob["state_dict"])
    policy.eval()
    return policy


def rollout(policy: NCPCore, env_seed: int, max_steps: int, device, deterministic: bool):
    env = VelocityMaskedPendulum()
    obs, info = env.reset(seed=env_seed)
    h = policy.initial_state(1, device=device)

    obs_buf, h_buf, act_buf, rew_buf, ang_buf, av_buf = [], [], [], [], [], []
    for t in range(max_steps):
        o = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        obs_buf.append(obs.copy())
        h_buf.append(h.squeeze(0).cpu().numpy())
        action, _, _, h_next = policy.act(o, h, deterministic=deterministic)
        env_a = action_to_env(action[0])
        next_obs, reward, term, trunc, info = env.step(env_a)
        act_buf.append(np.asarray(env_a, dtype=np.float32))
        rew_buf.append(float(reward))
        ang_buf.append(float(info["angle"]))
        av_buf.append(float(info["angvel"]))
        obs = next_obs
        h = h_next
        if term or trunc:
            break
    env.close()
    return {
        "obs": np.stack(obs_buf).astype(np.float32),
        "h": np.stack(h_buf).astype(np.float32),
        "action": np.stack(act_buf).astype(np.float32),
        "reward": np.asarray(rew_buf, dtype=np.float32),
        "angle": np.asarray(ang_buf, dtype=np.float32),
        "angvel": np.asarray(av_buf, dtype=np.float32),
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    base = Path(__file__).parent / "runs" / args.run_id

    for s in args.seeds:
        seed_dir = base / f"seed_{s}"
        ckpt = seed_dir / "policy.pt"
        if not ckpt.exists():
            print(f"[skip] no ckpt: {ckpt}")
            continue
        policy = load_policy(ckpt, device)

        # Save group metadata once per seed.
        gi = {k: v.tolist() for k, v in policy.group_indices.items()}
        gi["state_size"] = policy.state_size
        with (seed_dir / "group_indices.json").open("w") as f:
            json.dump(gi, f, indent=2)

        for e in range(args.episodes):
            ep_seed = 10_000 + s * 1000 + e
            traj = rollout(policy, ep_seed, args.max_steps, device, args.deterministic)
            out = seed_dir / f"eval_ep{e:03d}.npz"
            np.savez_compressed(out, **traj, episode=e, seed=s)
            ret = float(traj["reward"].sum())
            print(f"seed={s} ep={e:03d} T={len(traj['reward']):3d} return={ret:.1f}")


if __name__ == "__main__":
    main()
