"""Custom PPO loop for NCP × POMDP Pendulum.

Minimal, recurrent-state-aware PPO. Vectorized envs run synchronously; each env
keeps its own hidden state across episode boundaries (reset to zero on done).
Rollout is unrolled with the same hidden state used at collection time so
gradients flow correctly through `CfC.forward(seq, h0)`.

Saves:
    runs/{run_id}/seed_{s}/policy.pt
    runs/{run_id}/seed_{s}/train_log.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from Analysis.ncp_policy import NCPPolicy, NCPPolicyConfig, action_to_env
from Analysis.pendulum_pomdp import make_env


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", default="ncp_pendulum")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--rollout", type=int, default=128)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=4)   # over the env axis
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent_coef", type=float, default=0.0)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--units", type=int, default=32)
    p.add_argument("--output_size", type=int, default=8)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """rewards/values/dones: (T, N). last_value: (N,). Returns adv, ret (T,N)."""
    T, N = rewards.shape
    adv = torch.zeros_like(rewards)
    gae = torch.zeros(N, device=rewards.device)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        next_nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterm - values[t]
        gae = delta + gamma * lam * next_nonterm * gae
        adv[t] = gae
    ret = adv + values
    return adv, ret


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(__file__).parent / "runs" / args.run_id / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    ckpt_path = out_dir / "policy.pt"

    envs = gym.vector.SyncVectorEnv(
        [make_env(seed=args.seed * 100 + i) for i in range(args.num_envs)]
    )
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]

    policy = NCPPolicy(NCPPolicyConfig(
        obs_dim=obs_dim, action_dim=act_dim,
        units=args.units, output_size=args.output_size,
    )).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    obs_buf = torch.zeros(args.rollout, args.num_envs, obs_dim, device=device)
    act_buf = torch.zeros(args.rollout, args.num_envs, act_dim, device=device)
    logp_buf = torch.zeros(args.rollout, args.num_envs, device=device)
    rew_buf = torch.zeros(args.rollout, args.num_envs, device=device)
    val_buf = torch.zeros(args.rollout, args.num_envs, device=device)
    done_buf = torch.zeros(args.rollout, args.num_envs, device=device)

    obs_np, _ = envs.reset(seed=args.seed)
    obs = torch.from_numpy(obs_np).float().to(device)
    h = policy.initial_state(args.num_envs, device=device)

    n_updates = args.timesteps // (args.rollout * args.num_envs)
    ep_returns = []           # per-episode totals
    cur_ret = np.zeros(args.num_envs)
    cur_len = np.zeros(args.num_envs, dtype=np.int64)
    global_step = 0

    log_path.unlink(missing_ok=True)
    t_start = time.time()

    for upd in range(n_updates):
        # ---- rollout ----
        h_start = h.detach().clone()    # hidden state at the start of this rollout
        for t in range(args.rollout):
            with torch.no_grad():
                action_raw, log_prob, value, h = policy.act(obs, h)
            env_action = action_to_env(action_raw)
            next_obs, reward, term, trunc, _ = envs.step(env_action)
            done = np.logical_or(term, trunc)
            obs_buf[t] = obs
            act_buf[t] = action_raw
            logp_buf[t] = log_prob
            val_buf[t] = value
            rew_buf[t] = torch.from_numpy(reward).float().to(device)
            done_buf[t] = torch.from_numpy(done.astype(np.float32)).to(device)
            cur_ret += reward
            cur_len += 1
            # Reset hidden state for envs that terminated.
            if done.any():
                done_t = torch.from_numpy(done.astype(np.float32)).to(device).unsqueeze(-1)
                h = h * (1.0 - done_t)
                for i in np.where(done)[0]:
                    ep_returns.append(float(cur_ret[i]))
                    cur_ret[i] = 0.0
                    cur_len[i] = 0
            obs = torch.from_numpy(next_obs).float().to(device)
            global_step += args.num_envs

        # ---- advantage ----
        with torch.no_grad():
            _, _, last_value, _ = policy.act(obs, h)
        adv, ret = compute_gae(rew_buf, val_buf, done_buf, last_value,
                               args.gamma, args.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Reshape to (N, T, ...) for sequence-based loss; we update per-env-sequence
        # so hidden-state-aware BPTT works.
        obs_seq = obs_buf.permute(1, 0, 2).contiguous()        # (N, T, F)
        act_seq = act_buf.permute(1, 0, 2).contiguous()        # (N, T, A)
        old_logp = logp_buf.permute(1, 0).contiguous()         # (N, T)
        adv_seq = adv.permute(1, 0).contiguous()
        ret_seq = ret.permute(1, 0).contiguous()
        N = args.num_envs
        mb_size = max(1, N // args.minibatches)
        idx_all = np.arange(N)

        # ---- optimize ----
        for _ in range(args.epochs):
            np.random.shuffle(idx_all)
            for mb in range(0, N, mb_size):
                mb_idx = idx_all[mb:mb + mb_size]
                mb_idx_t = torch.from_numpy(mb_idx).to(device)
                h0 = h_start.index_select(0, mb_idx_t)
                new_logp, new_val, ent = policy.evaluate_actions(
                    obs_seq.index_select(0, mb_idx_t),
                    h0,
                    act_seq.index_select(0, mb_idx_t),
                )
                ratio = (new_logp - old_logp.index_select(0, mb_idx_t)).exp()
                surr1 = ratio * adv_seq.index_select(0, mb_idx_t)
                surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * \
                    adv_seq.index_select(0, mb_idx_t)
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (new_val - ret_seq.index_select(0, mb_idx_t)).pow(2).mean()
                entropy = ent.mean()
                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

        # ---- log ----
        recent = ep_returns[-50:] if ep_returns else [float("nan")]
        log = {
            "update": upd,
            "global_step": global_step,
            "elapsed_s": round(time.time() - t_start, 1),
            "mean_return_50": float(np.mean(recent)),
            "n_episodes": len(ep_returns),
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy.detach()),
        }
        with log_path.open("a") as f:
            f.write(json.dumps(log) + "\n")
        if upd % 10 == 0 or upd == n_updates - 1:
            print(json.dumps(log))

    torch.save({
        "state_dict": policy.state_dict(),
        "config": vars(policy.cfg),
    }, ckpt_path)
    print(f"saved → {ckpt_path}")


if __name__ == "__main__":
    main()
