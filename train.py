import os
import time
import json
import argparse
import numpy as np
import torch
import gymnasium as gym
from gymnasium.envs.registration import register

from src.envs.odor_env import OdorHoldEnv
from src.utils.buffer import EpisodeReplayBuffer
from src.agents.drqn_agent import DRQNAgent
from src.agents.dqn_agent import DQNAgent
from src.utils import plotter

def make_env(env_id, **kwargs):
    if env_id not in gym.envs.registry:
        register(id=env_id, entry_point='src.envs.odor_env:OdorHoldEnv', kwargs=kwargs)
    return gym.make(env_id, **kwargs)

def ema(arr, alpha):
    out = []
    m = arr[0] if arr else 0
    for x in arr:
        m = alpha * x + (1 - alpha) * m
        out.append(m)
    return out

def train(args):
    # 1. Setup
    run_name = args.run_name or time.strftime(f"{args.agent_type}_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    
    # Save Config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # 2. Init
    env = make_env(
        args.env_id,
        src_x=args.src_x,
        src_y=args.src_y,
        wind_x=args.wind_x,
        sigma_c=args.sigma_c,
    )
    if args.agent_type == "dqn":
        agent = DQNAgent(
            env.observation_space.shape[0],
            env.action_space.n,
            device,
            hidden=args.dqn_hidden,
            lr=args.lr,
        )
    else:
        agent = DRQNAgent(
            env.observation_space.shape[0],
            env.action_space.n,
            device,
            rnn_hidden=args.rnn_hidden,
            lr=args.lr,
        )
    buffer = EpisodeReplayBuffer(args.buffer_size)

    # 3. Training Loop
    ep_returns = []
    ep_steps_to_goal = []
    best_return = -np.inf
    
    interrupted = False
    try:
        for ep in range(1, args.total_episodes + 1):
            obs, _ = env.reset(seed=args.seed + ep)
            h = None
            done = False
            ep_ret = 0
            traj = []
            step_count = 0
            first_goal_step = None
            
            # Epsilon Decay
            frac = min(1.0, (ep / args.eps_decay_steps))
            epsilon = args.eps_start - frac * (args.eps_start - args.eps_end)

            while not done:
                action, h_next = agent.get_action(obs, h, epsilon)
                next_obs, reward, term, trunc, info = env.step(action)
                done = bool(term or trunc)
                step_count += 1
                if first_goal_step is None and info.get("in_goal", 0):
                    first_goal_step = step_count
                
                traj.append((obs, action, float(reward), next_obs, float(done)))
                obs = next_obs
                h = h_next
                ep_ret += reward
                
                if len(buffer) > args.learning_starts and len(buffer) > args.batch_size * args.seq_len:
                    batch = buffer.sample(args.batch_size, args.seq_len)
                    agent.update(batch)

            buffer.add_episode(traj)
            ep_returns.append(ep_ret)
            if first_goal_step is None:
                ep_steps_to_goal.append(step_count)
            else:
                ep_steps_to_goal.append(first_goal_step)

            if ep_ret > best_return:
                best_return = ep_ret
                agent.save(os.path.join(run_dir, "checkpoints", "best.pt"))
            
            if ep % args.target_update_every == 0:
                agent.sync_target()

            if ep % args.log_every == 0:
                avg_ret = np.mean(ep_returns[-args.log_every:])
                avg_steps = np.mean(ep_steps_to_goal[-args.log_every:])
                print(f"Ep {ep} | Avg Ret: {avg_ret:.2f} | Avg Step-to-Goal: {avg_steps:.1f} | Eps: {epsilon:.3f}")
    except KeyboardInterrupt:
        interrupted = True
        print("\n[Warn] Training interrupted. Saving partial artifacts...")
    finally:
        agent.save(os.path.join(run_dir, "checkpoints", "final.pt"))
        if ep_returns:
            plotter.save_raw_ema_png(
                run_dir,
                "returns.png",
                range(1, len(ep_returns) + 1),
                ep_returns,
                ema(ep_returns, 0.05),
                "Return",
                (-5, 20),
            )
            max_steps_plot = max(ep_steps_to_goal) if ep_steps_to_goal else 1
            plotter.save_raw_ema_png(
                run_dir,
                "steps_to_goal.png",
                range(1, len(ep_steps_to_goal) + 1),
                ep_steps_to_goal,
                ema(ep_steps_to_goal, 0.05),
                "Step to Source",
                (0, max_steps_plot + 5),
            )
        env.close()
    if interrupted:
        raise KeyboardInterrupt

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="OdorHold-v3")
    p.add_argument("--agent-type", choices=["drqn", "dqn"], default="drqn")
    p.add_argument("--total-episodes", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--rnn-hidden", type=int, default=147)
    p.add_argument("--dqn-hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--src-x", type=float, default=0.0)
    p.add_argument("--src-y", type=float, default=0.0)
    p.add_argument("--wind-x", type=float, default=0.0)
    p.add_argument("--sigma-c", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    
    # Hyperparams
    p.add_argument("--buffer-size", type=int, default=150000)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--learning-starts", type=int, default=5000) # steps가 아니라 buffer size 체크용
    p.add_argument("--target-update-every", type=int, default=10) # episode 단위
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-steps", type=int, default=300) # episode 단위
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--force-cpu", action="store_true")

    args = p.parse_args()
    if args.run_name is None:
        args.run_name = time.strftime(f"{args.agent_type}_%Y%m%d_%H%M%S")
    train(args)
