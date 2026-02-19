import os
import time
import json
import shutil
import argparse
import numpy as np
import torch
import gymnasium as gym
from gymnasium.envs.registration import register

from src.utils.buffer import EpisodeReplayBuffer
from src.agents.drqn_agent import DRQNAgent
from src.agents.dqn_agent import DQNAgent
from src.agents.rsac_agent import RSACAgent
from src.utils import plotter

def make_env(env_id, **kwargs):
    if env_id not in gym.envs.registry:
        if str(env_id).endswith("-v4"):
            ep = 'src.envs.odor_env_v4:OdorHoldEnvV4'
        else:
            ep = 'src.envs.odor_env:OdorHoldEnv'
        register(id=env_id, entry_point=ep, kwargs=kwargs)
    return gym.make(env_id, **kwargs)

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
    elif args.agent_type == "drqn":
        agent = DRQNAgent(
            env.observation_space.shape[0],
            env.action_space.n,
            device,
            rnn_hidden=args.rnn_hidden,
            lr=args.lr,
        )
    else:
        agent = RSACAgent(
            env.observation_space.shape[0],
            env.action_space.shape[0],
            env.action_space.low,
            env.action_space.high,
            device,
            rnn_hidden=args.rnn_hidden,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            lr_alpha=args.lr_alpha,
            gamma=args.gamma,
            tau=args.tau,
            cell_type=args.rnn_cell,
        )
    buffer = EpisodeReplayBuffer(args.buffer_size)

    # 3. Training Loop
    ep_returns = []
    ep_steps_to_goal = []
    best_return = -np.inf
    best_ep = 0
    saved_eps = []
    
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
            log_stats = None

            while not done:
                if args.agent_type == "rsac":
                    action, h_next = agent.get_action(obs, h, epsilon=1.0)
                else:
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
                    if args.agent_type == "rsac":
                        batch = buffer.sample_continuous(args.batch_size, args.seq_len)
                    else:
                        batch = buffer.sample(args.batch_size, args.seq_len)
                    log_stats = agent.update(batch)

            buffer.add_episode(traj)
            ep_returns.append(ep_ret)
            if first_goal_step is None:
                ep_steps_to_goal.append(step_count)
            else:
                ep_steps_to_goal.append(first_goal_step)

            if ep_ret > best_return:
                best_return = ep_ret
                best_ep = ep
                agent.save(os.path.join(run_dir, "checkpoints", "best.pt"))

            if args.save_milestones and ep == 1:
                p1 = os.path.join(run_dir, "checkpoints", "ep_1.pt")
                agent.save(p1)
                if 1 not in saved_eps:
                    saved_eps.append(1)
                # Always keep first.pt; starts from ep1 and may be overwritten at first_milestone_ep.
                shutil.copy2(p1, os.path.join(run_dir, "checkpoints", "first.pt"))

            if args.save_milestones and ep == args.first_milestone_ep:
                p_first = os.path.join(run_dir, "checkpoints", f"ep_{ep}.pt")
                agent.save(p_first)
                if ep not in saved_eps:
                    saved_eps.append(ep)
                # Overwrite first.pt with milestone checkpoint (default ep100) when available.
                shutil.copy2(p_first, os.path.join(run_dir, "checkpoints", "first.pt"))

            if args.save_milestones and ep >= args.first_milestone_ep and ep % args.milestone_every == 0:
                pep = os.path.join(run_dir, "checkpoints", f"ep_{ep}.pt")
                agent.save(pep)
                if ep not in saved_eps:
                    saved_eps.append(ep)
            
            if args.agent_type != "rsac" and ep % args.target_update_every == 0:
                agent.sync_target()

            if ep % args.log_every == 0:
                avg_ret = np.mean(ep_returns[-args.log_every:])
                avg_steps = np.mean(ep_steps_to_goal[-args.log_every:])
                if args.agent_type == "rsac" and isinstance(log_stats, dict):
                    print(
                        f"Ep {ep} | Avg Ret: {avg_ret:.2f} | Avg Step-to-Goal: {avg_steps:.1f} "
                        f"| alpha: {log_stats['alpha']:.4f} | td|: {log_stats['td_abs']:.4f}"
                    )
                else:
                    print(f"Ep {ep} | Avg Ret: {avg_ret:.2f} | Avg Step-to-Goal: {avg_steps:.1f} | Eps: {epsilon:.3f}")
    except KeyboardInterrupt:
        interrupted = True
        print("\n[Warn] Training interrupted. Saving partial artifacts...")
    finally:
        agent.save(os.path.join(run_dir, "checkpoints", "final.pt"))
        if args.save_milestones and best_ep > 0:
            default_first_ep = int(args.first_milestone_ep)
            p_default_first = os.path.join(run_dir, "checkpoints", f"ep_{default_first_ep}.pt")
            if os.path.exists(p_default_first):
                first_ep = default_first_ep
            else:
                first_ep = 1
            lo = min(first_ep, int(best_ep))
            hi = max(first_ep, int(best_ep))
            mid_target = (lo + hi) // 2

            cands = [ep for ep in saved_eps if lo <= ep <= hi]
            if first_ep not in cands and os.path.exists(os.path.join(run_dir, "checkpoints", f"ep_{first_ep}.pt")):
                cands.append(first_ep)
            cands = sorted(set(cands))

            if cands:
                mid_ep = min(cands, key=lambda e: abs(e - mid_target))
                mid_src = os.path.join(run_dir, "checkpoints", f"ep_{mid_ep}.pt")
                if os.path.exists(mid_src):
                    shutil.copy2(mid_src, os.path.join(run_dir, "checkpoints", "mid.pt"))
            else:
                mid_ep = best_ep
                shutil.copy2(
                    os.path.join(run_dir, "checkpoints", "best.pt"),
                    os.path.join(run_dir, "checkpoints", "mid.pt"),
                )

            os.makedirs(os.path.join(run_dir, "plot_data"), exist_ok=True)
            milestone_meta = {
                "first_ep": int(first_ep),
                "best_ep": int(best_ep),
                "mid_target_ep": int(mid_target),
                "mid_saved_ep": int(mid_ep),
                "milestone_every": int(args.milestone_every),
            }
            with open(os.path.join(run_dir, "plot_data", "milestones.json"), "w") as f:
                json.dump(milestone_meta, f, indent=2)

        if ep_returns:
            plotter.save_training_plot_data(run_dir, ep_returns, ep_steps_to_goal, ema_alpha=0.05)
            plotter.plot_training_pngs_from_data(run_dir)
        env.close()
    if interrupted:
        raise KeyboardInterrupt

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="OdorHold-v3")
    p.add_argument("--agent-type", choices=["drqn", "dqn", "rsac"], default="drqn")
    p.add_argument("--total-episodes", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--rnn-hidden", type=int, default=147)
    p.add_argument("--dqn-hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-actor", type=float, default=3e-4)
    p.add_argument("--lr-critic", type=float, default=3e-4)
    p.add_argument("--lr-alpha", type=float, default=3e-4)
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--src-x", type=float, default=0.0)
    p.add_argument("--src-y", type=float, default=0.0)
    p.add_argument("--wind-x", type=float, default=0.0)
    p.add_argument("--sigma-c", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--rnn-cell", choices=["gru", "rnn"], default="gru")
    
    # Hyperparams
    p.add_argument("--buffer-size", type=int, default=150000)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--learning-starts", type=int, default=5000) # steps가 아니라 buffer size 체크용
    p.add_argument("--target-update-every", type=int, default=20) # episode 단위
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-steps", type=int, default=300) # episode 단위
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--save-milestones", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--first-milestone-ep", type=int, default=100)
    p.add_argument("--milestone-every", type=int, default=10)

    args = p.parse_args()
    if args.run_name is None:
        args.run_name = time.strftime(f"{args.agent_type}_%Y%m%d_%H%M%S")
    train(args)
