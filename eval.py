import os
import json
import argparse
import numpy as np
import torch

from src.envs.odor_env import OdorHoldEnv
from src.envs.odor_env_v4 import OdorHoldEnvV4
from src.agents.drqn_agent import DRQNAgent
from src.agents.dqn_agent import DQNAgent
from src.agents.rsac_agent import RSACAgent
from src.utils import plotter

def load_config(run_dir):
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found in {run_dir}")
    with open(config_path, "r") as f:
        return json.load(f)

def _rollout_trajectories(env, agent, agent_type, episodes, seed_base):
    trajectories = []
    success_count = 0
    for i in range(episodes):
        obs, _ = env.reset(seed=seed_base + i)
        h = None
        done = False
        xs, ys = [env.x], [env.y]
        ep_ret = 0.0
        in_goal = False

        while not done:
            if agent_type == "rsac":
                action, h = agent.get_action_deterministic(obs, h)
            else:
                action, h = agent.get_action(obs, h, epsilon=0.0)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            xs.append(env.x)
            ys.append(env.y)
            ep_ret += r
            if info.get("in_goal", 0):
                in_goal = True

        if in_goal:
            success_count += 1
        trajectories.append({"return": ep_ret, "success": in_goal, "x": xs, "y": ys})
    return trajectories, success_count

def evaluate(args):
    conf = load_config(args.run_dir)
    device = torch.device("cpu") if args.force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Evaluating on {device}")

    # Env Params Restore
    env_kwargs = {
        'L': conf.get('L', 3.0),
        'dt': conf.get('dt', 0.1),
        'v_fixed': conf.get('v_fixed', 0.25),
        'src_x': conf.get('src_x', 0.0),
        'src_y': conf.get('src_y', 0.0),
        'wind_x': conf.get('wind_x', 1.0),
        'wind_y': conf.get('wind_y', 0.0),
        'sigma_c': conf.get('sigma_c', 1.0),
        'r_goal': conf.get('r_goal', 0.35),
    }
    
    # 1. Trajectory Eval Env
    env_id = conf.get("env_id", "OdorHold-v3")
    env_cls = OdorHoldEnvV4 if str(env_id).endswith("-v4") else OdorHoldEnv
    env = env_cls(**env_kwargs)
    obs_dim = env.observation_space.shape[0]

    agent_type = conf.get("agent_type", "drqn")
    if agent_type == "dqn":
        act_dim = env.action_space.n
        agent = DQNAgent(
            obs_dim,
            act_dim,
            device,
            hidden=conf.get("dqn_hidden", 256),
            lr=conf.get("lr", 1e-4),
        )
    elif agent_type == "drqn":
        act_dim = env.action_space.n
        agent = DRQNAgent(
            obs_dim,
            act_dim,
            device,
            rnn_hidden=conf.get("rnn_hidden", 147),
            lr=conf.get("lr", 1e-4),
        )
    else:
        act_dim = env.action_space.shape[0]
        agent = RSACAgent(
            obs_dim,
            act_dim,
            env.action_space.low,
            env.action_space.high,
            device,
            rnn_hidden=conf.get("rnn_hidden", 147),
            lr_actor=conf.get("lr_actor", 3e-4),
            lr_critic=conf.get("lr_critic", 3e-4),
            lr_alpha=conf.get("lr_alpha", 3e-4),
            gamma=conf.get("gamma", 0.99),
            tau=conf.get("tau", 0.005),
            cell_type=conf.get("rnn_cell", "gru"),
        )

    ckpt_name = args.ckpt
    if ckpt_name is None:
        ckpt_name = "best.pt" if os.path.exists(os.path.join(args.run_dir, "checkpoints", "best.pt")) else "final.pt"
    
    ckpt_path = os.path.join(args.run_dir, "checkpoints", ckpt_name)
    print(f"[Info] Loading model from {ckpt_path}")
    
    try:
        agent.load(ckpt_path)
    except Exception as e:
        print(f"[Error] Failed to load checkpoint: {e}")
        return

    # --- Metrics Evaluation ---
    print(f"[Info] Starting evaluation over {args.episodes} episodes...")
    trajectories, success_count = _rollout_trajectories(env, agent, agent_type, args.episodes, args.seed_base)

    success_rate = success_count / args.episodes
    avg_return = np.mean([t['return'] for t in trajectories])
    
    print(f"  > Success Rate: {success_rate * 100:.1f}%")
    print(f"  > Avg Return:   {avg_return:.2f}")

    plots_dir = os.path.join(args.run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    # Remove legacy single-trajectory artifacts and eval trajectory JSON files.
    legacy_paths = [
        os.path.join(plots_dir, "trajectory.png"),
        os.path.join(args.run_dir, "plot_data", "eval_trajectories.json"),
        os.path.join(args.run_dir, "plot_data", "eval_trajectories_first.json"),
        os.path.join(args.run_dir, "plot_data", "eval_trajectories_mid.json"),
        os.path.join(args.run_dir, "plot_data", "eval_trajectories_best.json"),
    ]
    for p in legacy_paths:
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
    
    env.close()

    # --- GIF Generation (Here!) ---
    if args.save_gif:
        print(f"[Info] Generating GIF with model: {ckpt_name}")
        env_gif = env_cls(render_mode="rgb_array", **env_kwargs)
        
        frames = []
        # 성공하는 시드를 찾으면 좋겠지만, 일단 seed_base로 녹화
        obs, _ = env_gif.reset(seed=args.seed_base)
        h = None
        done = False
        
        # 첫 프레임: trajectory plot 스타일 + agent/casting 오버레이
        frames.append(plotter.render_rollout_frame_png_style(env_gif, title=f"Eval: {ckpt_name}"))

        while not done:
            if agent_type == "rsac":
                action, h = agent.get_action_deterministic(obs, h)
            else:
                action, h = agent.get_action(obs, h, epsilon=0.0)
            obs, _, term, trunc, _ = env_gif.step(action)
            done = term or trunc
            
            frames.append(plotter.render_rollout_frame_png_style(env_gif, title=f"Eval: {ckpt_name}"))
        
        env_gif.close()
        
        gif_path = os.path.join(plots_dir, "best_agent.gif")
        if frames:
            plotter.save_gif(frames, gif_path, fps=30)
        else:
            print("[Warn] GIF skipped: no render frames were produced.")

    if args.plot_milestones:
        milestone_path = os.path.join(args.run_dir, "plot_data", "milestones.json")
        if os.path.exists(milestone_path):
            with open(milestone_path, "r") as f:
                ms = json.load(f)
            first_ckpt = os.path.join(args.run_dir, "checkpoints", "first.pt")
            if not os.path.exists(first_ckpt):
                # backward compatibility with older runs
                first_ckpt = os.path.join(args.run_dir, "checkpoints", "ep100.pt")
            ckpt_map = [
                ("first", first_ckpt, int(ms.get("first_ep", 1))),
                ("mid", os.path.join(args.run_dir, "checkpoints", "mid.pt"), int(ms.get("mid_saved_ep", -1))),
                ("best", os.path.join(args.run_dir, "checkpoints", "best.pt"), int(ms.get("best_ep", -1))),
            ]
            for label, ckpt_i, ep_i in ckpt_map:
                if not os.path.exists(ckpt_i):
                    continue
                try:
                    agent.load(ckpt_i)
                except Exception as e:
                    print(f"[Warn] milestone load failed ({label}): {e}")
                    continue
                env_m = env_cls(**env_kwargs)
                traj_i, _ = _rollout_trajectories(env_m, agent, agent_type, args.episodes, args.seed_base)
                env_m.close()
                title_i = f"Eval {label} (ep={ep_i})"
                out_png = os.path.join(plots_dir, f"trajectory_{label}.png")
                plotter.plot_trajs_png(env_kwargs, out_png, traj_i, title_i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=20000)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument(
        "--save-gif",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save rollout GIF (default: True)",
    )
    parser.add_argument(
        "--plot-milestones",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render milestone trajectories (first/mid/best) when available",
    )

    args = parser.parse_args()
    evaluate(args)
