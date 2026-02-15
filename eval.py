import os
import json
import argparse
import numpy as np
import torch

from src.envs.odor_env import OdorHoldEnv
from src.agents.drqn_agent import DRQNAgent
from src.utils import plotter

def load_config(run_dir):
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found in {run_dir}")
    with open(config_path, "r") as f:
        return json.load(f)

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
    env = OdorHoldEnv(**env_kwargs)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = DRQNAgent(obs_dim, act_dim, device, rnn_hidden=conf.get('rnn_hidden', 147))

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
    trajectories = []
    success_count = 0

    print(f"[Info] Starting evaluation over {args.episodes} episodes...")
    for i in range(args.episodes):
        obs, _ = env.reset(seed=args.seed_base + i)
        h = None
        done = False
        xs, ys = [env.x], [env.y]
        ep_ret = 0.0
        in_goal = False
        
        while not done:
            action, h = agent.get_action(obs, h, epsilon=0.0)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            xs.append(env.x); ys.append(env.y)
            ep_ret += r
            if info.get('in_goal', 0): in_goal = True

        if in_goal: success_count += 1
        trajectories.append({"return": ep_ret, "success": in_goal, "x": xs, "y": ys})

    success_rate = success_count / args.episodes
    avg_return = np.mean([t['return'] for t in trajectories])
    
    print(f"  > Success Rate: {success_rate * 100:.1f}%")
    print(f"  > Avg Return:   {avg_return:.2f}")

    # Plot
    plot_config = env_kwargs.copy()
    plot_config['sigma_c'] = conf.get('sigma_c', 1.0)
    plotter.plot_trajs_png(plot_config, os.path.join(args.run_dir, "eval_result.png"), trajectories, f"Eval: {ckpt_name}")
    
    env.close()

    # --- GIF Generation (Here!) ---
    if args.save_gif:
        print(f"[Info] Generating GIF with best model...")
        # 중요: render_mode="rgb_array"로 환경 다시 생성
        env_gif = OdorHoldEnv(render_mode="rgb_array", **env_kwargs)
        
        frames = []
        # 성공하는 시드를 찾으면 좋겠지만, 일단 seed_base로 녹화
        obs, _ = env_gif.reset(seed=args.seed_base)
        h = None
        done = False
        
        # 첫 프레임
        frame = env_gif.render()
        if frame is not None: frames.append(frame)

        while not done:
            action, h = agent.get_action(obs, h, epsilon=0.0)
            obs, _, term, trunc, _ = env_gif.step(action)
            done = term or trunc
            
            frame = env_gif.render()
            if frame is not None: frames.append(frame)
        
        env_gif.close()
        
        gif_path = os.path.join(args.run_dir, "best_agent.gif")
        plotter.save_gif(frames, gif_path, fps=30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=20000)
    parser.add_argument("--force-cpu", action="store_true")
    # GIF 옵션 추가 (기본 True)
    parser.add_argument("--save-gif", action="store_true", default=True, help="Save GIF animation")

    args = parser.parse_args()
    evaluate(args)