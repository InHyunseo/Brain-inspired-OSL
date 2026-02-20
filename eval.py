import os
import json
import argparse
import numpy as np
import torch

from src.envs.odor_env_v3 import OdorHoldEnv
from src.envs.odor_env_v4 import OdorHoldEnvV4
from src.agents.drqn_agent import DRQNAgent
from src.agents.dqn_agent import DQNAgent
from src.agents.rsac_agent import RSACAgent
from src.utils import plotter
from src.utils.seed import set_global_seed

def load_config(run_dir):
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found in {run_dir}")
    with open(config_path, "r") as f:
        return json.load(f)

def _rollout_trajectories(env, agent, agent_type, episodes, seed_base):
    trajectories = []
    success_entry_count = 0
    success_hold_count = 0
    final_in_goal_count = 0
    cast_start_counts = []
    cast_step_counts = []
    cast_step_ratios = []
    can_turn_ratios = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed_base + i)
        h = None
        done = False
        xs, ys = [env.x], [env.y]
        ep_ret = 0.0
        in_goal = False
        hold_success = False
        final_in_goal = False
        cast_start_count = 0
        cast_step_count = 0
        cast_steps = 0
        can_turn_steps = 0
        total_steps = 0

        while not done:
            if agent_type == "rsac":
                action, h = agent.get_action_deterministic(obs, h)
            else:
                action, h = agent.get_action(obs, h, epsilon=0.0)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            total_steps += 1
            cast_start_count += int(info.get("cast_start", 0))
            cast_step_count += int(info.get("did_cast", 0))
            cast_steps += int(info.get("in_cast", 0))
            can_turn_steps += int(info.get("can_turn", 0))
            xs.append(env.x)
            ys.append(env.y)
            ep_ret += r
            if info.get("in_goal", 0):
                in_goal = True
            if info.get("success_hold", 0):
                hold_success = True
            final_in_goal = bool(info.get("in_goal", 0))

        if in_goal:
            success_entry_count += 1
        if hold_success:
            success_hold_count += 1
        if final_in_goal:
            final_in_goal_count += 1
        cast_start_counts.append(float(cast_start_count))
        cast_step_counts.append(float(cast_step_count))
        if total_steps > 0:
            cast_step_ratios.append(float(cast_steps) / float(total_steps))
            can_turn_ratios.append(float(can_turn_steps) / float(total_steps))
        else:
            cast_step_ratios.append(0.0)
            can_turn_ratios.append(0.0)
        trajectories.append({"return": ep_ret, "success": in_goal, "x": xs, "y": ys})
    stats = {
        "success_entry_rate": float(success_entry_count / episodes) if episodes > 0 else 0.0,
        "success_hold_rate": float(success_hold_count / episodes) if episodes > 0 else 0.0,
        "final_in_goal_rate": float(final_in_goal_count / episodes) if episodes > 0 else 0.0,
        "cast_start_count_mean": float(np.mean(cast_start_counts)) if cast_start_counts else 0.0,
        "cast_step_count_mean": float(np.mean(cast_step_counts)) if cast_step_counts else 0.0,
        "cast_step_ratio_mean": float(np.mean(cast_step_ratios)) if cast_step_ratios else 0.0,
        "can_turn_ratio_mean": float(np.mean(can_turn_ratios)) if can_turn_ratios else 0.0,
    }
    return trajectories, stats

def evaluate(args):
    conf = load_config(args.run_dir)
    seed = int(getattr(args, "seed", conf.get("seed", 0)))
    set_global_seed(seed)
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
    if str(conf.get("env_id", "OdorHold-v3")).endswith("-v4"):
        env_kwargs.update({
            'reward_mode': conf.get('reward_mode', 'dense'),
            'cast_penalty': conf.get('cast_penalty', 0.01),
            'odor_abs_weight': conf.get('odor_abs_weight', 0.0),
            'odor_delta_weight': conf.get('odor_delta_weight', 1.0),
            'cast_info_bonus': conf.get('cast_info_bonus', 0.0),
            'goal_hold_steps': conf.get('goal_hold_steps', 20),
            'goal_complete_bonus': conf.get('goal_complete_bonus', 1.0),
            'goal_exit_penalty': conf.get('goal_exit_penalty', 0.3),
            'terminate_on_hold': conf.get('terminate_on_hold', True),
        })
    
    # 1. Trajectory Eval Env
    env_id = conf.get("env_id", "OdorHold-v3")
    env_cls = OdorHoldEnvV4 if str(env_id).endswith("-v4") else OdorHoldEnv
    env = env_cls(**env_kwargs)
    env.action_space.seed(seed)
    if hasattr(env, "observation_space") and hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)
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
            critic_type=conf.get("critic_type", "recurrent"),
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
    trajectories, rollout_stats = _rollout_trajectories(env, agent, agent_type, args.episodes, args.seed_base)
    avg_return = np.mean([t['return'] for t in trajectories])
    
    print(f"  > Entry Success: {rollout_stats['success_entry_rate'] * 100:.1f}%")
    print(f"  > Hold Success:  {rollout_stats['success_hold_rate'] * 100:.1f}%")
    print(f"  > Final In-Goal: {rollout_stats['final_in_goal_rate'] * 100:.1f}%")
    print(f"  > Avg Return:   {avg_return:.2f}")
    print(f"  > Avg Cast Starts: {rollout_stats['cast_start_count_mean']:.2f} / episode")
    print(f"  > Avg Cast Steps:  {rollout_stats['cast_step_count_mean']:.2f} / episode")
    print(f"  > Cast Step %:  {rollout_stats['cast_step_ratio_mean'] * 100:.1f}%")
    print(f"  > Can-Turn %:   {rollout_stats['can_turn_ratio_mean'] * 100:.1f}%")

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
        ms = {}
        if os.path.exists(milestone_path):
            with open(milestone_path, "r") as f:
                ms = json.load(f)

        first_ckpt = os.path.join(args.run_dir, "checkpoints", "first.pt")
        if not os.path.exists(first_ckpt):
            # backward compatibility with older runs
            first_ckpt = os.path.join(args.run_dir, "checkpoints", "ep100.pt")

        ckpt_map = [
            ("first", first_ckpt, int(ms.get("first_ep", -1))),
            ("mid", os.path.join(args.run_dir, "checkpoints", "mid.pt"), int(ms.get("mid_saved_ep", -1))),
            ("best", os.path.join(args.run_dir, "checkpoints", "best.pt"), int(ms.get("best_ep", -1))),
        ]

        plotted_any = False
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
            ep_text = f"ep={ep_i}" if ep_i >= 1 else "ep=unknown"
            title_i = f"Eval {label} ({ep_text})"
            out_png = os.path.join(plots_dir, f"trajectory_{label}.png")
            plotter.plot_trajs_png(env_kwargs, out_png, traj_i, title_i)
            plotted_any = True

        if not plotted_any:
            print("[Warn] No milestone checkpoints found for trajectory plotting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=None)
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
