import os
import time
import json
import shutil
import tempfile
import numpy as np
import torch

from src.utils.buffer import EpisodeReplayBuffer
from src.utils import plotter
from src.utils.seed import set_global_seed
from src.utils.factory import build_env_kwargs, make_env, make_agent, make_agent_from_conf
from src.envs.odor_env_v3 import OdorHoldEnv
from src.envs.odor_env_v4 import OdorHoldEnvV4


class Args:
    pass


def build_args():
    args = Args()

    # ---------------------------
    # experiment identity
    # ---------------------------
    args.exp_name = "exp1_baseline"
    args.env_id = "OdorHold-v4"
    args.out_dir = "runs"
    args.run_name = None

    # ---------------------------
    # system
    # ---------------------------
    args.seed = 42
    args.force_cpu = False

    # ---------------------------
    # agent / training
    # ---------------------------
    args.agent_type = "rsac"
    args.total_episodes = 20000

    args.lr = 2e-4
    args.lr_actor = 3e-4
    args.lr_critic = 3e-4
    args.lr_alpha = 3e-4
    args.gamma = 0.99
    args.tau = 0.005

    args.rnn_hidden = 147
    args.rsac_actor_backbone = "gru"
    args.connectome_steps = 4
    args.connectome_hidden = 180
    args.dqn_hidden = 256

    args.batch_size = 128
    args.seq_len = 16
    args.buffer_size = 150000
    args.learning_starts = 5000
    args.target_update_every = 20

    args.eps_start = 1.0
    args.eps_end = 0.05
    args.eps_decay_steps = 4000
    args.log_every = 20

    # ---------------------------
    # env
    # ---------------------------
    args.src_x = 0.0
    args.src_y = 0.0
    args.wind_x = 0.0
    args.sigma_c = 1.0
    args.reward_mode = "bio"
    args.bio_reward_scale = 0.5
    args.cast_penalty = 0.025
    args.turn_penalty = 0.01
    args.b_hold = 0.5
    args.goal_hold_steps = 20
    args.terminate_on_hold = True

    # ---------------------------
    # evaluation
    # ---------------------------
    args.eval_episodes = 100
    args.seed_base = 20000
    args.save_gif = True
    args.plot_milestones = False 

    # runtime-only
    args.run_dir = None
    args.ckpt = None
    args.episodes = None

    return args


def _safe_exists(path):
    try:
        return os.path.exists(path)
    except OSError:
        return False


def _default_recovery_root():
    if _safe_exists("/content"):
        return "/content/osl_recovery"
    return "/tmp/osl_recovery"


def load_config(run_dir):
    config_path = os.path.join(run_dir, "config.json")
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

    for ep in range(episodes):
        ep_seed = seed_base + ep
        obs, _ = env.reset(seed=ep_seed)
        base_env = env.unwrapped

        h = None
        done = False
        ep_ret = 0.0

        xs, ys = [base_env.x], [base_env.y]

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
            done = bool(term or trunc)
            total_steps += 1

            cast_start_count += int(info.get("did_cast", 0))
            cast_step_count += int(info.get("in_cast", 0))
            cast_steps += int(info.get("in_cast", 0))
            can_turn_steps += int(info.get("can_turn", 0))

            xs.append(base_env.x)
            ys.append(base_env.y)
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

        trajectories.append(
            {
                "return": ep_ret,
                "success": in_goal,
                "x": xs,
                "y": ys,
                "seed": ep_seed,
            }
        )

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


def train_experiment(args):
    set_global_seed(args.seed)

    first_milestone_ep = 100
    mid_snapshot_every = 10

    if args.run_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.run_name = f"{args.exp_name}_{args.agent_type}_{timestamp}"

    run_dir = os.path.abspath(os.path.join(args.out_dir, args.run_name))
    ckpt_dir = os.path.join(run_dir, "checkpoints")

    recovery_root = os.environ.get("OSL_RECOVERY_DIR", _default_recovery_root())
    recovery_run_dir = os.path.join(recovery_root, args.run_name)
    recovery_ckpt_dir = os.path.join(recovery_run_dir, "checkpoints")

    primary_io_ok = True
    try:
        os.makedirs(ckpt_dir, exist_ok=True)
    except OSError as e:
        primary_io_ok = False
        print(f"[Warn] Primary output dir unavailable ({ckpt_dir}): {e}")

    os.makedirs(recovery_ckpt_dir, exist_ok=True)

    tmp_mid_ctx = tempfile.TemporaryDirectory(prefix=f"{args.run_name}_mid_", dir="/tmp")
    tmp_mid_dir = tmp_mid_ctx.name

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print(f"[Info] Training on {device}")

    if primary_io_ok:
        try:
            with open(os.path.join(run_dir, "config.json"), "w") as f:
                json.dump(vars(args), f, indent=2)
        except OSError as e:
            primary_io_ok = False
            print(f"[Warn] Failed to write primary config.json: {e}")

    with open(os.path.join(recovery_run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    env_kwargs = build_env_kwargs(args)
    env = make_env(args.env_id, **env_kwargs)
    env.action_space.seed(args.seed)
    if hasattr(env, "observation_space") and hasattr(env.observation_space, "seed"):
        env.observation_space.seed(args.seed)

    agent = make_agent(args, env, device)
    buffer = EpisodeReplayBuffer(args.buffer_size)

    ep_returns = []
    ep_steps_to_goal = []
    best_return = -np.inf
    best_ep = 0
    saved_eps = []

    def save_checkpoint_dual(agent_obj, filename):
        nonlocal primary_io_ok
        wrote_any = False

        if primary_io_ok:
            try:
                agent_obj.save(os.path.join(ckpt_dir, filename))
                wrote_any = True
            except OSError as e:
                primary_io_ok = False
                print(f"[Warn] Primary checkpoint save failed ({filename}): {e}")

        try:
            agent_obj.save(os.path.join(recovery_ckpt_dir, filename))
            wrote_any = True
        except OSError as e:
            print(f"[Warn] Recovery checkpoint save failed ({filename}): {e}")

        return wrote_any

    try:
        for ep in range(1, args.total_episodes + 1):
            obs, _ = env.reset(seed=args.seed + ep)
            h = None
            done = False
            ep_ret = 0.0
            traj = []
            step_count = 0
            first_goal_step = None

            frac = min(1.0, ep / args.eps_decay_steps)
            epsilon = args.eps_start - frac * (args.eps_start - args.eps_end)
            log_stats = None

            is_rsac = (args.agent_type == "rsac")
            sample_batch = buffer.sample_continuous if is_rsac else buffer.sample

            while not done:
                if is_rsac:
                    action, h_next = agent.get_action(obs, h, epsilon=1.0)
                else:
                    action, h_next = agent.get_action(obs, h, epsilon)

                next_obs, reward, term, trunc, info = env.step(action)
                done = bool(term or trunc)
                terminal = float(term)
                step_count += 1

                if first_goal_step is None and info.get("in_goal", 0):
                    first_goal_step = step_count

                traj.append((obs, action, float(reward), next_obs, terminal))
                obs = next_obs
                h = h_next
                ep_ret += reward

                if len(buffer) > args.learning_starts and len(buffer) > args.batch_size * args.seq_len:
                    batch = sample_batch(args.batch_size, args.seq_len)
                    log_stats = agent.update(batch)

            buffer.add_episode(traj)
            ep_returns.append(ep_ret)
            ep_steps_to_goal.append(step_count if first_goal_step is None else first_goal_step)

            if ep_ret > best_return:
                best_return = ep_ret
                best_ep = ep
                save_checkpoint_dual(agent, "best.pt")

            if ep == first_milestone_ep:
                save_checkpoint_dual(agent, "first.pt")

            if ep >= first_milestone_ep and ep % mid_snapshot_every == 0:
                pep = os.path.join(tmp_mid_dir, f"ep_{ep}.pt")
                try:
                    agent.save(pep)
                    if ep not in saved_eps:
                        saved_eps.append(ep)
                except OSError as e:
                    print(f"[Warn] Temporary snapshot save failed ({pep}): {e}")

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
                    print(
                        f"Ep {ep} | Avg Ret: {avg_ret:.2f} | Avg Step-to-Goal: {avg_steps:.1f} | Eps: {epsilon:.3f}"
                    )

    except KeyboardInterrupt:
        print("\n[Warn] Training interrupted. Saving partial artifacts...")

    finally:
        if best_ep > 0:
            first_primary = os.path.join(ckpt_dir, "first.pt")
            first_recovery = os.path.join(recovery_ckpt_dir, "first.pt")
            best_primary = os.path.join(ckpt_dir, "best.pt")
            best_recovery = os.path.join(recovery_ckpt_dir, "best.pt")

            cands = []
            if _safe_exists(best_primary):
                cands.append((int(best_ep), best_primary))
            if _safe_exists(best_recovery):
                cands.append((int(best_ep), best_recovery))
            if not cands:
                p_best_tmp = os.path.join(tmp_mid_dir, f"ep_{best_ep}.pt")
                if _safe_exists(p_best_tmp):
                    cands = [(int(best_ep), p_best_tmp)]

            if cands:
                first_ep = -1
                first_path = None
                if _safe_exists(first_primary):
                    first_path = first_primary
                elif _safe_exists(first_recovery):
                    first_path = first_recovery

                if first_path is not None:
                    first_ep = int(first_milestone_ep)
                    cands.append((first_ep, first_path))

                lo = min(ep for ep, _ in cands)
                hi = max(ep for ep, _ in cands)
                mid_target = (lo + hi) // 2

                for ep in sorted(set(saved_eps)):
                    if lo <= ep <= hi:
                        p_ep = os.path.join(tmp_mid_dir, f"ep_{ep}.pt")
                        if _safe_exists(p_ep):
                            cands.append((int(ep), p_ep))

                mid_ep, mid_src = min(cands, key=lambda item: abs(item[0] - mid_target))

                if primary_io_ok:
                    try:
                        shutil.copy2(mid_src, os.path.join(ckpt_dir, "mid.pt"))
                    except OSError as e:
                        primary_io_ok = False
                        print(f"[Warn] Failed to copy mid.pt to primary dir: {e}")

                try:
                    shutil.copy2(mid_src, os.path.join(recovery_ckpt_dir, "mid.pt"))
                except OSError as e:
                    print(f"[Warn] Failed to copy mid.pt to recovery dir: {e}")

                plot_data_dir = os.path.join(run_dir, "plot_data")
                os.makedirs(plot_data_dir, exist_ok=True)

                milestones = {
                    "first_ep": int(first_ep),
                    "mid_target_ep": int(mid_target),
                    "mid_saved_ep": int(mid_ep),
                    "best_ep": int(best_ep),
                }

                with open(os.path.join(plot_data_dir, "milestones.json"), "w") as f:
                    json.dump(milestones, f, indent=2)

        plots_dir = os.path.join(run_dir, "plots")
        plot_data_dir = os.path.join(run_dir, "plot_data")
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(plot_data_dir, exist_ok=True)

        plotter.plot_returns(ep_returns, os.path.join(plots_dir, "returns.png"))
        plotter.plot_steps_to_goal(ep_steps_to_goal, os.path.join(plots_dir, "steps_to_goal.png"))

        env.close()
        tmp_mid_ctx.cleanup()

    return {
        "run_dir": run_dir,
        "best_ep": best_ep,
        "best_return": float(best_return) if best_ep > 0 else None,
    }


def evaluate_experiment(args):
    conf = load_config(args.run_dir)

    seed = int(conf.get("seed", 0))
    set_global_seed(seed)

    device = torch.device("cpu") if args.force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Evaluating on {device}")

    env_id = conf.get("env_id", "OdorHold-v4")
    env_kwargs = {
        "L": conf.get("L", 3.0),
        "dt": conf.get("dt", 0.1),
        "src_x": conf.get("src_x", 0.0),
        "src_y": conf.get("src_y", 0.0),
        "wind_x": conf.get("wind_x", 0.0),
        "wind_y": conf.get("wind_y", 0.0),
        "sigma_c": conf.get("sigma_c", 1.0),
        "b_hold": conf.get("b_hold", 0.5),
        "r_goal": conf.get("r_goal", 0.35),
    }

    if str(env_id).endswith("-v4"):
        env_kwargs.update({
            "reward_mode": conf.get("reward_mode", "bio"),
            "bio_reward_scale": conf.get("bio_reward_scale", 0.5),
            "cast_penalty": conf.get("cast_penalty", 0.025),
            "turn_penalty": conf.get("turn_penalty", 0.01),
            "goal_hold_steps": conf.get("goal_hold_steps", 20),
            "terminate_on_hold": conf.get("terminate_on_hold", True),
        })
    else:
        env_kwargs.update({
            "v_fixed": conf.get("v_fixed", 0.25),
        })

    env_cls = OdorHoldEnvV4 if str(env_id).endswith("-v4") else OdorHoldEnv

    env = make_env(env_id, **env_kwargs)
    env.action_space.seed(seed)
    if hasattr(env, "observation_space") and hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)

    agent_type = conf.get("agent_type", "drqn")
    agent = make_agent_from_conf(conf, env, device)

    ckpt_name = "best.pt"
    ckpt_path = os.path.join(args.run_dir, "checkpoints", ckpt_name)
    print(f"[Info] Loading model from {ckpt_path}")
    agent.load(ckpt_path)

    print(f"[Info] Starting evaluation over {args.eval_episodes} episodes...")
    trajectories, rollout_stats = _rollout_trajectories(env, agent, agent_type, args.eval_episodes, args.seed_base)
    avg_return = np.mean([t["return"] for t in trajectories])

    print(f"  > Entry Success: {rollout_stats['success_entry_rate'] * 100:.1f}%")
    print(f"  > Hold Success:  {rollout_stats['success_hold_rate'] * 100:.1f}%")
    print(f"  > Final In-Goal: {rollout_stats['final_in_goal_rate'] * 100:.1f}%")
    print(f"  > Avg Return:    {avg_return:.2f}")
    print(f"  > Avg Cast Starts: {rollout_stats['cast_start_count_mean']:.2f}")
    print(f"  > Avg Cast Steps:  {rollout_stats['cast_step_count_mean']:.2f}")
    print(f"  > Cast Step %:     {rollout_stats['cast_step_ratio_mean'] * 100:.1f}%")
    print(f"  > Can-Turn %:      {rollout_stats['can_turn_ratio_mean'] * 100:.1f}%")

    plots_dir = os.path.join(args.run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plotter.plot_trajs_png(
        env_kwargs,
        os.path.join(plots_dir, "trajectory_best.png"),
        trajectories,
        title="Eval best checkpoint",
    )

    env.close()

    if args.save_gif:
        print("[Info] Generating GIF...")
        env_gif = env_cls(render_mode="rgb_array", **env_kwargs)

        frames = []
        best_traj = max(trajectories, key=lambda t: float(t["return"])) if trajectories else None
        gif_seed = int(best_traj["seed"]) if best_traj is not None else int(args.seed_base)
        gif_return = float(best_traj["return"]) if best_traj is not None else float("nan")

        obs, _ = env_gif.reset(seed=gif_seed)
        h = None
        done = False

        frames.append(
            plotter.render_rollout_frame_png_style(
                env_gif,
                title=f"Eval best | seed={gif_seed} | return={gif_return:.2f}",
            )
        )

        while not done:
            if agent_type == "rsac":
                action, h = agent.get_action_deterministic(obs, h)
            else:
                action, h = agent.get_action(obs, h, epsilon=0.0)

            obs, _, term, trunc, _ = env_gif.step(action)
            done = bool(term or trunc)

            frames.append(
                plotter.render_rollout_frame_png_style(
                    env_gif,
                    title=f"Eval best | seed={gif_seed} | return={gif_return:.2f}",
                )
            )

        env_gif.close()

        gif_path = os.path.join(plots_dir, "best_agent.gif")
        if frames:
            plotter.save_gif(frames, gif_path, fps=30)

    if args.plot_milestones:
        milestone_path = os.path.join(args.run_dir, "plot_data", "milestones.json")
        ms = {}
        if os.path.exists(milestone_path):
            with open(milestone_path, "r") as f:
                ms = json.load(f)

        ckpt_map = [
            ("first", os.path.join(args.run_dir, "checkpoints", "first.pt"), int(ms.get("first_ep", -1))),
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
            traj_i, _ = _rollout_trajectories(env_m, agent, agent_type, args.eval_episodes, args.seed_base)
            env_m.close()

            ep_text = f"ep={ep_i}" if ep_i >= 1 else "ep=unknown"
            out_png = os.path.join(plots_dir, f"trajectory_{label}.png")
            plotter.plot_trajs_png(env_kwargs, out_png, traj_i, title=f"Eval {label} ({ep_text})")


def run_experiment():
    args = build_args()
    set_global_seed(args.seed)

    print(f"\n{'=' * 50}")
    print(f"[Experiment] {args.exp_name}")
    print(f"{'=' * 50}")

    train_result = train_experiment(args)
    args.run_dir = train_result["run_dir"]

    print(f"\n[Info] Training finished. Results saved at: {args.run_dir}")
    print(f"\n{'=' * 50}")
    print("[Step] Evaluation & plotting")
    print(f"{'=' * 50}")

    evaluate_experiment(args)

    print(f"\n{'=' * 50}")
    print(f"[Done] Experiment completed: {args.run_dir}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    run_experiment()