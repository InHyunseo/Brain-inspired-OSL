import os
import json
import numpy as np
import torch

from src.utils import plotter
from src.utils.seed import set_global_seed
from src.utils.factory import make_env, make_agent_from_conf


class Args:
    pass


def build_args():
    args = Args()

    args.run_dir = "exp1_baseline_rsac_20260326_214908"

    args.eval_episodes = 100
    args.seed_base = 20000
    args.force_cpu = False
    args.save_gif = False

    return args


def load_config(run_dir):
    with open(os.path.join(run_dir, "config.json"), "r") as f:
        return json.load(f)


def rollout(env, agent, agent_type, episodes, seed_base):
    trajectories = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        base_env = env.unwrapped

        h = None
        done = False
        ep_ret = 0.0

        xs, ys = [base_env.x], [base_env.y]

        while not done:
            if agent_type == "rsac":
                action, h = agent.get_action_deterministic(obs, h)
            else:
                action, h = agent.get_action(obs, h, epsilon=0.0)

            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc

            xs.append(base_env.x)
            ys.append(base_env.y)
            ep_ret += r

        trajectories.append({
            "return": ep_ret,
            "x": xs,
            "y": ys,
        })

    return trajectories


def run_experiment():
    args = build_args()

    conf = load_config(args.run_dir)
    set_global_seed(conf.get("seed", 0))

    device = torch.device(
        "cpu" if args.force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"[Info] Device: {device}")

    env = make_env(conf.get("env_id", "OdorHold-v4"), **conf)

    agent = make_agent_from_conf(conf, env, device)
    agent_type = conf.get("agent_type", "drqn")

    ckpt_dir = os.path.join(args.run_dir, "checkpoints")
    plots_dir = os.path.join(args.run_dir, "plots")

    os.makedirs(plots_dir, exist_ok=True)

    ckpts = [
        ("first", "first.pt"),
        ("mid", "mid.pt"),
        ("best", "best.pt"),
    ]

    for label, ckpt_name in ckpts:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)

        if not os.path.exists(ckpt_path):
            print(f"[Skip] {ckpt_name} 없음")
            continue

        print(f"[Load] {ckpt_name}")
        agent.load(ckpt_path)

        trajectories = rollout(
            env,
            agent,
            agent_type,
            args.eval_episodes,
            args.seed_base,
        )

        out_png = os.path.join(plots_dir, f"trajectory_{label}.png")

        print(f"[Save] {out_png}")

        plotter.plot_trajs_png(
            conf,
            out_png,
            trajectories,
            title=f"{label} trajectory"
        )

    env.close()

    print("\n[Done] exp4 완료")
    print(f"→ 저장 위치: {plots_dir}")


if __name__ == "__main__":
    run_experiment()