import os
import time
import argparse

from train import train
from eval import evaluate
from src.utils.seed import set_global_seed


def main():
    parser = argparse.ArgumentParser(description="End-to-End RL Pipeline")
    
    parser.add_argument("--env-id", default="OdorHold-v3")
    parser.add_argument("--out-dir", default="runs", help="Root directory for results")
    parser.add_argument("--run-name", default=None, help="Optional fixed run name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-cpu", action="store_true")

    parser.add_argument("--agent-type", choices=["drqn", "dqn", "rsac"], default="drqn")
    parser.add_argument("--total-episodes", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=3e-4)
    parser.add_argument("--lr-alpha", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--rnn-cell", choices=["gru", "rnn"], default="gru")
    parser.add_argument("--critic-type", choices=["recurrent", "mlp"], default="recurrent")
    parser.add_argument("--rnn-hidden", type=int, default=147)
    parser.add_argument("--dqn-hidden", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=16)
    
    # Keep names aligned with train.py arguments.
    parser.add_argument("--buffer-size", type=int, default=150000)
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--target-update-every", type=int, default=20)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-steps", type=int, default=4000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-milestones", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--first-milestone-ep", type=int, default=100)
    parser.add_argument("--milestone-every", type=int, default=10)
    
    # Env params
    parser.add_argument("--src-x", type=float, default=0.0)
    parser.add_argument("--src-y", type=float, default=0.0)
    parser.add_argument("--wind-x", type=float, default=0.0)
    parser.add_argument("--sigma-c", type=float, default=1.0)
    parser.add_argument("--reward-mode", choices=["dense", "bio"], default="bio")
    parser.add_argument("--cast-penalty", type=float, default=0.03)
    parser.add_argument("--odor-abs-weight", type=float, default=0.0)
    parser.add_argument("--odor-delta-weight", type=float, default=1.0)
    parser.add_argument("--cast-info-bonus", type=float, default=0.0)
    parser.add_argument("--goal-hold-steps", type=int, default=20)
    parser.add_argument("--goal-complete-bonus", type=float, default=1.0)
    parser.add_argument("--goal-exit-penalty", type=float, default=0.3)
    parser.add_argument("--terminate-on-hold", action=argparse.BooleanOptionalAction, default=True)

    # Eval/plot params
    parser.add_argument("--eval-episodes", type=int, default=100, help="Number of episodes for evaluation")
    parser.add_argument("--seed-base", type=int, default=20000)
    parser.add_argument(
        "--save-gif",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save rollout GIF during evaluation (default: True)",
    )
    parser.add_argument(
        "--plot-milestones",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render milestone trajectory PNGs when available",
    )

    args = parser.parse_args()
    set_global_seed(args.seed)

    if args.run_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.agent_type}_main_{timestamp}"
    else:
        run_name = str(args.run_name)
    
    args.run_name = run_name
    
    print(f"\n{'=' * 40}")
    print(f"[Step 1] Training started: {run_name}")
    print(f"{'=' * 40}")
    
    try:
        train(args)
    except KeyboardInterrupt:
        print("\n[Warn] Training interrupted by user. Proceeding to evaluation...")
    except Exception as e:
        print(f"[Error] Training failed: {e}")
        return

    run_dir = os.path.join(args.out_dir, run_name)
    
    if not os.path.exists(run_dir):
        print(f"[Error] Run directory not found at {run_dir}")
        return

    print(f"\n[Info] Training finished. Results saved at: {run_dir}")
    print(f"\n{'=' * 40}")
    print("[Step 2] Evaluation & plotting")
    print(f"{'=' * 40}")

    args.run_dir = run_dir
    args.episodes = args.eval_episodes
    args.ckpt = None  # Auto-detect best.pt

    try:
        evaluate(args)
    except Exception as e:
        print(f"[Error] Evaluation failed: {e}")
        return

    print(f"\n{'=' * 40}")
    print(f"[Done] All jobs completed. Check '{run_dir}' for plots.")
    print(f"{'=' * 40}")

if __name__ == "__main__":
    main()
