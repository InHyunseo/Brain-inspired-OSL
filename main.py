import os
import time

from train import train
from eval import evaluate
from src.utils.seed import set_global_seed
from src.utils.config import build_parser


def _safe_exists(path):
    try:
        return os.path.exists(path)
    except OSError:
        return False


def main():
    args = build_parser().parse_args()
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
    
    train_result = None
    try:
        train_result = train(args)
    except KeyboardInterrupt:
        print("\n[Warn] Training interrupted by user. Proceeding to evaluation...")
    except Exception as e:
        print(f"[Error] Training failed: {e}")
        return

    run_dir = os.path.abspath(os.path.join(args.out_dir, run_name))
    if isinstance(train_result, dict):
        run_dir = str(train_result.get("run_dir", run_dir))
    
    if not _safe_exists(run_dir):
        print(f"[Error] Run directory not found at {run_dir}")
        if isinstance(train_result, dict):
            recovery_dir = train_result.get("recovery_run_dir")
            if recovery_dir:
                print(f"[Info] Recovery output may be available at: {recovery_dir}")
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
