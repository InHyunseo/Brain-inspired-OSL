"""End-to-end: PPO train + eval back-to-back."""
from __future__ import annotations

from train import _dump_config, _make_run_dir, train_ppo
from eval import _apply_conf_overrides, _load_conf, eval_ppo
from src.utils.config import build_parser
from src.utils.seed import set_global_seed


def main():
    args = build_parser().parse_args()
    if args.episodes is not None:
        args.eval_episodes = args.episodes

    set_global_seed(args.seed)
    run_dir = _make_run_dir(args)
    _dump_config(run_dir, args)
    print(f"[run_dir] {run_dir}")

    print("=" * 50)
    print("[Step 1] Training (PPO)")
    print("=" * 50)
    train_ppo(args, run_dir)

    print("\n" + "=" * 50)
    print("[Step 2] Evaluation (PPO)")
    print("=" * 50)
    args.run_dir = run_dir
    conf = _load_conf(run_dir)
    args = _apply_conf_overrides(args, conf)
    eval_ppo(args, run_dir)

    print(f"\n[done] outputs in {run_dir}")


if __name__ == "__main__":
    main()
