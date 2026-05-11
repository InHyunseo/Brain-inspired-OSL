"""End-to-end: train + eval. Agent_type chooses the path inside each."""
from train import (
    _device,
    _dump_config,
    _make_run_dir,
    train_episode_loop,
    train_ppo,
)
from eval import (
    _load_conf,
    _apply_conf_overrides,
    eval_episode_loop,
    eval_ppo,
)
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
    print(f"[Step 1] Training ({args.agent_type})")
    print("=" * 50)
    if args.agent_type == "ppo":
        train_ppo(args, run_dir)
    else:
        train_episode_loop(args, run_dir)

    print("\n" + "=" * 50)
    print(f"[Step 2] Evaluation ({args.agent_type})")
    print("=" * 50)
    args.run_dir = run_dir
    conf = _load_conf(run_dir)
    args = _apply_conf_overrides(args, conf)
    if args.agent_type == "ppo":
        eval_ppo(args, run_dir)
    else:
        eval_episode_loop(args, run_dir)

    print(f"\n[done] outputs in {run_dir}")


if __name__ == "__main__":
    main()
