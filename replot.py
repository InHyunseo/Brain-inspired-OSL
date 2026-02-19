import argparse

from src.utils import plotter
from eval import evaluate


def main():
    p = argparse.ArgumentParser(description="Regenerate PNG plots.")
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument(
        "--target",
        choices=["all", "train", "eval"],
        default="all",
        help="Which plots to regenerate",
    )
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed-base", type=int, default=20000)
    p.add_argument("--force-cpu", action="store_true")
    args = p.parse_args()

    if args.target in ("all", "train"):
        plotter.plot_training_pngs_from_data(args.run_dir)
        print("[Replot] Regenerated returns.png and steps_to_goal.png")
    if args.target in ("all", "eval"):
        eval_args = argparse.Namespace(
            run_dir=args.run_dir,
            ckpt=None,
            episodes=args.episodes,
            seed_base=args.seed_base,
            force_cpu=args.force_cpu,
            save_gif=False,
            plot_milestones=True,
        )
        evaluate(eval_args)
        print("[Replot] Regenerated trajectory_first.png, trajectory_mid.png, trajectory_best.png")


if __name__ == "__main__":
    main()
