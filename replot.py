import argparse

from src.utils import plotter


def main():
    p = argparse.ArgumentParser(description="Regenerate PNG plots from saved plot_data files.")
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument(
        "--target",
        choices=["all", "train", "eval"],
        default="all",
        help="Which plots to regenerate from plot_data",
    )
    args = p.parse_args()

    if args.target in ("all", "train"):
        plotter.plot_training_pngs_from_data(args.run_dir)
        print("[Replot] Regenerated returns.png and steps_to_goal.png")
    if args.target in ("all", "eval"):
        plotter.plot_trajectory_png_from_data(args.run_dir)
        print("[Replot] Regenerated trajectory.png")


if __name__ == "__main__":
    main()
