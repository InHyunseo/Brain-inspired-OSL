"""Regenerate plots from saved plot_data without re-training.

For PPO runs only training-data plots are minimal (sb3 logs to TensorBoard
separately) — this script primarily targets the RSAC / DRQN curves.
"""
import argparse

from src.utils import plotter
from src.utils.seed import set_global_seed


def main():
    p = argparse.ArgumentParser(description="Regenerate PNG plots.")
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_global_seed(args.seed)
    plotter.plot_training_pngs_from_data(args.run_dir)
    print("[Replot] Regenerated returns.png and steps_to_goal.png")


if __name__ == "__main__":
    main()
