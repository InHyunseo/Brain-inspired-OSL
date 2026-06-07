"""Evaluate the trained PPO-GRU policy over the standard OSL noise sweep.

Saves a baseline-style 3-panel bar plot:
  1. success rate per noise condition
  2. mean steps-to-source over successful episodes
  3. high-cast-like step fraction per episode

Example:
    /home/hyunseo/anaconda3/envs/osl/bin/python analysis/ppo_gru_noise_sweep_eval.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.osl2d.eval_dump import _env_config_from_ckpt
from analysis.osl2d.policy_adapter import Policy2DAdapter
from src.envs.osl_env import OslEnv


DEFAULT_RUN_DIR = Path("/home/hyunseo/Personal_Research/OSL/runs/ppo_gru_nb_20260531_113633")
DEFAULT_SWEEP = (
    (0, 0.0),
    (1, 0.3),
    (1, 0.6),
    (1, 1.0),
    (2, 0.3),
    (2, 0.6),
    (2, 1.0),
)

SUCCESS_COLOR = "#2e8b57"
STEPS_COLOR = "#4682b4"
CAST_COLOR = "#d65f5f"


def _label(stage: int, strength: float) -> str:
    return f"s{stage}-a{strength:.1f}"


def run_episode(
    adapter: Policy2DAdapter,
    env: OslEnv,
    seed: int,
    *,
    stochastic: bool,
    max_steps: int | None,
) -> dict:
    obs, _ = env.reset(seed=seed)
    h = adapter.initial_state()
    step_cap = int(max_steps) if max_steps is not None else int(env.max_steps)
    generator = None
    if stochastic:
        generator = torch.Generator(device=adapter.device)
        generator.manual_seed(int(seed) + 777)

    total_return = 0.0
    casts = 0
    success = False
    final_info = {}

    for step_idx in range(step_cap):
        if stochastic:
            action, h_next = adapter.step_stochastic(obs, h, generator=generator)
        else:
            action, h_next = adapter.step_patched(obs, h)

        obs, reward, terminated, truncated, info = env.step(action)
        total_return += float(reward)
        casts += int(bool(info.get("event_is_high_cast_like", False)))
        success = bool(info.get("success", False))
        final_info = info
        h = h_next

        if terminated or truncated:
            break

    steps = step_idx + 1
    return {
        "seed": int(seed),
        "success": bool(success),
        "steps": int(steps),
        "return": float(total_return),
        "casts": int(casts),
        "cast_fraction": float(casts / max(steps, 1)),
        "final_distance_mm": float(final_info.get("distance_to_source_mm", float("nan"))),
    }


def evaluate_condition(
    adapter: Policy2DAdapter,
    ckpt_path: Path,
    stage: int,
    strength: float,
    *,
    episodes: int,
    seed_base: int,
    stochastic: bool,
    max_steps: int | None,
) -> dict:
    env = OslEnv(_env_config_from_ckpt(ckpt_path, stage, strength))
    episodes_out = []
    for idx in range(int(episodes)):
        seed = int(seed_base) + idx
        episodes_out.append(
            run_episode(
                adapter,
                env,
                seed,
                stochastic=stochastic,
                max_steps=max_steps,
            )
        )
    env.close()

    succ = np.asarray([int(ep["success"]) for ep in episodes_out], dtype=float)
    steps = np.asarray([ep["steps"] for ep in episodes_out], dtype=float)
    returns = np.asarray([ep["return"] for ep in episodes_out], dtype=float)
    casts = np.asarray([ep["casts"] for ep in episodes_out], dtype=float)
    cast_fraction = np.asarray([ep["cast_fraction"] for ep in episodes_out], dtype=float)
    succ_steps = steps[succ == 1]

    return {
        "stage": int(stage),
        "strength": float(strength),
        "label": _label(stage, strength),
        "n": int(len(episodes_out)),
        "success_rate": float(np.mean(succ)) if len(succ) else float("nan"),
        "mean_steps_all": float(np.mean(steps)) if len(steps) else float("nan"),
        "mean_steps_success": float(np.mean(succ_steps)) if len(succ_steps) else float("nan"),
        "mean_return": float(np.mean(returns)) if len(returns) else float("nan"),
        "mean_casts": float(np.mean(casts)) if len(casts) else float("nan"),
        "cast_fraction": float(np.mean(cast_fraction)) if len(cast_fraction) else float("nan"),
        "episodes": episodes_out,
    }


def plot_rows(rows: list[dict], out_path: Path, *, title_prefix: str) -> None:
    labels = [r["label"] for r in rows]
    success = [r["success_rate"] for r in rows]
    steps = [r["mean_steps_success"] for r in rows]
    cast_pct = [100.0 * r["cast_fraction"] for r in rows]

    fig, ax = plt.subplots(1, 3, figsize=(18, 4))
    ax[0].bar(labels, success, color=SUCCESS_COLOR)
    ax[0].set_ylim(0.0, 1.0)
    ax[0].set_ylabel("success rate")
    ax[0].set_title(f"{title_prefix} success vs. noise condition")

    ax[1].bar(labels, steps, color=STEPS_COLOR)
    ax[1].set_ylabel("mean steps to source (successful eps)")
    ax[1].set_title("Steps-to-source vs. noise condition")

    ax[2].bar(labels, cast_pct, color=CAST_COLOR)
    ax[2].set_ylabel("cast steps (% of episode)")
    ax[2].set_title("Cast fraction vs. noise condition")
    ymax = max([v for v in cast_pct if np.isfinite(v)] + [1.0])
    ax[2].set_ylim(0.0, ymax * 1.15)

    for axis in ax:
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--seed-base", type=int, default=20_000)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--out-name", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.torch_threads and args.torch_threads > 0:
        torch.set_num_threads(int(args.torch_threads))

    ckpt_path = args.ckpt or (args.run_dir / "ckpt_final.pt")
    adapter = Policy2DAdapter.from_checkpoint(ckpt_path, device=args.device)
    mode = "stochastic" if args.stochastic else "deterministic"
    print(f"[eval] ckpt={ckpt_path}")
    print(f"[eval] backbone={adapter.backbone_kind} mode={mode} episodes/condition={args.episodes}")

    rows = []
    for stage, strength in DEFAULT_SWEEP:
        row = evaluate_condition(
            adapter,
            ckpt_path,
            stage,
            strength,
            episodes=args.episodes,
            seed_base=args.seed_base,
            stochastic=bool(args.stochastic),
            max_steps=args.max_steps,
        )
        rows.append(row)
        print(
            f"{row['label']:>7} | success={row['success_rate']:5.1%} "
            f"| steps(succ)={row['mean_steps_success']:7.1f} "
            f"| return={row['mean_return']:7.2f} "
            f"| casts={row['mean_casts']:6.1f} "
            f"| cast%={row['cast_fraction']:5.1%}"
        )

    suffix = "stochastic" if args.stochastic else "deterministic"
    base = args.out_name or f"ppo_gru_noise_sweep_{suffix}"
    out_dir = args.run_dir / "plots"
    png_path = out_dir / f"{base}.png"
    json_path = out_dir / f"{base}.json"
    plot_rows(rows, png_path, title_prefix="PPO_GRU -")
    json_path.write_text(
        json.dumps(
            {
                "run_dir": str(args.run_dir),
                "ckpt": str(ckpt_path),
                "mode": mode,
                "episodes_per_condition": int(args.episodes),
                "seed_base": int(args.seed_base),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[saved] {png_path}")
    print(f"[saved] {json_path}")


if __name__ == "__main__":
    main()
