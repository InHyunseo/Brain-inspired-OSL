"""Analysis CLI for the CartPole actor-critic demo.

This keeps the analysis pivot local to demo/RL_practice: load a trained
checkpoint, dump deterministic rollout traces, probe hidden activations, and
measure simple hidden-unit patching effects.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from demo.RL_practice.train_cartpole_ac import ActorCritic, shaped_reward


CHECKPOINTS = ("init", "mid", "final")
PATCH_MODES = ("zero", "mean", "flip")


def parse_args():
    p = argparse.ArgumentParser(description="Analyze demo/RL_practice CartPole actor-critic runs.")
    p.add_argument("--run-dir", required=True, help="Path to a cartpole_ac_* run directory.")
    p.add_argument("--checkpoint", choices=CHECKPOINTS, default="final")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return torch.device(name)


def load_config(run_dir: Path) -> dict:
    path = run_dir / "config.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_model(run_dir: Path, checkpoint: str, env_id: str, device: torch.device) -> ActorCritic:
    ckpt_path = run_dir / "checkpoints" / f"{checkpoint}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {ckpt_path}. Run train() first, or choose a run directory "
            "that contains checkpoints/{init,mid,final}.pt."
        )

    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    env.close()

    model = ActorCritic(obs_dim, act_dim).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def deterministic_action(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits, dim=-1).item())


@torch.no_grad()
def rollout_trace(
    model: ActorCritic,
    env_id: str,
    seed: int,
    device: torch.device,
    reward_scale: float,
    patch: dict | None = None,
) -> dict:
    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)

    obs_buf = []
    action_buf = []
    raw_reward_buf = []
    shaped_reward_buf = []
    scaled_shaped_reward_buf = []
    logits_buf = []
    value_buf = []
    hidden_buf = []
    terminated_buf = []
    truncated_buf = []

    done = False
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        logits, value, hidden = model(obs_t, patch=patch, return_hidden=True)
        action = deterministic_action(logits)

        next_obs, raw_reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        shaped = float(shaped_reward(env, next_obs, terminated, truncated))

        obs_buf.append(np.asarray(obs, dtype=np.float32))
        action_buf.append(action)
        raw_reward_buf.append(float(raw_reward))
        shaped_reward_buf.append(shaped)
        scaled_shaped_reward_buf.append(shaped * reward_scale)
        logits_buf.append(logits.detach().cpu().numpy().astype(np.float32))
        value_buf.append(float(value.detach().cpu().item()))
        hidden_buf.append(hidden.detach().cpu().numpy().astype(np.float32))
        terminated_buf.append(bool(terminated))
        truncated_buf.append(bool(truncated))

        obs = next_obs

    env.close()
    return {
        "seed": int(seed),
        "obs": np.asarray(obs_buf, dtype=np.float32),
        "action": np.asarray(action_buf, dtype=np.int64),
        "raw_reward": np.asarray(raw_reward_buf, dtype=np.float32),
        "shaped_reward": np.asarray(shaped_reward_buf, dtype=np.float32),
        "scaled_shaped_reward": np.asarray(scaled_shaped_reward_buf, dtype=np.float32),
        "logits": np.asarray(logits_buf, dtype=np.float32),
        "value": np.asarray(value_buf, dtype=np.float32),
        "hidden": np.asarray(hidden_buf, dtype=np.float32),
        "terminated": np.asarray(terminated_buf, dtype=np.bool_),
        "truncated": np.asarray(truncated_buf, dtype=np.bool_),
    }


def trace_summary(trace: dict) -> dict:
    steps = int(len(trace["action"]))
    terminated = bool(trace["terminated"][-1]) if steps else False
    truncated = bool(trace["truncated"][-1]) if steps else False
    return {
        "steps": steps,
        "raw_return": float(np.sum(trace["raw_reward"])),
        "shaped_return": float(np.sum(trace["shaped_reward"])),
        "scaled_shaped_return": float(np.sum(trace["scaled_shaped_reward"])),
        "success": int(truncated and not terminated),
        "terminated": int(terminated),
        "truncated": int(truncated),
    }


def save_trace(path: Path, trace: dict, checkpoint: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        checkpoint=np.asarray(checkpoint),
        seed=np.asarray(trace["seed"], dtype=np.int64),
        obs=trace["obs"],
        action=trace["action"],
        raw_reward=trace["raw_reward"],
        shaped_reward=trace["shaped_reward"],
        scaled_shaped_reward=trace["scaled_shaped_reward"],
        logits=trace["logits"],
        value=trace["value"],
        hidden=trace["hidden"],
        terminated=trace["terminated"],
        truncated=trace["truncated"],
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_patch_specs(hidden_dim: int) -> list[dict]:
    first_half = np.arange(0, hidden_dim // 2, dtype=np.int64)
    second_half = np.arange(hidden_dim // 2, hidden_dim, dtype=np.int64)
    specs = []
    group_specs = [
        ("all_hidden", np.arange(hidden_dim, dtype=np.int64)),
        ("first_half", first_half),
        ("second_half", second_half),
    ]
    for target, indices in group_specs:
        for mode in PATCH_MODES:
            specs.append({"target": target, "mode": mode, "unit": "", "indices": indices})
    for unit in range(hidden_dim):
        specs.append(
            {
                "target": f"unit_{unit:03d}",
                "mode": "zero",
                "unit": unit,
                "indices": np.asarray([unit], dtype=np.int64),
            }
        )
    return specs


def fit_probe(hidden: np.ndarray, target: np.ndarray, seed: int) -> dict:
    n = int(hidden.shape[0])
    if n < 2:
        return {"n_samples": n, "train_samples": 0, "test_samples": 0, "r2": float("nan")}

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    split = int(max(1, min(n - 1, round(n * 0.8))))
    train_idx = order[:split]
    test_idx = order[split:]

    x_train = np.concatenate(
        [hidden[train_idx], np.ones((len(train_idx), 1), dtype=np.float32)],
        axis=1,
    )
    x_test = np.concatenate(
        [hidden[test_idx], np.ones((len(test_idx), 1), dtype=np.float32)],
        axis=1,
    )
    y_train = target[train_idx].astype(np.float32)
    y_test = target[test_idx].astype(np.float32)

    coef, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
    pred = x_test @ coef
    ss_res = float(np.sum((y_test - pred) ** 2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
    r2 = float("nan") if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
    return {
        "n_samples": n,
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "r2": r2,
    }


def run_analysis(args) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    config = load_config(run_dir)
    env_id = config.get("env_id", "CartPole-v1")
    reward_scale = float(config.get("reward_scale", 0.01))
    device = resolve_device(args.device)
    model = load_model(run_dir, args.checkpoint, env_id, device)

    eval_dir = run_dir / "eval"
    analysis_dir = run_dir / "analysis"
    episode_seeds = [int(args.seed + i) for i in range(args.episodes)]

    baseline_rows = []
    baseline_by_seed = {}
    hidden_parts = []
    theta_parts = []
    theta_dot_parts = []

    for ep_idx, seed in enumerate(episode_seeds):
        trace = rollout_trace(model, env_id, seed, device, reward_scale)
        trace_path = eval_dir / f"trace_{args.checkpoint}_seed{seed}_ep{ep_idx:03d}.npz"
        save_trace(trace_path, trace, args.checkpoint)

        summary = trace_summary(trace)
        baseline_by_seed[seed] = summary
        baseline_rows.append(
            {
                "checkpoint": args.checkpoint,
                "seed": seed,
                **summary,
                "trace_path": str(trace_path),
            }
        )

        hidden_parts.append(trace["hidden"])
        theta_parts.append(trace["obs"][:, 2])
        theta_dot_parts.append(trace["obs"][:, 3])

    write_csv(
        analysis_dir / "baseline_eval.csv",
        [
            "checkpoint",
            "seed",
            "steps",
            "raw_return",
            "shaped_return",
            "scaled_shaped_return",
            "success",
            "terminated",
            "truncated",
            "trace_path",
        ],
        baseline_rows,
    )

    hidden = np.concatenate(hidden_parts, axis=0)
    theta = np.concatenate(theta_parts, axis=0)
    theta_dot = np.concatenate(theta_dot_parts, axis=0)
    probe_rows = []
    for name, target in (("theta", theta), ("theta_dot", theta_dot)):
        result = fit_probe(hidden, target, seed=args.seed)
        probe_rows.append({"checkpoint": args.checkpoint, "target": name, **result})
    write_csv(
        analysis_dir / "probe_results.csv",
        ["checkpoint", "target", "n_samples", "train_samples", "test_samples", "r2"],
        probe_rows,
    )

    patch_rows = []
    for spec in build_patch_specs(hidden.shape[1]):
        patch = {"indices": spec["indices"], "value": spec["mode"]}
        for seed in episode_seeds:
            trace = rollout_trace(model, env_id, seed, device, reward_scale, patch=patch)
            summary = trace_summary(trace)
            base = baseline_by_seed[seed]
            patch_rows.append(
                {
                    "checkpoint": args.checkpoint,
                    "target": spec["target"],
                    "mode": spec["mode"],
                    "unit": spec["unit"],
                    "seed": seed,
                    **summary,
                    "baseline_steps": base["steps"],
                    "delta_steps": summary["steps"] - base["steps"],
                    "baseline_success": base["success"],
                    "delta_success": summary["success"] - base["success"],
                    "delta_raw_return": summary["raw_return"] - base["raw_return"],
                    "delta_shaped_return": summary["shaped_return"] - base["shaped_return"],
                    "delta_scaled_shaped_return": summary["scaled_shaped_return"] - base["scaled_shaped_return"],
                }
            )
    write_csv(
        analysis_dir / "patch_results.csv",
        [
            "checkpoint",
            "target",
            "mode",
            "unit",
            "seed",
            "steps",
            "raw_return",
            "shaped_return",
            "scaled_shaped_return",
            "success",
            "terminated",
            "truncated",
            "baseline_steps",
            "delta_steps",
            "baseline_success",
            "delta_success",
            "delta_raw_return",
            "delta_shaped_return",
            "delta_scaled_shaped_return",
        ],
        patch_rows,
    )

    mean_steps = float(np.mean([row["steps"] for row in baseline_rows]))
    mean_success = float(np.mean([row["success"] for row in baseline_rows]))
    print(
        f"analysis saved to {analysis_dir} | checkpoint={args.checkpoint} "
        f"episodes={args.episodes} mean_steps={mean_steps:.1f} success_rate={mean_success:.2f}"
    )
    return 0


def main():
    args = parse_args()
    try:
        raise SystemExit(run_analysis(args))
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[error] {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
