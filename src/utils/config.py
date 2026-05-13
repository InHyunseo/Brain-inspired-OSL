"""CLI argument parser for train.py / eval.py / main.py.

Two agent tracks share this parser: PPO (custom on-policy with separate
actor/critic, larva connectome actor) and RSAC (off-policy episode loop with
selectable backbone for ablations). Each uses only the subset of flags
relevant to it.
"""
from __future__ import annotations

import argparse
import json


# Default 4-phase noise curriculum: [noise_stage, noise_strength, timesteps].
# Stage 0 = clean Gaussian, stage 1 = static white noise, stage 2 = temporally
# correlated noise (advanced each step). Strength scales the noise variance.
DEFAULT_PHASES_JSON = json.dumps([
    [0, 0.0, 1_500_000],
    [1, 0.3, 500_000],
    [1, 0.6, 500_000],
    [2, 1.0, 1_000_000],
])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="2D OSL — bilateral sensor + larva connectome RL")

    # Run / device
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--run-dir", default=None, help="Existing run dir for eval/replot")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-cpu", action="store_true")

    # Agent
    p.add_argument("--agent-type", choices=["ppo", "rsac"], default="ppo")

    # Env (shared)
    p.add_argument("--sensor-spacing-mm", type=float, default=0.15)
    p.add_argument("--episode-seconds", type=float, default=120.0)
    p.add_argument("--arena-width-mm", type=float, default=80.0)
    p.add_argument("--arena-height-mm", type=float, default=120.0)
    p.add_argument("--source-x-mm", type=float, default=40.0)
    p.add_argument("--source-y-mm", type=float, default=100.0)
    p.add_argument("--gaussian-sigma-mm", type=float, default=30.0)
    p.add_argument("--success-radius-mm", type=float, default=7.5)

    # Reward shaping (biologically-structured energy budget)
    p.add_argument("--reward-goal", type=float, default=20.0,
                   help="Sparse food reward on success (terminal). Must dominate cumulative motion cost.")
    p.add_argument("--reward-log-k", type=float, default=0.05,
                   help="Coefficient on dlog(c)/dt chemotaxis shaping (clipped).")
    p.add_argument("--reward-log-clip", type=float, default=0.5,
                   help="Symmetric clip on the dlog/dt shaping term.")
    p.add_argument("--reward-conc-k", type=float, default=0.02,
                   help="Dense reward coefficient on normalized current concentration (c_avg / c_peak).")
    p.add_argument("--reward-time-penalty", type=float, default=-0.005,
                   help="Basal metabolism per step (alive cost).")
    p.add_argument("--reward-run-cost", type=float, default=-0.01,
                   help="Per-step coefficient on (v / v_max)^2.")
    p.add_argument("--reward-body-turn-cost", type=float, default=-0.005,
                   help="Per-step coefficient on (body_omega / max)^2.")
    p.add_argument("--reward-head-cast-cost", type=float, default=-0.02,
                   help="Per-step coefficient on (head_omega / max)^2 — most expensive motion.")
    p.add_argument("--reward-head-cast-stopped-mult", type=float, default=2.0,
                   help="Head-cast cost multiplier when the body is stopped (true 'cast' behaviour).")
    p.add_argument("--reward-spin-penalty", type=float, default=-0.05,
                   help="Penalty applied when the agent is classified as spinning.")
    p.add_argument("--wall-penalty", type=float, default=-2.0,
                   help="Terminal penalty on wall contact.")

    # Connectome (shared)
    p.add_argument("--weights-csv", dest="weights_csv",
                   default="assets/connectome/weights.csv")
    p.add_argument("--metadata-csv", dest="metadata_csv",
                   default="assets/connectome/metadata.csv")
    p.add_argument("--message-passing-steps", type=int, default=6)
    p.add_argument("--latent-dim", type=int, default=32)

    # PPO
    p.add_argument("--curriculum-phases", type=str, default=DEFAULT_PHASES_JSON,
                   help="JSON list of [noise_stage, noise_strength, timesteps].")
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--minibatch-envs", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-epsilon", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.005)
    p.add_argument("--value-loss-coef", type=float, default=0.5)
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--actor-max-grad-norm", type=float, default=0.5)
    p.add_argument("--critic-max-grad-norm", type=float, default=0.5)
    p.add_argument("--log-std-init", type=float, default=-0.5)
    p.add_argument("--log-every-updates", type=int, default=1)
    p.add_argument("--eval-interval-updates", type=int, default=10)
    p.add_argument("--eval-episodes-during-train", type=int, default=3,
                   dest="eval_episodes_during_train")
    p.add_argument("--checkpoint-every-timesteps", type=int, default=100_000)
    p.add_argument("--parallel-envs", action=argparse.BooleanOptionalAction, default=True)

    # RSAC
    p.add_argument("--total-episodes", type=int, default=20000)
    p.add_argument("--lr-actor", type=float, default=3e-4)
    p.add_argument("--lr-critic", type=float, default=3e-4)
    p.add_argument("--lr-alpha", type=float, default=3e-4)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--rnn-hidden", type=int, default=147)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--buffer-size", type=int, default=150_000)
    p.add_argument("--learning-starts", type=int, default=5000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--rsac-actor-backbone", choices=["gru", "connectome", "mlp"], default="connectome")
    # Eval-time noise schedule for RSAC (single env)
    p.add_argument("--rsac-noise-stage", type=int, default=0)
    p.add_argument("--rsac-noise-strength", type=float, default=0.0)

    # Eval (shared)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--seed-base", type=int, default=20000)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--save-gif", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval-noise-stage", type=int, default=2)
    p.add_argument("--eval-noise-strength", type=float, default=1.0)
    p.add_argument("--episodes", type=int, default=None, help="Alias for --eval-episodes")

    return p


def parse_curriculum_phases(args) -> list[tuple[int, float, int]]:
    """Parse --curriculum-phases JSON into a list of (noise_stage, noise_strength, timesteps)."""
    raw = json.loads(args.curriculum_phases)
    out = []
    for entry in raw:
        if len(entry) != 3:
            raise ValueError(f"Bad phase entry {entry}; expected [noise_stage, noise_strength, timesteps].")
        out.append((int(entry[0]), float(entry[1]), int(entry[2])))
    return out
