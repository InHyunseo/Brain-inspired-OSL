"""CLI argument parser for train.py / eval.py / main.py.

All three agent types (ppo / rsac / drqn) share this parser; each uses only
the subset of flags relevant to it.
"""
import argparse
import json


# Default 4-phase curriculum matching ipynb/PPO_framework.ipynb.
DEFAULT_PHASES_JSON = json.dumps([
    ["static", 1_500_000],
    ["dynamic_0.3", 500_000],
    ["dynamic_0.6", 500_000],
    ["dynamic_1.0", 1_000_000],
])


def build_parser():
    p = argparse.ArgumentParser(description="2D OSL End-to-End RL Pipeline")

    # Run / device
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--run-dir", default=None, help="Existing run dir for eval/replot")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-cpu", action="store_true")

    # Agent
    p.add_argument("--agent-type", choices=["ppo", "rsac", "drqn"], default="ppo")

    # PPO (sb3 RecurrentPPO)
    p.add_argument("--phases", type=str, default=DEFAULT_PHASES_JSON,
                   help="JSON list of [env_kind, total_steps] phases for PPO curriculum.")
    p.add_argument("--n-envs", type=int, default=16, help="SubprocVecEnv parallel envs (PPO)")
    p.add_argument("--features-dim", type=int, default=180,
                   help="Connectome extractor features_dim (multiple of 90)")
    p.add_argument("--ppo-lr", type=float, default=3e-4)
    p.add_argument("--ppo-batch-size", type=int, default=256)
    p.add_argument("--ppo-n-steps", type=int, default=128)
    p.add_argument("--ppo-ent-coef", type=float, default=0.01)
    p.add_argument("--tb-log", type=str, default=None,
                   help="TensorBoard log dir. Default: <run_dir>/tb")

    # RSAC / DRQN common (recurrent / episode-loop training)
    p.add_argument("--total-episodes", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-4, help="DRQN learning rate")
    p.add_argument("--lr-actor", type=float, default=3e-4)
    p.add_argument("--lr-critic", type=float, default=3e-4)
    p.add_argument("--lr-alpha", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--rnn-hidden", type=int, default=147)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--buffer-size", type=int, default=150000)
    p.add_argument("--learning-starts", type=int, default=5000)
    p.add_argument("--target-update-every", type=int, default=20)
    p.add_argument("--log-every", type=int, default=20)

    # RSAC
    p.add_argument("--rsac-actor-backbone", choices=["gru", "connectome", "mlp"], default="gru")
    p.add_argument("--connectome-steps", type=int, default=4)
    p.add_argument("--connectome-hidden", type=int, default=180)
    p.add_argument("--rsac-env-kind", choices=["static", "dynamic_0.3", "dynamic_0.6", "dynamic_1.0"],
                   default="dynamic_1.0", help="Single-env phase for RSAC training")

    # DRQN
    p.add_argument("--drqn-recurrent", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--drqn-env-kind", choices=["static", "dynamic_0.3", "dynamic_0.6", "dynamic_1.0"],
                   default="dynamic_1.0")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-steps", type=int, default=4000)

    # Eval
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--seed-base", type=int, default=20000)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--save-gif", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--noise-coef", type=float, default=1.0, help="DynamicEnv noise_coef for eval")
    p.add_argument("--episodes", type=int, default=None, help="Alias for --eval-episodes")

    return p


def parse_phases(args):
    """Parse --phases JSON string into a list of (env_kind, steps) tuples."""
    raw = json.loads(args.phases)
    out = []
    for entry in raw:
        if len(entry) != 2:
            raise ValueError(f"Bad phase entry {entry}; expected [env_kind, steps].")
        out.append((str(entry[0]), int(entry[1])))
    return out
