import argparse


def build_parser():
    parser = argparse.ArgumentParser(description="End-to-End RL Pipeline")

    parser.add_argument("--env-id", default="OdorHold-v4")
    parser.add_argument("--out-dir", default="runs", help="Root directory for results")
    parser.add_argument("--run-name", default=None, help="Optional fixed run name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-cpu", action="store_true")

    parser.add_argument("--agent-type", choices=["drqn", "dqn", "rsac"], default="rsac")
    parser.add_argument("--total-episodes", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=3e-4)
    parser.add_argument("--lr-alpha", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--rnn-hidden", type=int, default=147)
    parser.add_argument("--rsac-actor-backbone", choices=["gru", "connectome", "connectome2"], default="gru")
    parser.add_argument("--connectome-steps", type=int, default=4)
    parser.add_argument("--connectome-hidden", type=int, default=180)
    parser.add_argument("--dqn-hidden", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=16)

    parser.add_argument("--buffer-size", type=int, default=150000)
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--target-update-every", type=int, default=20)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-steps", type=int, default=4000)
    parser.add_argument("--log-every", type=int, default=20)

    parser.add_argument("--src-x", type=float, default=0.0)
    parser.add_argument("--src-y", type=float, default=0.0)
    parser.add_argument("--wind-x", type=float, default=0.0)
    parser.add_argument("--sigma-c", type=float, default=1.0)
    parser.add_argument("--reward-mode", choices=["mechanical", "bio"], default="bio")
    parser.add_argument("--bio-reward-scale", type=float, default=0.5)
    parser.add_argument("--cast-penalty", type=float, default=0.025)
    parser.add_argument("--turn-penalty", type=float, default=0.01)
    parser.add_argument("--b-hold", type=float, default=0.5)
    parser.add_argument("--goal-hold-steps", type=int, default=20)
    parser.add_argument("--terminate-on-hold", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--eval-episodes", type=int, default=100, help="Number of episodes for evaluation")
    parser.add_argument("--seed-base", type=int, default=20000)
    parser.add_argument("--save-gif", action=argparse.BooleanOptionalAction, default=True,
                        help="Save rollout GIF during evaluation (default: True)")
    parser.add_argument("--plot-milestones", action=argparse.BooleanOptionalAction, default=True,
                        help="Render milestone trajectory PNGs when available")

    return parser