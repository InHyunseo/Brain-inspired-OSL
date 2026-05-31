"""End-to-end 2D analysis: collect traces → phase1 → 2a → 2b → 2c → 3a → 3b → 4.

Ports osl_analysis ``run_all.py`` to the 2D pipeline. Each checkpoint label is
rolled out then analyzed jointly. Phase 4 (live ablation) runs on a plain
``OslEnv`` so it works in Colab too.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from Analysis.osl2d import (
    eval_dump, phase1_label, phase2a_latent_viz, phase2b_probe,
    phase2c_neuron, phase3a_jacobian, phase3b_fixedpoint, phase4_ablation,
)


def main(argv=None):
    p = argparse.ArgumentParser("Analysis.osl2d.run_all")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoints", nargs="+", default=["best"],
                   help="Checkpoint labels to analyze (each rolled out then analyzed jointly).")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--episodes-per-seed", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--noise-stage", type=int, default=2)
    p.add_argument("--noise-strength", type=float, default=1.0)
    p.add_argument("--device", default=None)
    p.add_argument("--skip-collect", action="store_true",
                   help="Run analyses on existing traces; skip rollout/dump.")
    p.add_argument("--stochastic", action="store_true",
                   help="Sample actions during trace collection instead of using the "
                        "distribution mean. Needed for policies that only solve the task "
                        "with exploration noise (deterministic mean stalls on weak signal).")
    p.add_argument("--skip-ablation", action="store_true",
                   help="Skip Phase 4 (live-env ablation).")
    p.add_argument("--ablation-seeds", type=int, nargs="+", default=None)
    p.add_argument("--ablation-eps", type=int, default=3)
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument("--samples-per-label", type=int, default=200)
    p.add_argument("--probe-train-eps-now", type=int, default=None)
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)

    # Stochastic traces are written to `{label}__stoch` dirs (see eval_dump), so
    # the trace-consuming phases must look there. Checkpoint *loading* (phase4)
    # still uses the bare label.
    trace_labels = ([f"{cl}__stoch" for cl in args.checkpoints]
                    if args.stochastic else list(args.checkpoints))

    if not args.skip_collect:
        for cl in args.checkpoints:
            print(f"[run_all] collecting traces for ckpt='{cl}' "
                  f"({'stochastic' if args.stochastic else 'deterministic'}) ...")
            eval_dump.collect(
                run_dir=run_dir, ckpt_label=cl, seeds=args.seeds,
                episodes_per_seed=args.episodes_per_seed, max_steps=args.max_steps,
                noise_stage=args.noise_stage, noise_strength=args.noise_strength,
                device=args.device, stochastic=args.stochastic,
            )

    print("[run_all] phase1 ...")
    phase1_label.run(run_dir, trace_labels)
    print("[run_all] phase2a ...")
    phase2a_latent_viz.run(run_dir, trace_labels)
    print("[run_all] phase2b ...")
    phase2b_probe.run(run_dir, trace_labels, train_episodes_now=args.probe_train_eps_now)
    print("[run_all] phase2c ...")
    phase2c_neuron.run(run_dir, trace_labels, top_k=args.top_k)
    print("[run_all] phase3a ...")
    phase3a_jacobian.run(run_dir, trace_labels,
                         samples_per_label=args.samples_per_label, device=args.device)
    print("[run_all] phase3b ...")
    phase3b_fixedpoint.run(run_dir, trace_labels, device=args.device)

    if not args.skip_ablation:
        ck = args.checkpoints[0]
        print(f"[run_all] phase4 ablation (ckpt={ck}) ...")
        phase4_ablation.run(
            run_dir, ckpt_label=ck,
            seeds=args.ablation_seeds or args.seeds,
            episodes_per_seed=args.ablation_eps,
            max_steps=args.max_steps,
            noise_stage=args.noise_stage, noise_strength=args.noise_strength,
            device=args.device, stochastic=args.stochastic,
        )
    else:
        print("[run_all] phase4 skipped (--skip-ablation).")

    print(f"[run_all] DONE → {run_dir}/analysis/")


if __name__ == "__main__":
    main()
