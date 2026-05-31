"""Dump per-episode 2D-OSL traces to npz for the analysis pipeline.

Rolls out a trained connectome/GRU policy deterministically on ``OslEnv`` and
records, per timestep: observation, actor hidden state, action, reward, the
behavior label (from event flags), kinematics, and the 6 raw event flags.

Output layout (mirrors osl_analysis ``analysis/traces``):
    {run_dir}/analysis/traces/{ckpt_label}/eval_seed{s}_ep{e:03d}.npz
    {run_dir}/analysis/traces/{ckpt_label}/group_indices.json   (once per ckpt)

Phase 1-3 scripts consume the npz files only. Phase 4 reuses the same env +
adapter for live ablation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.envs.osl_env import EnvConfig, OslEnv
from Analysis.osl2d.policy_adapter import Policy2DAdapter
from Analysis.osl2d.segment import EVENT_KEYS, labels_from_event_flags


def _resolve_ckpt(run_dir: Path, ckpt_label: str) -> Path:
    """Find a checkpoint file for `ckpt_label` under a run dir.

    Tries, in order: ``{run_dir}/checkpoints/{label}.pt``,
    ``{run_dir}/{label}.pt``, ``{run_dir}/ckpt_{label}.pt``. The special label
    ``"final"`` also matches ``ckpt_final.pt``.
    """
    candidates = [
        run_dir / "checkpoints" / f"{ckpt_label}.pt",
        run_dir / f"{ckpt_label}.pt",
        run_dir / f"ckpt_{ckpt_label}.pt",
        run_dir / "checkpoints" / f"ckpt_{ckpt_label}.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No checkpoint for label '{ckpt_label}' under {run_dir}. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def _env_config_from_ckpt(ckpt_path: Path, noise_stage: int, noise_strength: float) -> EnvConfig:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    env_cfg = dict(payload.get("env_config", {}))
    env_cfg["noise_stage"] = noise_stage
    env_cfg["noise_strength"] = noise_strength
    return EnvConfig.from_dict(env_cfg)


def rollout(adapter: Policy2DAdapter, env: OslEnv, seed: int, max_steps: int | None,
            stochastic: bool = False):
    obs, _ = env.reset(seed=seed)
    h = adapter.initial_state()
    T_cap = max_steps if max_steps is not None else 10_000

    # Per-episode torch generator so stochastic rollouts are reproducible and
    # decoupled from the env RNG (seed offset keeps it distinct from env seed).
    gen = None
    if stochastic:
        gen = torch.Generator(device=adapter.device)
        gen.manual_seed(int(seed) + 777)

    obs_buf, h_buf, act_buf, rew_buf = [], [], [], []
    kin_buf, ev_buf = [], []
    success = 0
    for _ in range(T_cap):
        if stochastic:
            action, h_next = adapter.step_stochastic(obs, h, patch=None, generator=gen)
        else:
            action, h_next = adapter.step_patched(obs, h, patch=None)
        next_obs, reward, term, trunc, info = env.step(action)

        obs_buf.append(np.asarray(obs, dtype=np.float32))
        h_buf.append(np.asarray(h, dtype=np.float32))
        act_buf.append(np.asarray(action, dtype=np.float32))
        rew_buf.append(float(reward))
        kin_buf.append([
            float(info["x_mm"]), float(info["y_mm"]), float(info["heading_rad"]),
            float(info["head_relative_angle_rad"]), float(info["v_mm_s"]),
            float(info["body_omega_rad_s"]), float(info["head_omega_rad_s"]),
            float(info["distance_to_source_mm"]), float(info["bearing_to_source_rad"]),
        ])
        ev_buf.append([float(info[k]) for k in EVENT_KEYS])

        if info.get("success") or info.get("is_success"):
            success = 1
        obs, h = next_obs, h_next
        if term or trunc:
            break

    events = np.asarray(ev_buf, dtype=np.float32)
    labels = labels_from_event_flags(events)
    return {
        "obs": np.stack(obs_buf).astype(np.float32),
        "h": np.stack(h_buf).astype(np.float32),
        "action": np.stack(act_buf).astype(np.float32),
        "reward": np.asarray(rew_buf, dtype=np.float32),
        "label": labels,
        "kinematics": np.asarray(kin_buf, dtype=np.float32),
        "events": events,
        "success": np.int64(success),
    }


# Column names for the `kinematics` array, for downstream readers.
KINEMATIC_KEYS = (
    "x_mm", "y_mm", "heading_rad", "head_relative_angle_rad", "v_mm_s",
    "body_omega_rad_s", "head_omega_rad_s", "distance_to_source_mm", "bearing_to_source_rad",
)


def collect(
    run_dir: str | Path,
    ckpt_label: str = "best",
    seeds=(0,),
    episodes_per_seed: int = 20,
    max_steps: int | None = None,
    noise_stage: int = 2,
    noise_strength: float = 1.0,
    device: str | None = None,
    stochastic: bool = False,
) -> Path:
    run_dir = Path(run_dir)
    ckpt_path = _resolve_ckpt(run_dir, ckpt_label)
    adapter = Policy2DAdapter.from_checkpoint(ckpt_path, device=device)
    env_cfg = _env_config_from_ckpt(ckpt_path, noise_stage, noise_strength)

    # Stochastic traces go to their own ``{label}__stoch`` trace dir so they are
    # never mixed into the deterministic Jacobian/fixed-point/ablation analyses
    # (those require a deterministic map). Use them for Phase 1 behavior stats.
    trace_label = f"{ckpt_label}__stoch" if stochastic else ckpt_label
    out_dir = run_dir / "analysis" / "traces" / trace_label
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group metadata (cell-type / third partition) once per ckpt.
    gi = {k: [int(i) for i in v] for k, v in adapter.group_indices.items()}
    gi["state_size"] = adapter.state_size
    gi["backbone"] = adapter.backbone_kind
    with (out_dir / "group_indices.json").open("w") as f:
        json.dump(gi, f, indent=2)

    n_saved = 0
    for s in seeds:
        env = OslEnv(env_cfg)
        for e in range(episodes_per_seed):
            ep_seed = 10_000 + int(s) * 1000 + e
            traj = rollout(adapter, env, ep_seed, max_steps, stochastic=stochastic)
            episode_id = int(s) * 10_000 + e
            out = out_dir / f"eval_seed{s}_ep{e:03d}.npz"
            np.savez_compressed(
                out, **traj, episode=e, seed=int(s), episode_id=episode_id,
                ckpt_label=trace_label,
                action_mode=("stochastic" if stochastic else "deterministic"),
            )
            n_saved += 1
            ret = float(traj["reward"].sum())
            print(f"[dump] ckpt={trace_label} seed={s} ep={e:03d} "
                  f"T={len(traj['reward']):4d} return={ret:8.2f} success={int(traj['success'])}")
    print(f"[dump] saved {n_saved} episodes → {out_dir}")
    return out_dir


def main(argv=None):
    p = argparse.ArgumentParser("Analysis.osl2d.eval_dump")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoints", nargs="+", default=["best"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--episodes-per-seed", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--noise-stage", type=int, default=2)
    p.add_argument("--noise-strength", type=float, default=1.0)
    p.add_argument("--device", default=None)
    p.add_argument("--stochastic", action="store_true",
                   help="Sample actions from the policy distribution instead of "
                        "the deterministic mean. Traces go to a '{label}__stoch' "
                        "dir — use for Phase 1 behavior stats, not Phase 2-4.")
    args = p.parse_args(argv)

    for cl in args.checkpoints:
        collect(
            run_dir=args.run_dir,
            ckpt_label=cl,
            seeds=args.seeds,
            episodes_per_seed=args.episodes_per_seed,
            max_steps=args.max_steps,
            noise_stage=args.noise_stage,
            noise_strength=args.noise_strength,
            device=args.device,
            stochastic=args.stochastic,
        )


if __name__ == "__main__":
    main()
