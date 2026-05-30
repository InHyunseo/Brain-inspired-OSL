"""Bilateral-sensor chemotaxis baseline (no neural network, no training).

A purely computational controller that maps the same 5-D observation the RL
policies see — ``[c_left, c_right, prev_v, prev_body_omega, prev_head_omega]`` —
to a continuous action ``[v, body_omega, head_omega] in [-1, 1]`` using classic
larva chemotaxis rules:

1. **Bilateral gradient steering** — turn the body toward whichever antenna
   reads higher concentration (``c_left`` vs ``c_right``). The steering command
   is proportional to the normalized left/right asymmetry.
2. **Surge / brake** — drive forward fast when the average concentration is
   *rising* (we're heading up-gradient); slow down and steer harder when it is
   *falling*.
3. **Cast recovery** — when the plume is weak or lost (low concentration for a
   streak of steps), stop surging and sweep the head left/right to actively
   re-acquire the gradient, alternating sweep direction.

The controller keeps a short concentration history internally (it is *not* given
the source location, distance, or bearing), so it is a fair sensor-only
comparison group for the connectome / GRU policies. In a clean (noise-free)
Gaussian field the bilateral gradient is exact, so this baseline solves the task
essentially every episode; under turbulent fields its success degrades
gracefully, which is exactly what the noise sweep visualizes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ChemotaxisConfig:
    # Steering gain on the normalized left/right asymmetry (→ body_omega).
    # The bilateral difference is tiny (sensor spacing 0.15 mm ≪ plume sigma
    # 30 mm), so a large gain is needed to act on its sign. At 80 the clean
    # (noise-free) field is solved every episode.
    steer_gain: float = 80.0
    # Forward speed when confidently up-gradient (surge) and when unsure.
    surge_speed: float = 1.0       # maps to action[0]=+1 → v_max
    cautious_speed: float = 0.2
    # Concentration-rising detector: EWMA smoothing of c_avg.
    ewma_alpha: float = 0.3
    # If smoothed c_avg drops below this fraction of the running max → "weak".
    weak_frac: float = 0.35
    # Steps of weak/declining signal before entering cast (head-sweep) mode.
    cast_after_weak_steps: int = 4
    # Head sweep amplitude (action[2]) and half-period (steps) while casting.
    cast_head_omega: float = 1.0
    cast_half_period: int = 6
    # Small forward creep while casting so we don't stall in place.
    cast_creep_speed: float = 0.15
    # Body turn while casting (slow scan rotation), sign flips with the sweep.
    cast_body_omega: float = 0.25


class BilateralChemotaxis:
    """Stateful sensor-only chemotaxis controller.

    Usage mirrors a policy: call :meth:`reset` at episode start, then
    :meth:`act(obs)` each step to get a ``(3,)`` action in ``[-1, 1]``.
    """

    def __init__(self, cfg: ChemotaxisConfig | None = None):
        self.cfg = cfg or ChemotaxisConfig()
        self.reset()

    def reset(self) -> None:
        self._ewma = None          # smoothed c_avg
        self._prev_ewma = None
        self._cmax = 1e-9          # running max of smoothed c_avg
        self._weak_streak = 0
        self._cast_dir = 1.0       # current head-sweep direction
        self._cast_phase = 0       # steps spent in the current sweep half

    def act(self, obs: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        c_left = float(obs[0])
        c_right = float(obs[1])
        c_avg = 0.5 * (c_left + c_right)

        # --- smoothing + rising/falling detection ---
        if self._ewma is None:
            self._ewma = c_avg
        self._prev_ewma = self._ewma
        self._ewma = (1.0 - cfg.ewma_alpha) * self._ewma + cfg.ewma_alpha * c_avg
        self._cmax = max(self._cmax, self._ewma)
        rising = self._ewma >= self._prev_ewma

        # Normalized bilateral asymmetry in [-1, 1]: +1 → left much stronger.
        denom = c_left + c_right + 1e-9
        asym = (c_left - c_right) / denom

        # Weak/lost detection relative to the best signal seen so far.
        weak = self._ewma < cfg.weak_frac * self._cmax
        if weak or not rising:
            self._weak_streak += 1
        else:
            self._weak_streak = 0

        if self._weak_streak >= cfg.cast_after_weak_steps:
            return self._cast_action()

        # --- gradient-following (surge + steer) ---
        self._cast_phase = 0  # reset cast sweep when we have signal
        # Turn the body toward the stronger antenna. Left stronger (asym>0) →
        # positive body_omega (env turns CCW); the env's geometry places "left"
        # at +90° from heading, so steering sign is +asym.
        body_omega = float(np.clip(cfg.steer_gain * asym, -1.0, 1.0))
        # Surge when rising and roughly aligned (small asymmetry); else slow to
        # steer. The alignment factor tapers speed as |asym| grows.
        align = 1.0 - min(1.0, abs(asym) * 2.0)
        speed = cfg.surge_speed if rising else cfg.cautious_speed
        v = cfg.cautious_speed + (speed - cfg.cautious_speed) * align
        v_action = float(np.clip(2.0 * v - 1.0, -1.0, 1.0))  # map [0,1]→[-1,1]
        head_omega = 0.0
        return np.asarray([v_action, body_omega, head_omega], dtype=np.float32)

    def _cast_action(self) -> np.ndarray:
        """Head-sweep recovery: alternate sweep direction every half period."""
        cfg = self.cfg
        self._cast_phase += 1
        if self._cast_phase >= cfg.cast_half_period:
            self._cast_phase = 0
            self._cast_dir *= -1.0
        head_omega = float(np.clip(self._cast_dir * cfg.cast_head_omega, -1.0, 1.0))
        body_omega = float(np.clip(self._cast_dir * cfg.cast_body_omega, -1.0, 1.0))
        v_action = float(np.clip(2.0 * cfg.cast_creep_speed - 1.0, -1.0, 1.0))
        return np.asarray([v_action, body_omega, head_omega], dtype=np.float32)


def run_episode(env, controller: BilateralChemotaxis, seed: int,
                collect_traj: bool = False, render_fn=None):
    """Roll out one episode of a chemotaxis controller on ``OslEnv``.

    Returns a summary dict ``{seed, return, success, casts, steps[, frames]}``.
    ``render_fn`` (optional) is ``render_rollout_frame`` so the notebook can
    build a GIF with the same look as the policy notebooks.
    """
    obs, _ = env.reset(seed=seed)
    controller.reset()
    ret, casts, success = 0.0, 0, False
    traj_x, traj_y, cast_x, cast_y, frames = [], [], [], [], []
    for t in range(env.max_steps):
        action = controller.act(obs)
        if collect_traj:
            traj_x.append(env.x_mm); traj_y.append(env.y_mm)
        obs, r, term, trunc, info = env.step(action)
        ret += float(r)
        if info.get("event_is_high_cast_like"):
            casts += 1
            if collect_traj:
                cast_x.append(env.x_mm); cast_y.append(env.y_mm)
        if collect_traj and render_fn is not None:
            frames.append(render_fn(env, traj_x, traj_y, cast_x, cast_y, t,
                                    title=f"chemotaxis seed={seed} step={t} casts={casts}"))
        if term or trunc:
            success = bool(info.get("success", False))
            break
    return {
        "seed": seed, "return": ret, "success": success,
        "casts": casts, "steps": t + 1,
        "frames": frames if collect_traj else None,
    }
