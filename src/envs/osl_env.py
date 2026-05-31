from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.envs.events import classify_event
from src.envs.geometry import sensor_positions, wrap_angle
from src.envs.odor_field import GaussianOdorField


@dataclass
class EnvConfig:
    body_length_mm: float = 3.5
    sensor_spacing_mm: float = 0.15
    # Sensors sit this far ahead of the body point (head tip), along the sensor
    # heading. With spacing (0.15mm) << plume sigma (30mm) the instantaneous
    # left/right difference is tiny, so head casting barely changes the reading
    # when sensors are body-centered (forward=0). Moving them forward makes the
    # head tip swing on an arc when casting, so c_avg varies across a sweep
    # — i.e. casting yields real gradient information the policy can exploit.
    # 1.75 = body_length/2: sensors sit at the head tip if (x, y) is the body
    # centroid (the dorsal organ of a larva is at the anterior end).
    sensor_forward_mm: float = 1.75
    dt: float = 0.1
    episode_seconds: float = 60.0
    arena_width_mm: float = 80.0
    arena_height_mm: float = 120.0
    source_x_mm: float = 40.0
    source_y_mm: float = 100.0
    initial_x_mm: float = 40.0
    initial_y_mm: float = 20.0
    initial_heading_rad: float = math.pi / 2.0
    randomize_heading: bool = True
    success_radius_mm: float = 7.5
    # Spawn policy: sample (x, y) uniformly inside an annulus around the source.
    # The annulus is the intersection of three constraints:
    #   1) base Gaussian field exceeds `spawn_c_thresh_frac * c_peak` (cue zone),
    #   2) distance to source ≥ `spawn_min_radius_mm` (no spawning inside or
    #      near the success region — must be well outside it),
    #   3) distance to source ≤ `spawn_max_radius_mm` (and inside the cue zone).
    # Heading is initialised toward the source with Gaussian error of std
    # `spawn_heading_std_rad`.
    randomize_spawn: bool = True
    spawn_c_thresh_frac: float = 0.05
    spawn_min_radius_mm: float = 55.0
    spawn_max_radius_mm: float = 70.0
    spawn_max_tries: int = 400
    spawn_heading_std_rad: float = math.radians(30.0)
    terminate_on_wall: bool = True
    wall_penalty: float = -2.0
    gaussian_sigma_mm: float = 30.0
    c_peak: float = 1.0
    c_background: float = 0.0
    epsilon: float = 1e-8
    noise_stage: int = 0
    noise_strength: float = 0.0
    v_max_mm_s: float = 1.2
    body_omega_max_deg_s: float = 120.0
    head_omega_max_deg_s: float = 240.0
    initial_head_relative_angle_rad: float = 0.0
    stop_threshold_mm_s: float = 0.08
    run_threshold_mm_s: float = 0.2
    # ---- Reward terms (biologically structured energy budget) ----
    # Food (sparse, terminal): must dominate the cumulative energy cost so the
    # critic always sees a clearly-positive return for successful trajectories.
    reward_goal: float = 20.0
    # Shaping: rate of log-concentration change ("chemotaxis information").
    # Clipped to ±reward_log_clip to keep dlog/dt from exploding near c=0.
    # This is the ONLY directional signal ("am I heading up-gradient?"). Raised
    # 0.05 -> 0.15 because at 0.05 the per-step dlog reward was tiny and averaged
    # to ~0 over an episode, giving the policy almost no gradient to learn "go
    # toward the source" — dist stayed pinned near the spawn radius.
    reward_log_k: float = 0.15
    reward_log_clip: float = 0.5
    # Dense concentration reward: proportional to current normalized sensor
    # average c_avg / c_peak. Provides a smooth potential field toward the
    # source so the agent always knows which way is "warmer", independent of
    # the (noisy, clipped) dlog/dt term. Keep small so the cumulative dense
    # reward stays below reward_goal over an episode.
    reward_conc_k: float = 0.02
    # Basal metabolism: alive-cost per step.
    reward_time_penalty: float = -0.005
    # Movement energy ∝ v² / ω², per step (normalized actions in [-1, 1]).
    # head_omega cost is highest because head sweeping (cast-like) is the most
    # energetically expensive emergent behaviour for the larva.
    reward_run_cost: float = -0.01      # coefficient on (v_norm)^2
    reward_body_turn_cost: float = -0.005   # coefficient on (body_omega_norm)^2
    # head_omega cost lowered to run-cost level (was -0.02). With sensors at the
    # head tip (sensor_forward_mm) a cast yields real gradient info (dlog reward
    # ~+0.035/step when it finds the warmer side), so the cost must not exceed
    # that or the policy never explores casting. -0.01 keeps cast a net positive
    # when it actually informs the next surge.
    reward_head_cast_cost: float = -0.01    # coefficient on (head_omega_norm)^2
    # Multiplier for stopped head-shaking (cast). Set to 1.0 (was 2.0): the old
    # 2× made stopped casting cost -0.04/step, larger than its information gain,
    # which blocked the policy from ever discovering cast during exploration.
    reward_head_cast_stopped_mult: float = 1.0
    # Spin-like behaviour penalty (head sweep > 120° or stopped > 2 s).
    reward_spin_penalty: float = -0.05
    max_sampling_duration_s: float = 2.0
    seed: int = 7

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EnvConfig":
        valid = {field_name for field_name in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in valid})


# Observation: [c_left, c_right, dlog, prev_v_norm, prev_body_omega_norm, prev_head_omega_norm]
# dlog = clipped rate-of-change of log mean concentration ("am I heading up-gradient?").
# Added because the instantaneous left/right difference is ~0.2% (sensor spacing
# 0.15mm << plume sigma 30mm), too weak for the policy to learn a heading from;
# exposing the temporal gradient directly is the biologically-plausible fix
# (larva neurons compute concentration change over time) that keeps the sensor
# physics untouched.
OBS_DIM = 6
# Action: [v, body_omega, head_omega] in [-1, 1]
ACTION_DIM = 3


class OslEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config: EnvConfig | dict[str, Any] | None = None):
        super().__init__()
        if config is None:
            self.cfg = EnvConfig()
        elif isinstance(config, EnvConfig):
            self.cfg = config
        else:
            self.cfg = EnvConfig.from_dict(config)
        self.rng = np.random.default_rng(self.cfg.seed)
        self.field = GaussianOdorField(
            source_x_mm=self.cfg.source_x_mm,
            source_y_mm=self.cfg.source_y_mm,
            sigma_mm=self.cfg.gaussian_sigma_mm,
            c_peak=self.cfg.c_peak,
            c_background=self.cfg.c_background,
            epsilon=self.cfg.epsilon,
            noise_stage=self.cfg.noise_stage,
            noise_strength=self.cfg.noise_strength,
            arena_width_mm=self.cfg.arena_width_mm,
            arena_height_mm=self.cfg.arena_height_mm,
            rng=self.rng,
        )
        self.max_steps = int(round(self.cfg.episode_seconds / self.cfg.dt))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self.reset(seed=self.cfg.seed)

    def set_noise_stage(self, stage: int, strength: float) -> None:
        """Update plume noise schedule (used by curriculum). Rebuilds noise grid in place."""
        self.cfg.noise_stage = int(stage)
        self.cfg.noise_strength = float(strength)
        self.field.noise_stage = int(stage)
        self.field.noise_strength = float(strength)
        self.field.rebuild_noise_grid(initial=True)

    def _sample_spawn_xy(self) -> tuple[float, float]:
        """Rejection-sample (x, y) inside the cue region of the *base* Gaussian.

        We threshold on the noise-free field so the spawnable region is stable
        across curriculum phases. Falls back to the threshold-satisfying point
        farthest from the source if rejection fails.
        """
        c_thr = self.cfg.spawn_c_thresh_frac * self.cfg.c_peak
        r_min = self.cfg.spawn_min_radius_mm
        r_max = self.cfg.spawn_max_radius_mm
        W = self.cfg.arena_width_mm
        H = self.cfg.arena_height_mm
        sx, sy = self.cfg.source_x_mm, self.cfg.source_y_mm
        for _ in range(self.cfg.spawn_max_tries):
            x = float(self.rng.uniform(0.0, W))
            y = float(self.rng.uniform(0.0, H))
            d = math.hypot(x - sx, y - sy)
            if d < r_min or d > r_max:
                continue
            if self.field._base(x, y) >= c_thr:
                return x, y
        # Fallback: ring at r_min along a random angle (still satisfies min radius).
        theta = float(self.rng.uniform(-math.pi, math.pi))
        x = float(np.clip(sx + r_min * math.cos(theta), 0.0, W))
        y = float(np.clip(sy + r_min * math.sin(theta), 0.0, H))
        return x, y

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.field.rng = self.rng
        self.step_count = 0
        if self.cfg.randomize_spawn:
            self.x_mm, self.y_mm = self._sample_spawn_xy()
            dx = self.cfg.source_x_mm - self.x_mm
            dy = self.cfg.source_y_mm - self.y_mm
            heading_to_source = math.atan2(dy, dx)
            err = float(self.rng.normal(0.0, self.cfg.spawn_heading_std_rad))
            self.heading_rad = wrap_angle(heading_to_source + err)
        else:
            self.x_mm = self.cfg.initial_x_mm
            self.y_mm = self.cfg.initial_y_mm
            self.heading_rad = self.cfg.initial_heading_rad
            if self.cfg.randomize_heading:
                self.heading_rad = float(self.rng.uniform(-math.pi, math.pi))
        self.head_relative_angle_rad = self.cfg.initial_head_relative_angle_rad
        self.prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.prev_logs: list[dict[str, float]] = []
        self.accum_sampling_angle_rad = 0.0
        self.sampling_duration_s = 0.0
        self.field.rebuild_noise_grid(initial=True)
        obs, info = self._observe()
        return obs, info

    def _sensor_readings(self) -> dict[str, float]:
        sensor_heading = wrap_angle(self.heading_rad + self.head_relative_angle_rad)
        coords = sensor_positions(
            self.x_mm, self.y_mm, sensor_heading,
            self.cfg.sensor_spacing_mm, self.cfg.sensor_forward_mm,
        )
        left = self.field.sample(*coords["left"])
        right = self.field.sample(*coords["right"])
        return {
            "left": left,
            "right": right,
            "avg": 0.5 * (left + right),
            "sensor_heading_rad": sensor_heading,
        }

    def _observe(self):
        readings = self._sensor_readings()
        self.prev_logs.append({"left": readings["left"], "right": readings["right"], "avg": readings["avg"]})
        if len(self.prev_logs) > 4:
            self.prev_logs.pop(0)

        # Temporal log-concentration gradient (same formula as the reward_dlog
        # shaping term). Gives the policy an explicit "warmer / colder" heading
        # cue that the weak instantaneous left/right difference cannot. Clipped
        # to ±reward_log_clip for the same near-c=0 stability reason; 0 on the
        # first step of an episode (no previous reading yet).
        if len(self.prev_logs) >= 2:
            prev_avg = self.prev_logs[-2]["avg"]
            dlog = (math.log(readings["avg"] + self.cfg.epsilon)
                    - math.log(prev_avg + self.cfg.epsilon)) / self.cfg.dt
        else:
            dlog = 0.0
        clip = float(self.cfg.reward_log_clip)
        if clip > 0.0:
            dlog = max(-clip, min(clip, dlog))

        obs = np.asarray(
            [
                readings["left"],
                readings["right"],
                dlog,
                self.prev_action[0],
                self.prev_action[1],
                self.prev_action[2],
            ],
            dtype=np.float32,
        )

        distance = math.dist((self.x_mm, self.y_mm), (self.cfg.source_x_mm, self.cfg.source_y_mm))
        bearing = wrap_angle(math.atan2(self.cfg.source_y_mm - self.y_mm, self.cfg.source_x_mm - self.x_mm) - self.heading_rad)
        gradient_x, gradient_y = self.field.gradient(self.x_mm, self.y_mm)
        info = {
            "distance_to_source_mm": distance,
            "bearing_to_source_rad": bearing,
            "source_x_mm": self.cfg.source_x_mm,
            "source_y_mm": self.cfg.source_y_mm,
            "gradient_x": gradient_x,
            "gradient_y": gradient_y,
            "sensor_left": readings["left"],
            "sensor_right": readings["right"],
            "sensor_avg": readings["avg"],
            "sensor_heading_rad": readings["sensor_heading_rad"],
            "head_relative_angle_rad": self.head_relative_angle_rad,
        }
        return obs, info

    def step(self, action: np.ndarray):
        raw_v = float(np.clip(action[0], -1.0, 1.0))
        raw_body_omega = float(np.clip(action[1], -1.0, 1.0))
        raw_head_omega = float(np.clip(action[2], -1.0, 1.0))
        v_mm_s = (raw_v + 1.0) * 0.5 * self.cfg.v_max_mm_s
        body_omega_rad_s = math.radians(self.cfg.body_omega_max_deg_s) * raw_body_omega
        head_omega_rad_s = math.radians(self.cfg.head_omega_max_deg_s) * raw_head_omega

        self.x_mm += v_mm_s * math.cos(self.heading_rad) * self.cfg.dt
        self.y_mm += v_mm_s * math.sin(self.heading_rad) * self.cfg.dt
        self.heading_rad = wrap_angle(self.heading_rad + body_omega_rad_s * self.cfg.dt)
        self.head_relative_angle_rad = wrap_angle(self.head_relative_angle_rad + head_omega_rad_s * self.cfg.dt)
        self.prev_action = np.asarray(
            [v_mm_s / self.cfg.v_max_mm_s, raw_body_omega, raw_head_omega], dtype=np.float32
        )
        self.step_count += 1
        if self.cfg.noise_stage >= 2 and self.cfg.noise_strength > 0.0:
            self.field.advance()

        if v_mm_s < self.cfg.stop_threshold_mm_s:
            self.accum_sampling_angle_rad += head_omega_rad_s * self.cfg.dt
            self.sampling_duration_s += self.cfg.dt
        else:
            self.accum_sampling_angle_rad = 0.0
            self.sampling_duration_s = 0.0

        obs, info = self._observe()
        wall = not (0.0 <= self.x_mm <= self.cfg.arena_width_mm and 0.0 <= self.y_mm <= self.cfg.arena_height_mm)
        success = info["distance_to_source_mm"] <= self.cfg.success_radius_mm
        terminated = success or (wall and self.cfg.terminate_on_wall)
        truncated = (not terminated) and self.step_count >= self.max_steps

        reward_goal = self.cfg.reward_goal if success else 0.0

        # Chemotaxis information shaping: rate of log-concentration change.
        # Clipped because dlog/dt can blow up near c≈0 (epsilon-dominated).
        if len(self.prev_logs) >= 2:
            prev_avg = self.prev_logs[-2]["avg"]
            cur_avg = info["sensor_avg"]
            reward_dlog = float(
                (math.log(cur_avg + self.cfg.epsilon) - math.log(prev_avg + self.cfg.epsilon)) / self.cfg.dt
            )
        else:
            reward_dlog = 0.0
        reward_log = self.cfg.reward_log_k * reward_dlog
        if self.cfg.reward_log_clip > 0.0:
            clip = float(self.cfg.reward_log_clip)
            if reward_log > clip:
                reward_log = clip
            elif reward_log < -clip:
                reward_log = -clip

        # Basal metabolism — alive cost per step.
        reward_time = self.cfg.reward_time_penalty

        # Dense concentration reward: normalized current sensor average.
        c_peak = max(self.cfg.c_peak, self.cfg.epsilon)
        reward_conc = self.cfg.reward_conc_k * (float(info["sensor_avg"]) / c_peak)

        # Movement-energy cost (∝ v² / ω²). The head sweep is the most
        # expensive emergent action; doubled when the body is stopped, which
        # is biologically equivalent to a "cast" (head-only swing).
        v_norm = float(self.prev_action[0])              # already in [0, 1]
        body_norm = float(self.prev_action[1])           # in [-1, 1]
        head_norm = float(self.prev_action[2])           # in [-1, 1]
        head_cost_coef = self.cfg.reward_head_cast_cost
        if v_mm_s < self.cfg.stop_threshold_mm_s and abs(head_norm) > 0.05:
            head_cost_coef *= self.cfg.reward_head_cast_stopped_mult
        reward_run = self.cfg.reward_run_cost * (v_norm * v_norm)
        reward_body_turn = self.cfg.reward_body_turn_cost * (body_norm * body_norm)
        reward_head_cast = head_cost_coef * (head_norm * head_norm)
        reward_motion = reward_run + reward_body_turn + reward_head_cast

        reward_wall = self.cfg.wall_penalty if wall else 0.0
        flags = classify_event(
            speed_mm_s=v_mm_s,
            omega_rad_s=body_omega_rad_s,
            sampling_angle_rad=self.accum_sampling_angle_rad,
            sampling_duration_s=self.sampling_duration_s,
            run_threshold_mm_s=self.cfg.run_threshold_mm_s,
            stop_threshold_mm_s=self.cfg.stop_threshold_mm_s,
            max_sampling_duration_s=self.cfg.max_sampling_duration_s,
        )
        reward_spin = self.cfg.reward_spin_penalty if flags.is_spin_like else 0.0
        reward = (
            reward_goal
            + reward_log
            + reward_conc
            + reward_time
            + reward_motion
            + reward_wall
            + reward_spin
        )

        info.update(
            {
                "reward_goal": reward_goal,
                "reward_log": reward_log,
                "reward_conc": reward_conc,
                "reward_time": reward_time,
                "reward_run": reward_run,
                "reward_body_turn": reward_body_turn,
                "reward_head_cast": reward_head_cast,
                "reward_motion": reward_motion,
                "reward_wall": reward_wall,
                "reward_spin": reward_spin,
                "reward_total": reward,
                "success": success,
                "is_success": success,
                "wall_contact": wall,
                "termination_reason": "success"
                if success
                else "wall"
                if wall
                else "time_limit"
                if truncated
                else "running",
                "x_mm": self.x_mm,
                "y_mm": self.y_mm,
                "heading_rad": self.heading_rad,
                "head_relative_angle_rad": self.head_relative_angle_rad,
                "v_mm_s": v_mm_s,
                "body_omega_rad_s": body_omega_rad_s,
                "head_omega_rad_s": head_omega_rad_s,
                "event_is_run": flags.is_run,
                "event_is_stop": flags.is_stop,
                "event_is_low_sweep": flags.is_low_sweep,
                "event_is_high_cast_like": flags.is_high_cast_like,
                "event_is_turn_like": flags.is_turn_like,
                "event_is_spin_like": flags.is_spin_like,
                "step": self.step_count,
                "time_s": self.step_count * self.cfg.dt,
            }
        )
        return obs, float(reward), terminated, truncated, info

    def render(self):
        sensor_heading = wrap_angle(self.heading_rad + self.head_relative_angle_rad)
        return (
            f"OslEnv(x={self.x_mm:.2f}, y={self.y_mm:.2f}, "
            f"heading={self.heading_rad:.2f}, sensor_heading={sensor_heading:.2f})"
        )
