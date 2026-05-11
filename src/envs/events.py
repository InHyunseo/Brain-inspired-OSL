from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EventFlags:
    is_run: bool
    is_stop: bool
    is_low_sweep: bool
    is_high_cast_like: bool
    is_turn_like: bool
    is_spin_like: bool


def classify_event(
    speed_mm_s: float,
    omega_rad_s: float,
    sampling_angle_rad: float,
    sampling_duration_s: float,
    run_threshold_mm_s: float,
    stop_threshold_mm_s: float,
    max_sampling_duration_s: float,
) -> EventFlags:
    deg = abs(math.degrees(sampling_angle_rad))
    omega_deg_s = abs(math.degrees(omega_rad_s))
    is_run = speed_mm_s > run_threshold_mm_s and omega_deg_s < 12.0
    is_stop = speed_mm_s < stop_threshold_mm_s
    is_low_sweep = is_stop and 10.0 <= deg < 37.0
    is_high_cast_like = is_stop and 37.0 <= deg <= 120.0
    is_turn_like = omega_deg_s > 12.0 or deg >= 20.0
    is_spin_like = deg > 120.0 or sampling_duration_s > max_sampling_duration_s
    return EventFlags(is_run, is_stop, is_low_sweep, is_high_cast_like, is_turn_like, is_spin_like)
