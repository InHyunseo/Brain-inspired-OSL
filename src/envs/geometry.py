from __future__ import annotations

import math


def wrap_angle(theta: float) -> float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def sensor_positions(x_mm: float, y_mm: float, heading_rad: float, spacing_mm: float) -> dict[str, tuple[float, float]]:
    offset = spacing_mm / 2.0
    left_dx = -math.sin(heading_rad) * offset
    left_dy = math.cos(heading_rad) * offset
    return {
        "left": (x_mm + left_dx, y_mm + left_dy),
        "right": (x_mm - left_dx, y_mm - left_dy),
    }
