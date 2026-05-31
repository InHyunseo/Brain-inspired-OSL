from __future__ import annotations

import math


def wrap_angle(theta: float) -> float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def sensor_positions(
    x_mm: float,
    y_mm: float,
    heading_rad: float,
    spacing_mm: float,
    forward_mm: float = 0.0,
) -> dict[str, tuple[float, float]]:
    """Bilateral sensor coordinates.

    Sensors sit ``forward_mm`` ahead of the body point along ``heading_rad``
    (the head tip), then split laterally by ``spacing_mm``. ``heading_rad`` is
    the sensor heading (body heading + head-relative angle), so when the agent
    casts (sweeps its head), the head tip swings on an arc of radius
    ``forward_mm`` — making the head direction actually change what the sensors
    read. With ``forward_mm=0`` this reduces to the legacy body-centered pair.
    """
    half = spacing_mm / 2.0
    # Head tip: move forward along the sensor heading.
    hx = x_mm + math.cos(heading_rad) * forward_mm
    hy = y_mm + math.sin(heading_rad) * forward_mm
    # Lateral split (perpendicular to heading).
    left_dx = -math.sin(heading_rad) * half
    left_dy = math.cos(heading_rad) * half
    return {
        "left": (hx + left_dx, hy + left_dy),
        "right": (hx - left_dx, hy - left_dy),
    }
