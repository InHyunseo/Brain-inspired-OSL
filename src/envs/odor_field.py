from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _box_blur2d(grid: np.ndarray) -> np.ndarray:
    padded = np.pad(grid, ((1, 1), (1, 1)), mode="edge")
    out = np.zeros_like(grid, dtype=np.float64)
    for dy in range(3):
        for dx in range(3):
            out += padded[dy : dy + grid.shape[0], dx : dx + grid.shape[1]]
    return out / 9.0


@dataclass
class GaussianOdorField:
    source_x_mm: float
    source_y_mm: float
    sigma_mm: float = 30.0
    c_peak: float = 1.0
    c_background: float = 0.0
    epsilon: float = 1e-8
    noise_stage: int = 0
    noise_strength: float = 0.0
    noise_rho: float = 0.94
    arena_width_mm: float = 80.0
    arena_height_mm: float = 120.0
    noise_grid_spacing_mm: float = 0.05
    rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        self.rng = self.rng or np.random.default_rng(0)
        self._x_coords = np.arange(0.0, self.arena_width_mm + self.noise_grid_spacing_mm, self.noise_grid_spacing_mm, dtype=np.float64)
        self._y_coords = np.arange(0.0, self.arena_height_mm + self.noise_grid_spacing_mm, self.noise_grid_spacing_mm, dtype=np.float64)
        self._noise_grid = np.ones((len(self._y_coords), len(self._x_coords)), dtype=np.float64)
        self.rebuild_noise_grid(initial=True)

    def _base(self, x_mm: float, y_mm: float) -> float:
        dx = x_mm - self.source_x_mm
        dy = y_mm - self.source_y_mm
        return float(self.c_background + self.c_peak * np.exp(-(dx * dx + dy * dy) / (2.0 * self.sigma_mm**2)))

    def rebuild_noise_grid(self, initial: bool = False) -> None:
        if self.noise_stage == 0 or self.noise_strength <= 0.0:
            self._noise_grid.fill(1.0)
            return
        if self.noise_stage == 1:
            self._noise_grid = np.clip(self.rng.normal(1.0, self.noise_strength, size=self._noise_grid.shape), 0.0, None)
            return
        white = self.rng.normal(0.0, self.noise_strength, size=self._noise_grid.shape)
        blurred = _box_blur2d(white)
        if initial:
            candidate = blurred
        else:
            candidate = self.noise_rho * (self._noise_grid - 1.0) + np.sqrt(max(0.0, 1.0 - self.noise_rho**2)) * blurred
        self._noise_grid = np.clip(1.0 + candidate, 0.0, None)

    def advance(self) -> None:
        self.rebuild_noise_grid(initial=False)

    def _interpolate_noise(self, x_mm: float, y_mm: float) -> float:
        x = float(np.clip(x_mm, 0.0, self.arena_width_mm))
        y = float(np.clip(y_mm, 0.0, self.arena_height_mm))
        gx = x / self.noise_grid_spacing_mm
        gy = y / self.noise_grid_spacing_mm
        x0 = int(np.floor(gx))
        y0 = int(np.floor(gy))
        x1 = min(x0 + 1, self._noise_grid.shape[1] - 1)
        y1 = min(y0 + 1, self._noise_grid.shape[0] - 1)
        tx = gx - x0
        ty = gy - y0
        q00 = self._noise_grid[y0, x0]
        q10 = self._noise_grid[y0, x1]
        q01 = self._noise_grid[y1, x0]
        q11 = self._noise_grid[y1, x1]
        top = (1.0 - tx) * q00 + tx * q10
        bottom = (1.0 - tx) * q01 + tx * q11
        return float((1.0 - ty) * top + ty * bottom)

    def sample(self, x_mm: float, y_mm: float) -> float:
        base = self._base(x_mm, y_mm)
        if self.noise_stage == 0 or self.noise_strength <= 0.0:
            return base
        return float(max(0.0, base * self._interpolate_noise(x_mm, y_mm)))

    def gradient(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        dx = x_mm - self.source_x_mm
        dy = y_mm - self.source_y_mm
        base = self._base(x_mm, y_mm) - self.c_background
        coeff = -1.0 / (self.sigma_mm**2)
        return float(coeff * dx * base), float(coeff * dy * base)
