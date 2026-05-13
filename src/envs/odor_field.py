from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import math
import numpy as np


# ---- Bump-field noise model -------------------------------------------------
#
# The plume perturbation is a sum of independent local Gaussian bumps with
# signed amplitudes. The base Gaussian plume is multiplied by
#     1 + alpha * Σ_i  A_i(t) * exp(-r_i^2 / (2 sigma_i^2))
# clamped at 0. Each bump independently drifts, slowly oscillates its amplitude
# via an AR(1) process, and is occasionally respawned. A single curriculum
# scalar `alpha` ∈ [0, 1] scales every bump-field parameter so that one knob
# moves from "weak static perturbation" to "full hydrodynamic-like turbulence".
#
# Per-bump sigma is capped by `sigma_max_frac_of_dist_to_source * d_to_source`
# so that bumps near the source stay small. This preserves the broad shape of
# the source peak (the chemotaxis gradient does not collapse) while still
# making distal regions structurally noisy.

_DEFAULT_BUMP_RNG_SEED = 12345


@dataclass
class BumpFieldConfig:
    """Curriculum-controlled bump-field parameters.

    `alpha` is the curriculum scalar in [0, 1]. All `*_max` values are scaled
    linearly by `alpha`. Pass alpha=0 to disable noise entirely.
    """
    alpha: float = 0.0
    dynamic: bool = True  # if False, bumps are frozen after init (static phase)

    # Maximum number of simultaneous bumps at alpha=1.
    n_bumps_max: int = 40
    # Bump amplitude range. Sign is randomized on spawn; absolute value clipped
    # to amp_max_abs * alpha at runtime.
    amp_max_abs: float = 0.8
    # Bump radius range (mm). Independent of alpha — the spatial scale of an
    # individual eddy doesn't grow with curriculum strength. Capped per-bump
    # by sigma_max_frac_of_dist_to_source * distance_to_source.
    sigma_min_mm: float = 2.0
    sigma_max_mm: float = 12.0
    sigma_max_frac_of_dist_to_source: float = 0.5
    # AR(1) lifecycle for amplitude. Lower lifetime_inv = slower oscillation.
    # Capped from below to keep even alpha=1 from being too fast.
    lifetime_inv_max: float = 0.06  # per step, at alpha=1
    lifetime_inv_floor: float = 0.005
    # Per-step probability that a bump is fully respawned (jumps to new
    # position / radius / sign). Scaled by alpha.
    respawn_prob_max: float = 0.01
    # Bump drift velocity (mm/step). Random isotropic direction, magnitude
    # uniform in [0, drift_speed_max * alpha].
    drift_speed_max_mm_per_step: float = 0.4


@dataclass
class GaussianOdorField:
    source_x_mm: float
    source_y_mm: float
    sigma_mm: float = 30.0
    c_peak: float = 1.0
    c_background: float = 0.0
    epsilon: float = 1e-8
    # Stage semantics:
    #   0 = clean Gaussian, no perturbation.
    #   1 = static bump field (no advance() effect after reset).
    #   2 = dynamic bump field (advance() drifts + ages bumps).
    noise_stage: int = 0
    # Curriculum scalar in [0, 1]. Scales every bump-field parameter.
    noise_strength: float = 0.0
    arena_width_mm: float = 80.0
    arena_height_mm: float = 120.0
    bump_cfg: BumpFieldConfig = field(default_factory=BumpFieldConfig)
    rng: Optional[np.random.Generator] = None

    def __post_init__(self) -> None:
        self.rng = self.rng or np.random.default_rng(_DEFAULT_BUMP_RNG_SEED)
        # Bump state arrays — allocated to max capacity, active count varies.
        N = self.bump_cfg.n_bumps_max
        self._bx = np.zeros(N, dtype=np.float64)
        self._by = np.zeros(N, dtype=np.float64)
        self._bs = np.ones(N, dtype=np.float64)         # sigma_mm
        self._ba = np.zeros(N, dtype=np.float64)        # amplitude
        self._bvx = np.zeros(N, dtype=np.float64)       # drift velocity
        self._bvy = np.zeros(N, dtype=np.float64)
        self._b_active = np.zeros(N, dtype=bool)
        self._init_bumps()

    # ---- helpers -----------------------------------------------------------
    def _effective_alpha(self) -> float:
        if self.noise_stage == 0:
            return 0.0
        return float(max(0.0, min(1.0, self.noise_strength)))

    def _n_active_target(self, alpha: float) -> int:
        return int(round(self.bump_cfg.n_bumps_max * alpha))

    def _base(self, x_mm: float, y_mm: float) -> float:
        dx = x_mm - self.source_x_mm
        dy = y_mm - self.source_y_mm
        return float(self.c_background + self.c_peak * np.exp(
            -(dx * dx + dy * dy) / (2.0 * self.sigma_mm ** 2)))

    def _sigma_cap_for(self, x: float, y: float) -> float:
        d = math.hypot(x - self.source_x_mm, y - self.source_y_mm)
        return max(self.bump_cfg.sigma_min_mm,
                   self.bump_cfg.sigma_max_frac_of_dist_to_source * d)

    def _spawn_one(self, i: int, alpha: float) -> None:
        x = float(self.rng.uniform(0.0, self.arena_width_mm))
        y = float(self.rng.uniform(0.0, self.arena_height_mm))
        sigma_cap = min(self.bump_cfg.sigma_max_mm, self._sigma_cap_for(x, y))
        sigma_lo = self.bump_cfg.sigma_min_mm
        if sigma_cap <= sigma_lo:
            sigma = sigma_lo
        else:
            sigma = float(self.rng.uniform(sigma_lo, sigma_cap))
        sign = 1.0 if self.rng.random() < 0.5 else -1.0
        amp_max = self.bump_cfg.amp_max_abs * alpha
        amp = sign * float(self.rng.uniform(0.0, amp_max))
        speed = float(self.rng.uniform(0.0, self.bump_cfg.drift_speed_max_mm_per_step * alpha))
        theta = float(self.rng.uniform(-math.pi, math.pi))
        self._bx[i] = x
        self._by[i] = y
        self._bs[i] = sigma
        self._ba[i] = amp
        self._bvx[i] = speed * math.cos(theta)
        self._bvy[i] = speed * math.sin(theta)
        self._b_active[i] = True

    def _init_bumps(self) -> None:
        alpha = self._effective_alpha()
        n_active = self._n_active_target(alpha)
        N = self.bump_cfg.n_bumps_max
        self._b_active[:] = False
        self._ba[:] = 0.0
        if alpha <= 0.0 or n_active <= 0:
            return
        for i in range(n_active):
            self._spawn_one(i, alpha)

    # ---- public API --------------------------------------------------------
    def rebuild_noise_grid(self, initial: bool = False) -> None:
        """Compatibility shim — re-seeds bumps from scratch.

        Kept for `OslEnv.reset()` / `set_noise_stage()` which both call this.
        """
        self._init_bumps()

    def advance(self) -> None:
        """Advance one env step: AR(1) amplitude, drift, occasional respawn."""
        if self.noise_stage <= 1:
            return
        alpha = self._effective_alpha()
        if alpha <= 0.0:
            return
        cfg = self.bump_cfg
        # AR(1) amplitude update.
        lifetime_inv = max(cfg.lifetime_inv_floor, cfg.lifetime_inv_max * alpha)
        rho = math.sqrt(max(0.0, 1.0 - lifetime_inv))
        innovation_std = math.sqrt(max(0.0, 1.0 - rho * rho)) * cfg.amp_max_abs * alpha * 0.5
        active_idx = np.flatnonzero(self._b_active)
        if active_idx.size > 0:
            noise = self.rng.normal(0.0, innovation_std, size=active_idx.size)
            self._ba[active_idx] = rho * self._ba[active_idx] + noise
            amp_clip = cfg.amp_max_abs * alpha
            np.clip(self._ba[active_idx], -amp_clip, amp_clip, out=self._ba[active_idx])
            # Drift.
            self._bx[active_idx] += self._bvx[active_idx]
            self._by[active_idx] += self._bvy[active_idx]
            # Wrap-around to keep bumps inside the arena.
            self._bx[active_idx] = np.mod(self._bx[active_idx], self.arena_width_mm)
            self._by[active_idx] = np.mod(self._by[active_idx], self.arena_height_mm)
            # Random respawn.
            resp_p = cfg.respawn_prob_max * alpha
            if resp_p > 0.0:
                draws = self.rng.random(active_idx.size)
                for k, idx in enumerate(active_idx):
                    if draws[k] < resp_p:
                        self._spawn_one(int(idx), alpha)

    def _perturbation(self, x_mm: float, y_mm: float) -> float:
        if self.noise_stage == 0:
            return 0.0
        active_idx = np.flatnonzero(self._b_active)
        if active_idx.size == 0:
            return 0.0
        dx = x_mm - self._bx[active_idx]
        dy = y_mm - self._by[active_idx]
        s2 = self._bs[active_idx] ** 2
        contrib = self._ba[active_idx] * np.exp(-(dx * dx + dy * dy) / (2.0 * s2))
        return float(contrib.sum())

    def sample(self, x_mm: float, y_mm: float) -> float:
        base = self._base(x_mm, y_mm)
        if self.noise_stage == 0:
            return base
        pert = self._perturbation(x_mm, y_mm)
        return float(max(0.0, base * (1.0 + pert)))

    def gradient(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        # Analytic gradient is computed from the base Gaussian only; the noise
        # is treated as multiplicative perturbation and not differentiated for
        # the chemotaxis shaping term.
        dx = x_mm - self.source_x_mm
        dy = y_mm - self.source_y_mm
        base = self._base(x_mm, y_mm) - self.c_background
        coeff = -1.0 / (self.sigma_mm ** 2)
        return float(coeff * dx * base), float(coeff * dy * base)
