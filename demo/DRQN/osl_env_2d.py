"""2D OSL environments — Static (Gaussian plume) and Dynamic (turbulent plume).

Ported directly from ipynb/PPO_framework.ipynb. Both envs share the same
[c, did_cast] observation and Box([v, omega, cast]) action, so the same agent
code can drive either env.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.ndimage import gaussian_filter


class StaticEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, L=3.0, dt=0.1, src_x=0.0, src_y=0.0, sigma_c=1.0, r_goal=0.35):
        super().__init__()
        self.L, self.dt, self.ds = float(L), float(dt), 0.08
        self.src_x, self.src_y = float(src_x), float(src_y)
        self.sigma_c, self.r_goal = float(sigma_c), float(r_goal)

        self.v_min, self.v_max, self.omega_max = 0.2, 0.45, 5.0
        self.max_steps, self.reward_scale = 300, 0.5
        self.b_hold, self.b_oob = 0.5, 5.0
        self.goal_hold_steps = 20
        self.cast_penalty = 0.002
        self.bg_c = 0.0

        self.wind_x, self.wind_y = 0.0, 0.0
        self._wind_mag = float(np.hypot(self.wind_x, self.wind_y))
        self._wind_dir = (
            (self.wind_x / self._wind_mag, self.wind_y / self._wind_mag)
            if self._wind_mag > 1e-6 else (0.0, 0.0)
        )

        self.action_space = spaces.Box(
            low=np.array([self.v_min, -self.omega_max, 0.0], dtype=np.float32),
            high=np.array([self.v_max, self.omega_max, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        self._scan_seq = [0, 1, 0, 1]
        self._scan_dirs = [np.pi / 2, -np.pi / 2]
        self.np_random = np.random.default_rng(None)

    def _conc(self, x, y):
        x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
        xr, yr = x - np.float32(self.src_x), y - np.float32(self.src_y)

        if self._wind_mag <= 1e-6:
            r2 = xr * xr + yr * yr
            c = np.exp(-r2 / (2.0 * self.sigma_c * self.sigma_c))
        else:
            wx, wy = self._wind_dir
            t, s = xr * wx + yr * wy, -xr * wy + yr * wx
            stretch = 1.0 + min(self._wind_mag, 2.0)
            sigma_s, sigma_t, sigma_up = self.sigma_c, self.sigma_c * stretch, self.sigma_c / stretch
            t_pos, t_neg = np.maximum(0.0, t), np.maximum(0.0, -t)
            c = np.exp(-(
                (s * s) / (2.0 * sigma_s ** 2)
                + (t_pos ** 2) / (2.0 * sigma_t ** 2)
                + (t_neg ** 2) / (2.0 * sigma_up ** 2)
            ))

        c = self.bg_c + (1.0 - self.bg_c) * c
        return float(np.clip(c, 0.0, 1.0))

    def _sense(self, phi=0.0):
        ang = self.th + phi
        sx = self.x + np.cos(ang) * self.ds
        sy = self.y + np.sin(ang) * self.ds
        c = self._conc(sx, sy) + self.np_random.normal(0, 0.01)
        return float(np.clip(c, 0.0, 1.0))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        r0 = float(self.np_random.uniform(
            max(self.r_goal + 0.25, 0.6), min(0.8 * self.L, self.L - 0.25)
        ))
        ang = float(self.np_random.uniform(-np.pi, np.pi))
        self.x = self.src_x + r0 * np.cos(ang)
        self.y = self.src_y + r0 * np.sin(ang)

        heading_center = float(np.arctan2(self.src_y - self.y, self.src_x - self.x))
        self.th = (heading_center + self.np_random.uniform(-np.pi / 3, np.pi / 3) + np.pi) % (2 * np.pi) - np.pi

        self.v, self.omega, self._step = 0.0, 0.0, 0
        self.in_cast, self.cast_phase = False, 0
        self.goal_hold_count, self.cast_count = 0, 0
        self._trail = [(self.x, self.y)]

        return np.array([self._sense(0.0), 0.0], dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        v_cmd = float(np.clip(action[0], self.v_min, self.v_max))
        omega_cmd = float(np.clip(action[1], -self.omega_max, self.omega_max))
        cast_cmd = int(np.rint(np.clip(action[2], 0.0, 1.0)))

        did_cast = False

        if self.in_cast:
            phi = self._scan_dirs[self._scan_seq[self.cast_phase]]
            obs_c = self._sense(phi)
            self.v, self.omega = 0.0, 0.0
            self.cast_phase += 1
            if self.cast_phase >= 4:
                self.in_cast = False
            did_cast = True
        elif cast_cmd == 1:
            self.in_cast = True
            self.cast_phase = 0
            self.cast_count += 1
            obs_c = self._sense(self._scan_dirs[self._scan_seq[self.cast_phase]])
            self.v, self.omega = 0.0, 0.0
            self.cast_phase += 1
            did_cast = True
        else:
            self.v, self.omega = v_cmd, omega_cmd
            self.th = (self.th + self.omega * self.dt + np.pi) % (2 * np.pi) - np.pi
            self.x += self.v * np.cos(self.th) * self.dt
            self.y += self.v * np.sin(self.th) * self.dt
            obs_c = self._sense(0.0)

        d = np.hypot(self.x - self.src_x, self.y - self.src_y)
        in_goal = (d < self.r_goal)
        self.goal_hold_count = self.goal_hold_count + 1 if in_goal else 0
        success_hold = (self.goal_hold_count >= self.goal_hold_steps)
        oob = (abs(self.x) > self.L) or (abs(self.y) > self.L)

        r = self.reward_scale * obs_c
        if did_cast:
            r -= self.cast_penalty
        else:
            r -= 0.01 * (abs(self.v) / self.v_max) + 0.01 * (abs(self.omega) / self.omega_max)
        if in_goal:
            r += self.b_hold
        if oob:
            r -= self.b_oob

        terminated = bool(oob or success_hold)
        truncated = bool(self._step >= self.max_steps)
        self._trail.append((self.x, self.y))

        info = {"is_success": success_hold}
        if terminated or truncated:
            info["cast_ratio"] = self.cast_count / self._step if self._step > 0 else 0.0
            info["casts"] = self.cast_count

        return np.array([obs_c, float(did_cast)], dtype=np.float32), float(r), terminated, truncated, info


class DynamicEnv(StaticEnv):
    """Turbulent plume env. noise_coef scales all dynamic-irregularity params."""

    def __init__(self, noise_coef=1.0, **kwargs):
        super().__init__(**kwargs)
        self.noise_coef = float(noise_coef)
        self.field_size = 100
        self.floor_value = 0.03
        self.non_source_cap = 0.55
        self.bump_count = 10
        self.bump_sigma_min = 4.0
        self.bump_sigma_max = 12.0
        self.blur_sigma = 5.0
        self.noise_scale = 1.0
        self.noise_bias = 0.0
        self.drift_damping = 0.92

        # All dynamic-noise params scaled by noise_coef:
        # smaller coef -> lower amplitude / slower drift / less phase jitter.
        self.bump_amp_min = 0.04 * self.noise_coef
        self.bump_amp_max = 0.16 * self.noise_coef
        self.drift_jitter = 0.18 * self.noise_coef
        self.phase_jitter = 0.05 * self.noise_coef
        self.phase_speed_min = 0.06 * self.noise_coef
        self.phase_speed_max = 0.16 * self.noise_coef

        self._grid_x, self._grid_y = None, None
        self._base_field = None
        self._field_view = None

    def _grid_xy(self):
        if self._grid_x is None or self._grid_y is None:
            xs = np.arange(self.field_size, dtype=np.float32)
            ys = np.arange(self.field_size, dtype=np.float32)
            self._grid_x, self._grid_y = np.meshgrid(xs, ys)
        return self._grid_x, self._grid_y

    def _build_base_field(self):
        X, Y = self._grid_xy()
        world_x = (X / (self.field_size - 1)) * (2.0 * self.L) - self.L
        world_y = (Y / (self.field_size - 1)) * (2.0 * self.L) - self.L
        xr, yr = world_x - self.src_x, world_y - self.src_y

        if self._wind_mag <= 1e-6:
            r2 = xr * xr + yr * yr
            base = np.exp(-r2 / (2.0 * self.sigma_c * self.sigma_c))
        else:
            wx, wy = self._wind_dir
            t, s = xr * wx + yr * wy, -xr * wy + yr * wx
            stretch = 1.0 + min(self._wind_mag, 2.0)
            sigma_s, sigma_t, sigma_up = self.sigma_c, self.sigma_c * stretch, self.sigma_c / stretch
            t_pos, t_neg = np.maximum(0.0, t), np.maximum(0.0, -t)
            base = np.exp(-(
                (s * s) / (2.0 * sigma_s ** 2)
                + (t_pos ** 2) / (2.0 * sigma_t ** 2)
                + (t_neg ** 2) / (2.0 * sigma_up ** 2)
            ))

        base = self.floor_value + (self.non_source_cap - self.floor_value) * base
        return base.astype(np.float32)

    def _init_dynamic_bumps(self):
        count = self.bump_count
        self._bump_x = self.np_random.uniform(0.0, self.field_size - 1, size=count).astype(np.float32)
        self._bump_y = self.np_random.uniform(0.0, self.field_size - 1, size=count).astype(np.float32)
        self._bump_vx = self.np_random.normal(0.0, self.drift_jitter, size=count).astype(np.float32)
        self._bump_vy = self.np_random.normal(0.0, self.drift_jitter, size=count).astype(np.float32)
        self._bump_sigma = self.np_random.uniform(self.bump_sigma_min, self.bump_sigma_max, size=count).astype(np.float32)
        self._bump_amp = self.np_random.uniform(self.bump_amp_min, self.bump_amp_max, size=count).astype(np.float32)
        self._bump_phase = self.np_random.uniform(-np.pi, np.pi, size=count).astype(np.float32)
        self._bump_phase_speed = self.np_random.uniform(
            self.phase_speed_min, self.phase_speed_max, size=count
        ).astype(np.float32)

    def _advance_dynamic_bumps(self):
        self._bump_phase += self._bump_phase_speed
        if self.phase_jitter > 0.0:
            self._bump_phase += self.np_random.normal(0.0, self.phase_jitter, size=self.bump_count).astype(np.float32)

        if self.drift_jitter > 0.0:
            self._bump_vx = (self.drift_damping * self._bump_vx
                             + self.np_random.normal(0.0, self.drift_jitter, size=self.bump_count).astype(np.float32))
            self._bump_vy = (self.drift_damping * self._bump_vy
                             + self.np_random.normal(0.0, self.drift_jitter, size=self.bump_count).astype(np.float32))

        self._bump_x += self._bump_vx
        self._bump_y += self._bump_vy

        lo, hi = 0.0, float(self.field_size - 1)
        hit_lo_x, hit_hi_x = self._bump_x < lo, self._bump_x > hi
        hit_lo_y, hit_hi_y = self._bump_y < lo, self._bump_y > hi

        self._bump_x = np.clip(self._bump_x, lo, hi)
        self._bump_y = np.clip(self._bump_y, lo, hi)
        self._bump_vx[hit_lo_x | hit_hi_x] *= -1.0
        self._bump_vy[hit_lo_y | hit_hi_y] *= -1.0

    def _build_dynamic_noise(self):
        X, Y = self._grid_xy()
        noise = np.zeros((self.field_size, self.field_size), dtype=np.float32)
        if self.bump_count <= 0:
            return noise

        for i in range(self.bump_count):
            amp = self._bump_amp[i] * self.noise_scale * (np.sin(self._bump_phase[i]) + self.noise_bias)
            if abs(float(amp)) < 1e-6:
                continue
            sigma = max(float(self._bump_sigma[i]), 1e-3)
            g = np.exp(-((X - self._bump_x[i]) ** 2 + (Y - self._bump_y[i]) ** 2)
                       / (2.0 * sigma * sigma)).astype(np.float32)
            noise += (float(amp) * g).astype(np.float32)

        if self.blur_sigma > 0.0:
            noise = gaussian_filter(noise, sigma=self.blur_sigma).astype(np.float32)
        return noise

    def _compose_field_view(self):
        field = self._base_field + self._build_dynamic_noise()
        self._field_view = np.clip(field, self.floor_value, 1.0).astype(np.float32)

    def _world_to_grid(self, x, y):
        x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
        gx = (x + self.L) / (2.0 * self.L) * (self.field_size - 1)
        gy = (y + self.L) / (2.0 * self.L) * (self.field_size - 1)
        return np.clip(gx, 0.0, self.field_size - 1), np.clip(gy, 0.0, self.field_size - 1)

    def _bilinear_sample(self, field, x, y):
        gx, gy = self._world_to_grid(x, y)
        x0, y0 = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
        x1, y1 = np.clip(x0 + 1, 0, self.field_size - 1), np.clip(y0 + 1, 0, self.field_size - 1)
        wx, wy = gx - x0.astype(np.float32), gy - y0.astype(np.float32)
        v00, v10, v01, v11 = field[y0, x0], field[y0, x1], field[y1, x0], field[y1, x1]
        return ((1.0 - wx) * (1.0 - wy) * v00 + wx * (1.0 - wy) * v10
                + (1.0 - wx) * wy * v01 + wx * wy * v11)

    def _ensure_field(self):
        if self._field_view is None:
            self._base_field = self._build_base_field()
            self._init_dynamic_bumps()
            self._compose_field_view()

    def _conc(self, x, y):
        self._ensure_field()
        return float(np.clip(self._bilinear_sample(self._field_view, x, y), 0.0, 1.0))

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self._base_field = self._build_base_field()
        self._init_dynamic_bumps()
        self._compose_field_view()
        return super().reset(seed=None, options=options)

    def step(self, action):
        self._advance_dynamic_bumps()
        self._compose_field_view()
        return super().step(action)
