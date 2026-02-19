import numpy as np
import gymnasium as gym
from gymnasium import spaces


class OdorHoldEnvV4(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        L=3.0,
        dt=0.1,
        v_fixed=None,
        src_x=0.0,
        src_y=0.0,
        wind_x=1.0,
        wind_y=0.0,
        sensor_offset=0.08,
        sigma_c=1.0,
        sigma_r=0.8,
        r_goal=0.35,
        b_hold=0.5,
        b_oob=5.0,
        max_steps=300,
        stack_n=1,
        seed=0,
        bg_c=0.0,
        sensor_noise=0.01,
        v_min=0.0,
        v_max=0.45,
        omega_max=2.0,
        accel_max=1.2,
        omega_accel_max=8.0,
        control_penalty=0.01,
        turn_penalty=0.01,
        cast_penalty=0.02,
        turn_requires_cast=True,
        turn_window_steps=1,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.L = float(L)
        self.dt = float(dt)
        self.v_fixed = v_fixed
        self.ds = float(sensor_offset)
        self.src_x = float(np.clip(float(src_x), -self.L, self.L))
        self.src_y = float(np.clip(float(src_y), -self.L, self.L))
        self.wind_x = float(wind_x)
        self.wind_y = float(wind_y)
        self._wind_mag = float(np.hypot(self.wind_x, self.wind_y))

        if self._wind_mag > 1e-6:
            self._wind_dir = (self.wind_x / self._wind_mag, self.wind_y / self._wind_mag)
        else:
            self._wind_dir = (0.0, 0.0)

        self.sigma_c = float(sigma_c)
        self.sigma_r = float(sigma_r)
        self.r_goal = float(r_goal)
        self.b_hold = float(b_hold)
        self.b_oob = float(b_oob)
        self.max_steps = int(max_steps)
        self.stack_n = int(stack_n)
        self.bg_c = float(bg_c)
        self.sensor_noise = float(sensor_noise)

        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.omega_max = float(omega_max)
        self.accel_max = float(accel_max)
        self.omega_accel_max = float(omega_accel_max)
        self.control_penalty = float(control_penalty)
        self.turn_penalty = float(turn_penalty)
        self.cast_penalty = float(cast_penalty)
        self.turn_requires_cast = bool(turn_requires_cast)
        self.turn_window_steps = int(max(1, turn_window_steps))

        # action = [v_cmd, omega_cmd, cast_gate]
        self.action_space = spaces.Box(
            low=np.array([self.v_min, -self.omega_max, 0.0], dtype=np.float32),
            high=np.array([self.v_max, self.omega_max, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Keep v3-style per-step observation (no stacking in output): [concentration, mode]
        # mode: 0=run, 1=cast
        self.obs_step_dim = 2
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.obs_step_dim,), dtype=np.float32)

        self.np_random = np.random.default_rng(seed)

        self._step = 0
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0
        self.v = 0.0
        self.omega = 0.0

        self.in_cast = False
        self.cast_phase = 0
        self.turn_steps_left = 0
        self._scan_dirs = np.array([np.pi / 2, -np.pi / 2], dtype=np.float32)
        self._scan_seq = (0, 1, 0, 1)
        self._scan_c = np.zeros(4, dtype=np.float32)

        self._sense_pt = None
        self._trail = []
        self._img_size = 360
        self._heatmap_img = None
        self._cbar_img = None
        self._last_obs = np.zeros((self.obs_step_dim,), dtype=np.float32)

    def _conc(self, x, y):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        xr = x - np.float32(self.src_x)
        yr = y - np.float32(self.src_y)

        if self._wind_mag <= 1e-6:
            r2 = xr * xr + yr * yr
            c = np.exp(-r2 / (2.0 * self.sigma_c * self.sigma_c))
        else:
            wx, wy = self._wind_dir
            t = xr * wx + yr * wy
            s = -xr * wy + yr * wx
            stretch = 1.0 + min(self._wind_mag, 2.0)
            sigma_s = self.sigma_c
            sigma_t = self.sigma_c * stretch
            sigma_up = self.sigma_c / stretch
            t_pos = np.maximum(0.0, t)
            t_neg = np.maximum(0.0, -t)
            c = np.exp(-((s * s) / (2.0 * sigma_s ** 2) + (t_pos ** 2) / (2.0 * sigma_t ** 2) + (t_neg ** 2) / (2.0 * sigma_up ** 2)))

        c = self.bg_c + (1.0 - self.bg_c) * c
        return np.clip(c, 0.0, 1.0)

    def _sense(self, phi=0.0):
        ang = float(self.th + phi)
        sx = float(self.x + np.cos(ang) * self.ds)
        sy = float(self.y + np.sin(ang) * self.ds)
        c = self._conc(sx, sy)
        if self.sensor_noise > 0:
            c += float(self.np_random.normal(0, self.sensor_noise))
        self._sense_pt = (sx, sy)
        return float(np.clip(c, 0.0, 1.0))

    def _set_obs(self, c, mode):
        self._last_obs = np.array([float(c), float(mode)], dtype=np.float32)

    def _get_obs(self):
        return self._last_obs.copy()

    def _norm_angle(self, a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _cast_step(self):
        i = int(self.cast_phase)
        side = int(self._scan_seq[i])
        phi = float(self._scan_dirs[side])
        c = self._sense(phi)
        self._scan_c[i] = float(c)
        self._set_obs(c, mode=1.0)

        self.cast_phase += 1
        if self.cast_phase >= 4:
            self.in_cast = False
            self.cast_phase = 0
            self._scan_c[:] = 0.0
            self.turn_steps_left = self.turn_window_steps
        return c

    def reset(self, seed=None, options=None):
        del options
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        r_min, r_max = max(self.r_goal + 0.25, 0.6), min(0.8 * self.L, self.L - 0.25)
        cx, cy = self.src_x, self.src_y

        x0, y0 = 0.0, 0.0
        for _ in range(200):
            r0 = self.np_random.uniform(r_min, r_max)
            ang = self.np_random.uniform(-np.pi, np.pi)
            x0, y0 = cx + r0 * np.cos(ang), cy + r0 * np.sin(ang)
            if (-self.L + 0.25) <= x0 <= (self.L - 0.25) and (-self.L + 0.25) <= y0 <= (self.L - 0.25):
                break

        self.x, self.y = float(x0), float(y0)
        self.th = self.np_random.uniform(-np.pi, np.pi)
        self.v = 0.0
        self.omega = 0.0
        self._step = 0
        self.in_cast = False
        self.cast_phase = 0
        self.turn_steps_left = 0
        self._scan_c[:] = 0.0
        self._trail = [(self.x, self.y)]

        c0 = self._sense(0.0)
        self._set_obs(c0, mode=0.0)
        return self._get_obs(), {}

    def step(self, action):
        self._step += 1
        a = np.asarray(action, dtype=np.float32)
        a = np.clip(a, self.action_space.low, self.action_space.high)
        v_cmd = float(a[0])
        omega_cmd = float(a[1])
        cast_cmd = int(np.rint(np.clip(a[2], 0.0, 1.0)))

        did_cast = False
        can_turn_now = (not self.turn_requires_cast) or (self.turn_steps_left > 0)

        if self.in_cast:
            c = self._cast_step()
            self.v = 0.0
            self.omega = 0.0
            did_cast = True
        elif cast_cmd == 1:
            self.in_cast = True
            self.cast_phase = 0
            self._scan_c[:] = 0.0
            c = self._cast_step()
            self.v = 0.0
            self.omega = 0.0
            did_cast = True
        else:
            if not can_turn_now:
                omega_cmd = 0.0

            dv = np.clip(v_cmd - self.v, -self.accel_max * self.dt, self.accel_max * self.dt)
            domega = np.clip(omega_cmd - self.omega, -self.omega_accel_max * self.dt, self.omega_accel_max * self.dt)

            self.v = float(np.clip(self.v + dv, self.v_min, self.v_max))
            self.omega = float(np.clip(self.omega + domega, -self.omega_max, self.omega_max))

            self.th = self._norm_angle(self.th + self.omega * self.dt)
            self.x += self.v * np.cos(self.th) * self.dt
            self.y += self.v * np.sin(self.th) * self.dt

            c = self._sense(0.0)
            self._set_obs(c, mode=0.0)

            if self.turn_requires_cast and self.turn_steps_left > 0:
                self.turn_steps_left -= 1

        oob = (abs(self.x) > self.L) or (abs(self.y) > self.L)
        terminated = bool(oob)
        truncated = bool(self._step >= self.max_steps)

        dx, dy = self.x - self.src_x, self.y - self.src_y
        d = np.hypot(dx, dy)

        r = np.exp(-d / self.sigma_r)
        if d < self.r_goal:
            r += self.b_hold
        if did_cast:
            r -= self.cast_penalty
        else:
            r -= self.control_penalty * (abs(self.v) / (self.v_max + 1e-6))
            r -= self.turn_penalty * (abs(self.omega) / (self.omega_max + 1e-6))
        if oob:
            r -= self.b_oob

        info = {
            "d": d,
            "c": float(c),
            "in_goal": int(d < self.r_goal),
            "src_x": self.src_x,
            "src_y": self.src_y,
            "v": self.v,
            "omega": self.omega,
            "did_cast": int(did_cast),
            "cast_cmd": cast_cmd,
            "in_cast": int(self.in_cast),
            "can_turn": int(can_turn_now),
            "turn_steps_left": int(self.turn_steps_left),
        }
        self._trail.append((self.x, self.y))
        return self._get_obs(), float(r), terminated, truncated, info

    def _world_to_px(self, x, y):
        size = self._img_size - 1
        px = int(np.clip((float(x) + self.L) / (2.0 * self.L) * size, 0, size))
        py = int(np.clip((self.L - float(y)) / (2.0 * self.L) * size, 0, size))
        return px, py

    def _build_heatmap(self):
        size = self._img_size
        xs = np.linspace(-self.L, self.L, size, dtype=np.float32)
        ys = np.linspace(self.L, -self.L, size, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)
        c = self._conc(X, Y).astype(np.float32)
        try:
            from matplotlib import cm
            rgb = cm.inferno(c)[..., :3].astype(np.float32)
            return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        except Exception:
            r = np.clip(4.0 * c - 1.5, 0.0, 1.0)
            g = np.clip(4.0 * c - 2.5, 0.0, 1.0)
            b = np.clip(4.0 * c - 3.5, 0.0, 1.0)
            return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)

    def render(self):
        if self.render_mode not in (None, "rgb_array"):
            return None
        try:
            from PIL import Image, ImageDraw
        except Exception:
            if self._heatmap_img is None:
                self._heatmap_img = self._build_heatmap()
            return self._heatmap_img.copy()

        W = H = self._img_size
        if self._heatmap_img is None or self._heatmap_img.shape[:2] != (H, W):
            self._heatmap_img = self._build_heatmap()

        img = Image.fromarray(self._heatmap_img.copy(), mode="RGB")
        draw = ImageDraw.Draw(img)

        draw.rectangle((0, 0, W - 1, H - 1), outline=(255, 255, 255), width=1)

        src_px, src_py = self._world_to_px(self.src_x, self.src_y)
        rg = max(1, int(round(self.r_goal / (2.0 * self.L) * (W - 1))))
        draw.ellipse((src_px - rg, src_py - rg, src_px + rg, src_py + rg), outline=(190, 190, 190), width=2)
        draw.ellipse((src_px - 4, src_py - 4, src_px + 4, src_py + 4), fill=(0, 0, 0))

        if len(self._trail) > 1:
            pts = [self._world_to_px(tx, ty) for tx, ty in self._trail]
            draw.line(pts, fill=(80, 220, 255), width=2)

        ax, ay = self._world_to_px(self.x, self.y)
        size = 10
        p0 = (ax + size * np.cos(self.th), ay - size * np.sin(self.th))
        p1 = (ax + size * np.cos(self.th + 2.5), ay - size * np.sin(self.th + 2.5))
        p2 = (ax + size * np.cos(self.th - 2.5), ay - size * np.sin(self.th - 2.5))
        tri = [tuple(map(int, p0)), tuple(map(int, p1)), tuple(map(int, p2))]
        draw.polygon(tri, fill=(50, 100, 220))

        if self._sense_pt is not None:
            sx, sy = self._world_to_px(self._sense_pt[0], self._sense_pt[1])
            draw.line((ax, ay, sx, sy), fill=(220, 60, 60), width=2)
            draw.ellipse((sx - 3, sy - 3, sx + 3, sy + 3), fill=(220, 60, 60))

        return np.array(img, dtype=np.uint8)

    def close(self):
        self._heatmap_img = None
        self._cbar_img = None
