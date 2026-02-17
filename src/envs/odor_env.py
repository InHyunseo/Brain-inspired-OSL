import numpy as np
import gymnasium as gym
from gymnasium import spaces

class OdorHoldEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        L=3.0, dt=0.1, v_fixed=0.25,
        src_x=0.0, src_y=0.0,
        wind_x=1.0, wind_y=0.0,
        sensor_offset=0.08, sigma_c=1.0, sigma_r=1.0,
        r_goal=0.35, b_hold=0.5, b_oob=5.0,
        max_steps=300, stack_n=1, seed=0,
        bg_c=0.0, sensor_noise=0.01,
        scan_penalty=0.02, turn_penalty=0.02, cast_turn=0.4,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.L = float(L)
        self.dt = float(dt)
        self.v = float(v_fixed)
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
        self.scan_penalty = float(scan_penalty)
        self.turn_penalty = float(turn_penalty)
        self.cast_turn = float(cast_turn)

        self.action_space = spaces.Discrete(4) # 0:RUN, 1:CAST, 2:TURN_L, 3:TURN_R
        self.need_turn = False
        self.obs_step_dim = 2
        obs_dim = self.obs_step_dim * self.stack_n
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self._obs_buf = np.zeros((self.stack_n, self.obs_step_dim), dtype=np.float32)
        self.np_random = np.random.default_rng(seed)

        # state
        self._step = 0
        self.x = 0.0; self.y = 0.0; self.th = 0.0
        self.in_cast = False
        self.cast_phase = 0
        self._scan_dirs = np.array([np.pi / 2, -np.pi / 2], dtype=np.float32)
        self._scan_seq = (0, 1, 0, 1)
        self._scan_c = np.zeros(4, dtype=np.float32)
        self._last_scan_delta = 0.0
        self._last_scan_meanL = 0.0
        self._last_scan_meanR = 0.0

        # render cache
        self._img_size = 360
        self._sense_pt = None
        self._render_scan_idx = 0
        self._render_mode = 0
        self._heatmap_img = None
        self._cbar_img = None
        self._trail = []

    def _conc(self, x, y):
        # ... (기존 _conc 로직 유지) ...
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
            c = np.exp(-((s * s) / (2.0 * sigma_s**2) + (t_pos**2) / (2.0 * sigma_t**2) + (t_neg**2) / (2.0 * sigma_up**2)))

        c = self.bg_c + (1.0 - self.bg_c) * c
        return np.clip(c, 0.0, 1.0)

    def _sense(self, phi):
        ang = float(self.th + phi)
        sx = float(self.x + np.cos(ang) * self.ds)
        sy = float(self.y + np.sin(ang) * self.ds)
        c = self._conc(sx, sy)
        if self.sensor_noise > 0:
            c += float(self.np_random.normal(0, self.sensor_noise))
        self._sense_pt = (sx, sy)
        return float(np.clip(c, 0.0, 1.0))

    def _push_obs(self, c, mode):
        row = np.array([c, float(mode)], dtype=np.float32)
        self._obs_buf[:-1] = self._obs_buf[1:]
        self._obs_buf[-1] = row

    def _get_obs(self):
        return self._obs_buf.reshape(-1).copy()

    def _world_to_px(self, x, y):
        size = self._img_size - 1
        px = int(np.clip((float(x) + self.L) / (2.0 * self.L) * size, 0, size))
        py = int(np.clip((self.L - float(y)) / (2.0 * self.L) * size, 0, size))
        return px, py

    def _draw_disk(self, img, cx, cy, radius, color):
        h, w = img.shape[:2]
        x0 = max(0, cx - radius)
        x1 = min(w - 1, cx + radius)
        y0 = max(0, cy - radius)
        y1 = min(h - 1, cy + radius)
        if x0 > x1 or y0 > y1:
            return
        yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
        mask = (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy) <= radius * radius
        patch = img[y0:y1 + 1, x0:x1 + 1]
        patch[mask] = color

    def _draw_line(self, img, x0, y0, x1, y1, color):
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        xs = np.clip(np.round(np.linspace(x0, x1, n)).astype(np.int32), 0, img.shape[1] - 1)
        ys = np.clip(np.round(np.linspace(y0, y1, n)).astype(np.int32), 0, img.shape[0] - 1)
        img[ys, xs] = color

    def _build_heatmap(self):
        size = self._img_size
        xs = np.linspace(-self.L, self.L, size, dtype=np.float32)
        ys = np.linspace(self.L, -self.L, size, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)
        c = self._conc(X, Y).astype(np.float32)
        try:
            from matplotlib import cm
            # Keep GIF heatmap visually aligned with plot_trajs_png (inferno + alpha=0.5 on white).
            rgb = cm.inferno(c)[..., :3].astype(np.float32)
            rgb = rgb * 0.5 + 1.0 * 0.5
            return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        except Exception:
            # Fallback when matplotlib colormap is unavailable.
            r = np.clip(4.0 * c - 1.5, 0.0, 1.0)
            g = np.clip(4.0 * c - 2.5, 0.0, 1.0)
            b = np.clip(4.0 * c - 3.5, 0.0, 1.0)
            return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        
        # Spawn Logic
        r_min, r_max = max(self.r_goal + 0.25, 0.6), min(0.8 * self.L, self.L - 0.25)
        cx, cy = self.src_x, self.src_y
        
        for _ in range(200):
            r0 = self.np_random.uniform(r_min, r_max)
            ang = self.np_random.uniform(-np.pi, np.pi)
            x0, y0 = cx + r0 * np.cos(ang), cy + r0 * np.sin(ang)
            if (-self.L + 0.25) <= x0 <= (self.L - 0.25) and (-self.L + 0.25) <= y0 <= (self.L - 0.25):
                break
        
        self.x, self.y = float(x0), float(y0)
        self.th = self.np_random.uniform(-np.pi, np.pi)
        self._step = 0
        self.in_cast = False
        self.cast_phase = 0
        self.need_turn = False
        self._scan_c[:] = 0.0
        self._render_scan_idx = 0
        self._render_mode = 0
        self._trail = [(self.x, self.y)]
        
        # Init Obs
        c0 = self._sense(0.0)
        self._obs_buf[:] = 0.0
        for i in range(self.stack_n):
            self._obs_buf[i] = np.array([c0, 0.0], dtype=np.float32)
            
        return self._get_obs(), {}

    def _norm_angle(self, a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _cast_step(self):
        i = int(self.cast_phase)
        side = int(self._scan_seq[i])
        phi = float(self._scan_dirs[side])
        self._render_scan_idx = 1 if side == 0 else 2
        self._render_mode = 1
        c = self._sense(phi)
        self._scan_c[i] = float(c)
        self._push_obs(c, mode=1)
        self.cast_phase += 1
        
        finished = (self.cast_phase >= 4)
        if finished:
            meanL = (self._scan_c[0] + self._scan_c[2]) * 0.5
            meanR = (self._scan_c[1] + self._scan_c[3]) * 0.5
            self._last_scan_meanL, self._last_scan_meanR = meanL, meanR
            self._last_scan_delta = meanL - meanR
            self.in_cast = False
            self.cast_phase = 0
            self._scan_c[:] = 0.0
            self.need_turn = True
        return float(c), finished

    def step(self, action):
        self._step += 1
        a = int(action)
        did_scan, did_turn, invalid, moved = False, False, False, False
        mode = 0 # Default RUN

        if self.in_cast:
            mode = 1
            c, _ = self._cast_step()
            did_scan = True
        elif self.need_turn:
            if a == 2: # TURN_L
                self.th = self._norm_angle(self.th + self.cast_turn)
                self.need_turn = False; did_turn = True
                self._render_scan_idx = 0
                self._render_mode = 0
                c = self._sense(0.0); self._push_obs(c, mode=0)
            elif a == 3: # TURN_R
                self.th = self._norm_angle(self.th - self.cast_turn)
                self.need_turn = False; did_turn = True
                self._render_scan_idx = 0
                self._render_mode = 0
                c = self._sense(0.0); self._push_obs(c, mode=0)
            else: # Invalid
                invalid = True; mode = 1
                c = float(self._obs_buf[-1, 0])
        else: # Normal
            if a == 1: # START CAST
                self.in_cast = True; self.cast_phase = 0; self._scan_c[:] = 0.0
                mode = 1
                c, _ = self._cast_step(); did_scan = True
            elif a == 0: # RUN
                mode = 0
                self.x += self.v * np.cos(self.th) * self.dt
                self.y += self.v * np.sin(self.th) * self.dt
                moved = True
                self._render_scan_idx = 0
                self._render_mode = 0
                c = self._sense(0.0); self._push_obs(c, mode=0)
            else: # Invalid Turn
                invalid = True; mode = 0
                c = float(self._obs_buf[-1, 0])

        oob = (abs(self.x) > self.L) or (abs(self.y) > self.L)
        terminated = bool(oob)
        truncated = bool(self._step >= self.max_steps)
        
        dx, dy = self.x - self.src_x, self.y - self.src_y
        d = np.hypot(dx, dy)
        
        r = np.exp(-d / self.sigma_r)
        if d < self.r_goal: r += self.b_hold
        if did_scan: r -= self.scan_penalty
        if did_turn or invalid: r -= self.turn_penalty
        if oob: r -= self.b_oob

        info = {
            "d": d, "c": float(c), "mode": mode, "in_goal": int(d < self.r_goal),
            "src_x": self.src_x, "src_y": self.src_y
        }
        self._trail.append((self.x, self.y))
        return self._get_obs(), r, terminated, truncated, info

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
            bar_w = max(10, int(W * 0.04))
            bar_h = max(80, int(H * 0.5))
            grad = np.linspace(1.0, 0.0, bar_h, dtype=np.float32)[:, None]
            try:
                from matplotlib import cm
                bar_rgb = (cm.inferno(grad)[..., :3] * 255.0).astype(np.uint8)
            except Exception:
                v = (grad * 255.0).astype(np.uint8)
                bar_rgb = np.stack([v, v, v], axis=-1)
            # bar_rgb: (H, 1, 3) -> tile width axis to (H, W, 3)
            self._cbar_img = np.repeat(bar_rgb, bar_w, axis=1)

        img = Image.fromarray(self._heatmap_img.copy(), mode="RGB")
        draw = ImageDraw.Draw(img)

        # Border
        draw.rectangle((0, 0, W - 1, H - 1), outline=(255, 255, 255), width=1)

        # Goal + source marker
        src_px, src_py = self._world_to_px(self.src_x, self.src_y)
        rg = max(1, int(round(self.r_goal / (2.0 * self.L) * (W - 1))))
        draw.ellipse((src_px - rg, src_py - rg, src_px + rg, src_py + rg), outline=(190, 190, 190), width=2)
        rs = 4
        draw.ellipse((src_px - rs, src_py - rs, src_px + rs, src_py + rs), fill=(0, 0, 0))

        # Trajectory as connected polyline + start/end markers
        if len(self._trail) > 1:
            pts = [self._world_to_px(tx, ty) for tx, ty in self._trail]
            draw.line(pts, fill=(80, 220, 255), width=2)
            sx, sy = pts[0]
            ex, ey = pts[-1]
            draw.line((sx - 4, sy - 4, sx + 4, sy + 4), fill=(255, 255, 255), width=2)
            draw.line((sx - 4, sy + 4, sx + 4, sy - 4), fill=(255, 255, 255), width=2)
            draw.rectangle((ex - 3, ey - 3, ex + 3, ey + 3), outline=(255, 255, 255), width=2)

        # Agent as oriented triangle
        ax, ay = self._world_to_px(self.x, self.y)
        size = 10
        p0 = (ax + size * np.cos(self.th), ay - size * np.sin(self.th))
        p1 = (ax + size * np.cos(self.th + 2.5), ay - size * np.sin(self.th + 2.5))
        p2 = (ax + size * np.cos(self.th - 2.5), ay - size * np.sin(self.th - 2.5))
        tri = [tuple(map(int, p0)), tuple(map(int, p1)), tuple(map(int, p2))]
        draw.polygon(tri, fill=(50, 100, 220))

        # Sensing ray + current sensor point + scan direction label (F/L/R)
        if self._sense_pt is not None:
            sx, sy = self._world_to_px(self._sense_pt[0], self._sense_pt[1])
            draw.line((ax, ay, sx, sy), fill=(220, 60, 60), width=2)
            rr = 3
            draw.ellipse((sx - rr, sy - rr, sx + rr, sy + rr), fill=(220, 60, 60))
            labels = ("F", "L", "R")
            draw.text((ax + 8, ay + 8), labels[int(self._render_scan_idx)], fill=(0, 0, 0))

        # Concentration colorbar
        if self._cbar_img is not None:
            cbar = Image.fromarray(self._cbar_img, mode="RGB")
            pad = 6
            bx = W - cbar.size[0] - pad
            by = pad
            img.paste(cbar, (bx, by))
            draw.text((bx - 2, by - 2), "1.0", fill=(0, 0, 0))
            draw.text((bx - 2, by + cbar.size[1] - 10), "0.0", fill=(0, 0, 0))

        return np.array(img, dtype=np.uint8)
    
    def close(self):
        self._heatmap_img = None
        self._cbar_img = None
