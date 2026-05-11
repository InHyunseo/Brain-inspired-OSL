# odor_env.py for gymnasium environment
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class OdorHoldEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        L=3.0,
        dt=0.1,
        v_fixed=0.25,
        w_list=(-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0),
        sensor_offset=0.08,
        sigma_c=1.0,
        sigma_r=1.0,
        r_goal=0.35,
        b_hold=0.5,
        lam_w=0.02,
        b_oob=5.0,
        max_steps=300,
        stack_n=4,
        seed=0,
        bg_c=0.0,   # 필요하면 0.005 같은 값으로 배경농도 추가
        sensor_noise=0.01
    ):
        super().__init__()
        self.render_mode = render_mode

        self.L = float(L)
        self.dt = float(dt)
        self.v = float(v_fixed)
        self.w_list = np.array(w_list, dtype=np.float32)
        self.ds = float(sensor_offset)

        self.sigma_c = float(sigma_c)
        self.sigma_r = float(sigma_r)
        self.r_goal = float(r_goal)
        self.b_hold = float(b_hold)
        self.lam_w = float(lam_w)
        self.b_oob = float(b_oob)
        self.max_steps = int(max_steps)
        self.stack_n = int(stack_n)
        self.bg_c = float(bg_c)

        self.action_space = spaces.Discrete(len(self.w_list))
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2 * self.stack_n,), dtype=np.float32
        )

        self.np_random = np.random.default_rng(seed)
        self._obs_buf = np.zeros((self.stack_n, 2), dtype=np.float32)
        self._step = 0

        self.x = 0.0
        self.y = 0.0
        self.th = 0.0
        self.sensor_noise = float(sensor_noise)

        # render cache
        self._img_size = 360

    def _conc(self, x, y):
        r2 = x * x + y * y
        c = np.exp(-r2 / (2.0 * self.sigma_c * self.sigma_c))
        c = self.bg_c + (1.0 - self.bg_c) * c
        return float(np.clip(c, 0.0, 1.0))

    def _sense(self):
        c = np.cos(self.th)
        s = np.sin(self.th)

        # robot frame: +y left, -y right
        lx = self.x + (-s) * self.ds
        ly = self.y + ( c) * self.ds
        rx = self.x + ( s) * self.ds
        ry = self.y + (-c) * self.ds

        cL = self._conc(lx, ly)
        cR = self._conc(rx, ry)

        if self.sensor_noise > 0:
            cL += float(self.np_random.normal(0, self.sensor_noise))
            cR += float(self.np_random.normal(0, self.sensor_noise))
            cL = float(np.clip(cL, 0.0, 1.0))
            cR = float(np.clip(cR, 0.0, 1.0))

        return np.array([cL, cR], dtype=np.float32)

    def _get_obs(self):
        return self._obs_buf.reshape(-1).copy()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # 링에서 랜덤 스폰(너가 말한 방식)
        r_min = max(self.r_goal + 0.25, 0.6)
        r_max = min(0.8 * self.L, self.L - 0.25)
        r0 = self.np_random.uniform(r_min, r_max)
        ang = self.np_random.uniform(-np.pi, np.pi)

        self.x = r0 * np.cos(ang)
        self.y = r0 * np.sin(ang)
        self.th = self.np_random.uniform(-np.pi, np.pi)

        self._step = 0
        s = self._sense()
        self._obs_buf[:] = s
        return self._get_obs(), {}

    def step(self, action):
        self._step += 1
        w = float(self.w_list[int(action)])

        self.x += self.v * np.cos(self.th) * self.dt
        self.y += self.v * np.sin(self.th) * self.dt
        self.th += w * self.dt
        self.th = (self.th + np.pi) % (2 * np.pi) - np.pi

        s = self._sense()
        self._obs_buf[:-1] = self._obs_buf[1:]
        self._obs_buf[-1] = s

        oob = (abs(self.x) > self.L) or (abs(self.y) > self.L)
        terminated = bool(oob)
        truncated = bool(self._step >= self.max_steps)

        d = float(np.hypot(self.x, self.y))

        r = float(np.exp(-d / self.sigma_r))
        if d < self.r_goal:
            r += self.b_hold
        r -= self.lam_w * abs(w)
        if oob:
            r -= self.b_oob

        info = {"d": d, "w": w, "cL": float(s[0]), "cR": float(s[1]), "in_goal": int(d < self.r_goal)}
        return self._get_obs(), r, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None

        try:
            from PIL import Image, ImageDraw
        except Exception:
            return None

        W = H = self._img_size
        img = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        def to_px(x, y):
            # world [-L,L] -> image [0,W-1], y up
            px = int((x + self.L) / (2 * self.L) * (W - 1))
            py = int((self.L - y) / (2 * self.L) * (H - 1))
            return px, py

        # goal circle
        cx, cy = to_px(0.0, 0.0)
        rg = int(self.r_goal / (2 * self.L) * (W - 1))
        draw.ellipse((cx - rg, cy - rg, cx + rg, cy + rg), outline=(180, 180, 180), width=2)

        # source dot
        rs = 4
        draw.ellipse((cx - rs, cy - rs, cx + rs, cy + rs), fill=(0, 0, 0))

        # agent triangle
        ax, ay = to_px(self.x, self.y)
        size = 10
        th = self.th
        # triangle in pixel coords
        p0 = (ax + size * np.cos(th), ay - size * np.sin(th))
        p1 = (ax + size * np.cos(th + 2.5), ay - size * np.sin(th + 2.5))
        p2 = (ax + size * np.cos(th - 2.5), ay - size * np.sin(th - 2.5))
        tri = [tuple(map(int, p0)), tuple(map(int, p1)), tuple(map(int, p2))]
        draw.polygon(tri, fill=(50, 100, 220))

        return np.array(img, dtype=np.uint8)

    def close(self):
        pass