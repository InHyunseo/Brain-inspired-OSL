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
        self._heatmap_img = None
        self._cbar_img = None

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
                c = self._sense(0.0); self._push_obs(c, mode=0)
            elif a == 3: # TURN_R
                self.th = self._norm_angle(self.th - self.cast_turn)
                self.need_turn = False; did_turn = True
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
        return self._get_obs(), r, terminated, truncated, info

    def render(self):
        # ... (렌더링 코드는 길어서 생략하나 필요 시 기존 코드의 render 메소드를 그대로 사용하세요)
        # 중요: 서버 환경에서는 render 사용 시 주의
        return None 
    
    def close(self):
        pass