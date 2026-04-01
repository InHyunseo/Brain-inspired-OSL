import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from scipy.ndimage import gaussian_filter


class EvalEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        L=3.0,
        dt=0.1,
        src_x=0.0,
        src_y=0.0,
        wind_x=0.0,
        wind_y=0.0,
        sigma_c=1.0,
        r_goal=0.35,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.L = float(L)
        self.dt = float(dt)
        self.ds = 0.08
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
        self.r_goal = float(r_goal)
        self.b_hold = 0.5
        self.b_oob = 5.0
        self.max_steps = 300
        self.bg_c = 0.0
        self.sensor_noise = 0.01

        self.v_min = 0.2
        self.v_max = 0.45
        self.omega_max = 5.0
        self.accel_max = 1.2
        self.omega_accel_max = 50.0
        self.control_penalty = 0.01
        self.turn_penalty = 0.01
        self.cast_penalty = 0.028
        self.reward_scale = 0.5
        self.goal_hold_steps = 20
        self.terminate_on_hold = True
        self.turn_requires_cast = True
        self.turn_window_steps = 1

        self.np_random = np.random.default_rng(None)

        self.action_space = spaces.Box(
            low=np.array([self.v_min, -self.omega_max, 0.0], dtype=np.float32),
            high=np.array([self.v_max, self.omega_max, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.obs_step_dim = 2
        obs_low = np.array([0.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

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
        self.goal_hold_count = 0
        self.prev_in_goal = False

        self._sense_pt = None
        self._trail = []
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
            c = np.exp(
                -(
                    (s * s) / (2.0 * sigma_s**2)
                    + (t_pos**2) / (2.0 * sigma_t**2)
                    + (t_neg**2) / (2.0 * sigma_up**2)
                )
            )

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

        spawn_margin = 0.25
        spawn_radius_tries = 80
        spawn_angle_tries = 80
        r_min = max(self.r_goal + spawn_margin, 0.6)
        r_max = min(0.8 * self.L, self.L - spawn_margin)
        cx, cy = self.src_x, self.src_y

        x0, y0 = float(cx), float(cy)
        found = False
        for _ in range(max(1, int(spawn_radius_tries))):
            r0 = float(self.np_random.uniform(r_min, r_max))
            for _ in range(max(1, int(spawn_angle_tries))):
                ang = float(self.np_random.uniform(-np.pi, np.pi))
                tx = float(cx + r0 * np.cos(ang))
                ty = float(cy + r0 * np.sin(ang))
                if (-self.L + spawn_margin) <= tx <= (self.L - spawn_margin) and (-self.L + spawn_margin) <= ty <= (self.L - spawn_margin):
                    x0, y0 = tx, ty
                    found = True
                    break
            if found:
                break
        if not found:
            x0 = float(np.clip(cx, -self.L + spawn_margin, self.L - spawn_margin))
            y0 = float(np.clip(cy, -self.L + spawn_margin, self.L - spawn_margin))

        self.x, self.y = float(x0), float(y0)
        self.th = self.np_random.uniform(-np.pi, np.pi)
        self.v = 0.0
        self.omega = 0.0
        self._step = 0
        self.in_cast = False
        self.cast_phase = 0
        self.turn_steps_left = 0
        self._scan_c[:] = 0.0
        self.goal_hold_count = 0
        self.prev_in_goal = False
        self._trail = [(self.x, self.y)]

        c0 = self._sense(0.0)
        self._set_obs(c0, mode=0.0)
        return self._get_obs(), {}

    def step(self, action):
        self._step += 1
        prev_c = float(self._last_obs[0])
        prev_in_goal = bool(self.prev_in_goal)
        a = np.asarray(action, dtype=np.float32)
        a = np.clip(a, self.action_space.low, self.action_space.high)
        v_cmd = float(a[0])
        omega_cmd = float(a[1])
        cast_cmd = int(np.rint(np.clip(a[2], 0.0, 1.0)))

        did_cast = False
        cast_started = False
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
            cast_started = True
            c = self._cast_step()
            self.v = 0.0
            self.omega = 0.0
            did_cast = True
        else:
            dv = np.clip(v_cmd - self.v, -self.accel_max * self.dt, self.accel_max * self.dt)
            self.v = float(np.clip(self.v + dv, self.v_min, self.v_max))

            if can_turn_now:
                domega = np.clip(
                    omega_cmd - self.omega,
                    -self.omega_accel_max * self.dt,
                    self.omega_accel_max * self.dt,
                )
                self.omega = float(np.clip(self.omega + domega, -self.omega_max, self.omega_max))
                if self.turn_steps_left > 0:
                    self.turn_steps_left -= 1
            else:
                self.omega = 0.0

            self.th = self._norm_angle(self.th + self.omega * self.dt)
            self.x += self.v * np.cos(self.th) * self.dt
            self.y += self.v * np.sin(self.th) * self.dt
            c = self._sense(0.0)

        self._set_obs(c, mode=1.0 if self.in_cast else 0.0)

        dist = float(np.hypot(self.x - self.src_x, self.y - self.src_y))
        in_goal = dist <= self.r_goal
        if in_goal:
            self.goal_hold_count += 1
        else:
            self.goal_hold_count = 0
        success_hold = self.goal_hold_count >= self.goal_hold_steps
        self.prev_in_goal = in_goal

        terminated = False
        truncated = False
        out_of_bounds = abs(self.x) > self.L or abs(self.y) > self.L
        if out_of_bounds:
            terminated = True
        if success_hold and self.terminate_on_hold:
            terminated = True
        if self._step >= self.max_steps:
            truncated = True

        reward = 0.0
        reward += self.reward_scale * (float(c) - prev_c)
        if in_goal and not prev_in_goal:
            reward += self.b_hold
        reward -= self.control_penalty * abs(self.omega)
        if cast_started or did_cast:
            reward -= self.cast_penalty
        if abs(self.omega) > 1e-6:
            reward -= self.turn_penalty * abs(self.omega)
        if out_of_bounds:
            reward -= self.b_oob

        info = {
            "in_goal": int(in_goal),
            "success_hold": int(success_hold),
            "cast_start": int(cast_started),
            "did_cast": int(did_cast),
            "in_cast": int(self.in_cast),
            "can_turn": int(can_turn_now),
            "turn_steps_left": int(self.turn_steps_left),
        }
        self._trail.append((self.x, self.y))
        return self._get_obs(), float(reward), terminated, truncated, info

    def close(self):
        return None


class TurbulentEvalEnv(EvalEnv):
    def __init__(
        self,
        *args,
        field_size=100,
        blur_sigma=5.0,
        noise_scale=1.0,
        noise_bias=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.field_size = int(field_size)
        self.blur_sigma = float(blur_sigma)
        self.noise_scale = float(noise_scale)
        self.noise_bias = float(noise_bias)
        self._field_raw = None
        self._field_view = None
        self._initial_field_view = None

    def _source_grid_center(self):
        gx = (self.src_x + self.L) / (2.0 * self.L) * (self.field_size - 1)
        gy = (self.src_y + self.L) / (2.0 * self.L) * (self.field_size - 1)
        return float(gx), float(gy)

    def _build_base_field(self):
        xs = np.arange(self.field_size, dtype=np.float32)
        ys = np.arange(self.field_size, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)
        cx, cy = self._source_grid_center()
        field = np.exp(0.05 * (-np.abs(X - cx) - np.abs(Y - cy)))
        return field.astype(np.float32)

    def _refresh_field_view(self):
        self._field_view = np.clip(self._field_raw, 0.0, 1.0).astype(np.float32)

    def _update_field(self):
        noise = self.np_random.standard_normal((self.field_size, self.field_size)).astype(np.float32)
        noise_blur = gaussian_filter(noise, sigma=self.blur_sigma).astype(np.float32)
        noise_blur = noise_blur * self.noise_scale + self.noise_bias
        noise_blur_scaled = noise_blur * (self._field_view + 1.0) / 2.0
        self._field_raw = np.maximum(self._field_raw + noise_blur_scaled, 0.0).astype(np.float32)
        self._refresh_field_view()

    def _world_to_grid(self, x, y):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        gx = (x + self.L) / (2.0 * self.L) * (self.field_size - 1)
        gy = (y + self.L) / (2.0 * self.L) * (self.field_size - 1)
        return np.clip(gx, 0.0, self.field_size - 1), np.clip(gy, 0.0, self.field_size - 1)

    def _bilinear_sample(self, field, x, y):
        gx, gy = self._world_to_grid(x, y)
        x0 = np.floor(gx).astype(np.int64)
        y0 = np.floor(gy).astype(np.int64)
        x1 = np.clip(x0 + 1, 0, self.field_size - 1)
        y1 = np.clip(y0 + 1, 0, self.field_size - 1)
        wx = gx - x0.astype(np.float32)
        wy = gy - y0.astype(np.float32)

        v00 = field[y0, x0]
        v10 = field[y0, x1]
        v01 = field[y1, x0]
        v11 = field[y1, x1]
        return (
            (1.0 - wx) * (1.0 - wy) * v00
            + wx * (1.0 - wy) * v10
            + (1.0 - wx) * wy * v01
            + wx * wy * v11
        )

    def _conc(self, x, y):
        if self._field_view is None:
            self._field_raw = self._build_base_field()
            self._refresh_field_view()
        return np.clip(self._bilinear_sample(self._field_view, x, y), 0.0, 1.0)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self._field_raw = self._build_base_field()
        self._refresh_field_view()
        self._initial_field_view = self._field_view.copy()
        return super().reset(seed=None, options=options)

    def step(self, action):
        self._update_field()
        return super().step(action)

    def get_field(self):
        return self._field_view.copy()

    def get_initial_field(self):
        return self._initial_field_view.copy()


def _build_gru(input_size, hidden_size):
    return nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)


def _build_mlp(input_size, hidden_size):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
    )


class GRUActor(nn.Module):
    def __init__(self, obs_dim, cont_act_dim, hidden=147, log_std_min=-5.0, log_std_max=2.0):
        super().__init__()
        self.rnn = _build_gru(obs_dim, hidden)
        self.mu = nn.Linear(hidden, cont_act_dim)
        self.log_std = nn.Linear(hidden, cont_act_dim)
        self.cast_logit = nn.Linear(hidden, 1)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def forward(self, obs, h=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        y, h2 = self.rnn(obs, h)
        mu = self.mu(y)
        log_std = torch.clamp(self.log_std(y), self.log_std_min, self.log_std_max)
        cast_logit = self.cast_logit(y)
        return mu, log_std, cast_logit, h2

    def deterministic(self, obs, action_low, action_high, h=None):
        mu, _, cast_logit, h2 = self.forward(obs, h)
        y = torch.tanh(mu)
        cont_low = action_low[:2]
        cont_high = action_high[:2]
        cont_scale = (cont_high - cont_low) * 0.5
        cont_bias = (cont_high + cont_low) * 0.5
        cont_action = y * cont_scale + cont_bias
        cast_action = (cast_logit > 0.0).float()
        action = torch.cat([cont_action, cast_action], dim=-1)
        return action, h2


class MLPActor(nn.Module):
    def __init__(self, obs_dim, cont_act_dim, hidden=147, log_std_min=-5.0, log_std_max=2.0):
        super().__init__()
        self.encoder = _build_mlp(obs_dim, hidden)
        self.mu = nn.Linear(hidden, cont_act_dim)
        self.log_std = nn.Linear(hidden, cont_act_dim)
        self.cast_logit = nn.Linear(hidden, 1)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def forward(self, obs, h=None):
        del h
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        y = self.encoder(obs)
        mu = self.mu(y)
        log_std = torch.clamp(self.log_std(y), self.log_std_min, self.log_std_max)
        cast_logit = self.cast_logit(y)
        return mu, log_std, cast_logit, None

    def deterministic(self, obs, action_low, action_high, h=None):
        mu, _, cast_logit, h2 = self.forward(obs, h)
        y = torch.tanh(mu)
        cont_low = action_low[:2]
        cont_high = action_high[:2]
        cont_scale = (cont_high - cont_low) * 0.5
        cont_bias = (cont_high + cont_low) * 0.5
        cont_action = y * cont_scale + cont_bias
        cast_action = (cast_logit > 0.0).float()
        action = torch.cat([cont_action, cast_action], dim=-1)
        return action, h2


CONNECTOME_BASE = 90
CONNECTOME_RATIOS = (24, 7, 4, 54, 1)


def _population_sizes(total_size):
    if total_size % CONNECTOME_BASE != 0:
        raise ValueError(
            f"Connectome hidden_size must be a multiple of {CONNECTOME_BASE}; got {total_size}."
        )
    scale = total_size // CONNECTOME_BASE
    if scale < 2:
        raise ValueError(
            f"Connectome requires hidden_size >= {2 * CONNECTOME_BASE}; got {total_size}."
        )
    n_orn = CONNECTOME_RATIOS[0] * scale
    n_pn = CONNECTOME_RATIOS[1] * scale
    n_ln = CONNECTOME_RATIOS[2] * scale
    n_kc = CONNECTOME_RATIOS[3] * scale
    n_mbon = CONNECTOME_RATIOS[4] * scale
    return n_orn, n_pn, n_ln, n_kc, n_mbon


class ConnectomeBackbone(nn.Module):
    def __init__(self, obs_dim, hidden_size, inner_steps=4):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.inner_steps = int(max(1, inner_steps))

        n_orn, n_pn, n_ln, n_kc, n_mbon = _population_sizes(self.hidden_size)
        self.n_orn = n_orn
        self.n_pn = n_pn
        self.n_ln = n_ln
        self.n_kc = n_kc
        self.n_mbon = n_mbon

        self.in_orn = nn.Linear(obs_dim, n_orn)
        self.W_oto = nn.Linear(n_orn, n_orn, bias=False)
        self.W_lto = nn.Linear(n_ln, n_orn, bias=False)

        self.W_otp = nn.Linear(n_orn, n_pn, bias=False)
        self.W_ltp = nn.Linear(n_ln, n_pn, bias=False)
        self.W_ptp = nn.Linear(n_pn, n_pn, bias=False)

        self.W_otl = nn.Linear(n_orn, n_ln, bias=False)
        self.W_ptl = nn.Linear(n_pn, n_ln, bias=False)
        self.W_ltl = nn.Linear(n_ln, n_ln, bias=False)

        self.W_ktk = nn.Linear(n_kc, n_kc, bias=False)
        self.W_mtk = nn.Linear(n_mbon, n_kc, bias=False)
        self.W_ptk = nn.Linear(n_pn, n_kc, bias=False)

        self.W_ktm = nn.Linear(n_kc, n_mbon, bias=False)

        self.b_orn = nn.Parameter(torch.zeros(n_orn))
        self.b_pn = nn.Parameter(torch.zeros(n_pn))
        self.b_ln = nn.Parameter(torch.zeros(n_ln))
        self.b_kc = nn.Parameter(torch.zeros(n_kc))
        self.b_mbon = nn.Parameter(torch.zeros(n_mbon))

        self.readout = nn.Linear(n_mbon, self.hidden_size)

    def _split_state(self, h_flat):
        i0 = self.n_orn
        i1 = i0 + self.n_pn
        i2 = i1 + self.n_ln
        i3 = i2 + self.n_kc
        h_orn = h_flat[:, :i0]
        h_pn = h_flat[:, i0:i1]
        h_ln = h_flat[:, i1:i2]
        h_kc = h_flat[:, i2:i3]
        h_mbon = h_flat[:, i3:]
        return h_orn, h_pn, h_ln, h_kc, h_mbon

    def _pack_state(self, h_orn, h_pn, h_ln, h_kc, h_mbon):
        return torch.cat([h_orn, h_pn, h_ln, h_kc, h_mbon], dim=-1)

    def _init_state(self, batch_size, device, dtype):
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def _parse_hidden(self, h, batch_size, device, dtype):
        if h is None:
            return self._init_state(batch_size, device, dtype)
        if h.dim() == 3:
            if h.size(0) != 1:
                raise ValueError("Connectome hidden state expects first dim 1.")
            h_flat = h[0]
        elif h.dim() == 2:
            h_flat = h
        else:
            raise ValueError("Connectome hidden state must be rank-2 or rank-3 tensor.")
        if h_flat.size(0) != batch_size or h_flat.size(1) != self.hidden_size:
            raise ValueError(
                f"Connectome hidden state shape mismatch: got {tuple(h_flat.shape)}, expected ({batch_size}, {self.hidden_size})."
            )
        return h_flat.to(device=device, dtype=dtype)

    def step(self, h_orn, h_pn, h_ln, h_kc, h_mbon, x_t):
        orn_next = torch.tanh(self.W_oto(h_orn) + self.W_lto(h_ln) + x_t + self.b_orn)
        pn_next = torch.tanh(self.W_otp(orn_next) + self.W_ltp(h_ln) + self.W_ptp(h_pn) + self.b_pn)
        ln_next = torch.tanh(self.W_otl(orn_next) + self.W_ptl(pn_next) + self.W_ltl(h_ln) + self.b_ln)
        kc_next = torch.tanh(self.W_ktk(h_kc) + self.W_mtk(h_mbon) + self.W_ptk(pn_next) + self.b_kc)
        mbon_next = torch.tanh(self.W_ktm(kc_next) + self.b_mbon)
        return orn_next, pn_next, ln_next, kc_next, mbon_next

    def forward(self, obs, h=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        bsz, seq_len, _ = obs.shape
        device = obs.device
        dtype = obs.dtype

        h_flat = self._parse_hidden(h, bsz, device, dtype)
        h_orn, h_pn, h_ln, h_kc, h_mbon = self._split_state(h_flat)

        outputs = []
        for t in range(seq_len):
            x_t = self.in_orn(obs[:, t, :])
            for _ in range(self.inner_steps):
                h_orn, h_pn, h_ln, h_kc, h_mbon = self.step(h_orn, h_pn, h_ln, h_kc, h_mbon, x_t)
            outputs.append(self.readout(h_mbon).unsqueeze(1))

        y = torch.cat(outputs, dim=1)
        h2 = self._pack_state(h_orn, h_pn, h_ln, h_kc, h_mbon).unsqueeze(0)
        return y, h2


class ConnectomeActor(nn.Module):
    def __init__(self, obs_dim, cont_act_dim, hidden=180, log_std_min=-5.0, log_std_max=2.0, connectome_steps=4):
        super().__init__()
        self.backbone = ConnectomeBackbone(obs_dim=obs_dim, hidden_size=hidden, inner_steps=connectome_steps)
        self.mu = nn.Linear(hidden, cont_act_dim)
        self.log_std = nn.Linear(hidden, cont_act_dim)
        self.cast_logit = nn.Linear(hidden, 1)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def forward(self, obs, h=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        y, h2 = self.backbone(obs, h)
        mu = self.mu(y)
        log_std = torch.clamp(self.log_std(y), self.log_std_min, self.log_std_max)
        cast_logit = self.cast_logit(y)
        return mu, log_std, cast_logit, h2

    def deterministic(self, obs, action_low, action_high, h=None):
        mu, _, cast_logit, h2 = self.forward(obs, h)
        y = torch.tanh(mu)
        cont_low = action_low[:2]
        cont_high = action_high[:2]
        cont_scale = (cont_high - cont_low) * 0.5
        cont_bias = (cont_high + cont_low) * 0.5
        cont_action = y * cont_scale + cont_bias
        cast_action = (cast_logit > 0.0).float()
        action = torch.cat([cont_action, cast_action], dim=-1)
        return action, h2


class ActorPolicy:
    def __init__(self, actor, action_low, action_high, device):
        self.actor = actor
        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=device)
        self.device = device

    def get_action_deterministic(self, obs, h=None):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, h2 = self.actor.deterministic(obs_t, self.action_low, self.action_high, h)
        a = action[:, -1, :].squeeze(0).detach().cpu().numpy().astype(np.float32)
        return a, h2


def infer_hidden_size(actor_state):
    mu_weight = actor_state.get("mu.weight")
    if mu_weight is None:
        raise KeyError("Checkpoint actor state is missing 'mu.weight'.")
    return int(mu_weight.shape[1])


def load_policy(ckpt_path, model_type, env, device, connectome_steps):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "actor" not in ckpt:
        raise KeyError(f"Checkpoint {ckpt_path} is missing the 'actor' key.")
    actor_state = ckpt["actor"]
    hidden = infer_hidden_size(actor_state)
    obs_dim = int(env.observation_space.shape[0])

    if model_type == "gru":
        actor = GRUActor(obs_dim, cont_act_dim=2, hidden=hidden).to(device)
    elif model_type == "mlp":
        actor = MLPActor(obs_dim, cont_act_dim=2, hidden=hidden).to(device)
    elif model_type == "connectome":
        actor = ConnectomeActor(
            obs_dim,
            cont_act_dim=2,
            hidden=hidden,
            connectome_steps=connectome_steps,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    actor.load_state_dict(actor_state)
    actor.eval()
    return ActorPolicy(actor, env.action_space.low, env.action_space.high, device)


def rollout(env, policy, episodes, seed_base):
    trajectories = []
    success_entry_count = 0
    success_hold_count = 0
    final_in_goal_count = 0
    cast_start_counts = []
    cast_step_counts = []
    cast_step_ratios = []
    can_turn_ratios = []

    for i in range(episodes):
        ep_seed = int(seed_base + i)
        obs, _ = env.reset(seed=ep_seed)
        h = None
        done = False
        xs, ys = [env.x], [env.y]
        ep_ret = 0.0
        in_goal = False
        hold_success = False
        final_in_goal = False
        cast_start_count = 0
        cast_step_count = 0
        cast_steps = 0
        can_turn_steps = 0
        total_steps = 0

        while not done:
            action, h = policy.get_action_deterministic(obs, h)
            obs, r, term, trunc, info = env.step(action)
            done = bool(term or trunc)
            total_steps += 1
            cast_start_count += int(info.get("cast_start", 0))
            cast_step_count += int(info.get("did_cast", 0))
            cast_steps += int(info.get("in_cast", 0))
            can_turn_steps += int(info.get("can_turn", 0))
            xs.append(env.x)
            ys.append(env.y)
            ep_ret += float(r)
            if info.get("in_goal", 0):
                in_goal = True
            if info.get("success_hold", 0):
                hold_success = True
            final_in_goal = bool(info.get("in_goal", 0))

        if in_goal:
            success_entry_count += 1
        if hold_success:
            success_hold_count += 1
        if final_in_goal:
            final_in_goal_count += 1
        cast_start_counts.append(float(cast_start_count))
        cast_step_counts.append(float(cast_step_count))
        if total_steps > 0:
            cast_step_ratios.append(float(cast_steps) / float(total_steps))
            can_turn_ratios.append(float(can_turn_steps) / float(total_steps))
        else:
            cast_step_ratios.append(0.0)
            can_turn_ratios.append(0.0)
        trajectories.append(
            {
                "return": float(ep_ret),
                "success": bool(in_goal),
                "x": xs,
                "y": ys,
                "seed": ep_seed,
            }
        )

    stats = {
        "success_entry_rate": float(success_entry_count / episodes) if episodes > 0 else 0.0,
        "success_hold_rate": float(success_hold_count / episodes) if episodes > 0 else 0.0,
        "final_in_goal_rate": float(final_in_goal_count / episodes) if episodes > 0 else 0.0,
        "cast_start_count_mean": float(np.mean(cast_start_counts)) if cast_start_counts else 0.0,
        "cast_step_count_mean": float(np.mean(cast_step_counts)) if cast_step_counts else 0.0,
        "cast_step_ratio_mean": float(np.mean(cast_step_ratios)) if cast_step_ratios else 0.0,
        "can_turn_ratio_mean": float(np.mean(can_turn_ratios)) if can_turn_ratios else 0.0,
        "avg_return": float(np.mean([traj["return"] for traj in trajectories])) if trajectories else 0.0,
    }
    return trajectories, stats


def render_rollout_frame_complex(env, title=None):
    L = float(env.L)
    field = env.get_field()

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect("equal")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    im = ax.imshow(field, extent=[-L, L, -L, L], origin="lower", cmap="inferno", vmin=0.0, vmax=1.0)
    ax.plot(float(env.src_x), float(env.src_y), "ko")
    circle = plt.Circle((float(env.src_x), float(env.src_y)), float(env.r_goal), color="gray", fill=False)
    ax.add_patch(circle)

    if len(env._trail) > 0:
        tx = [float(p[0]) for p in env._trail]
        ty = [float(p[1]) for p in env._trail]
        ax.plot(tx, ty, alpha=0.8, color="#50dcff")
        ax.plot(tx[0], ty[0], "x", color="white")
        ax.plot(tx[-1], ty[-1], "s", color="white", markersize=5)

    ax_x = float(env.x)
    ax_y = float(env.y)
    th = float(env.th)
    tri_len = 0.18
    p0 = (ax_x + tri_len * np.cos(th), ax_y + tri_len * np.sin(th))
    p1 = (ax_x + tri_len * np.cos(th + 2.5), ax_y + tri_len * np.sin(th + 2.5))
    p2 = (ax_x + tri_len * np.cos(th - 2.5), ax_y + tri_len * np.sin(th - 2.5))
    tri = plt.Polygon(
        [p0, p1, p2],
        closed=True,
        facecolor=(50 / 255.0, 100 / 255.0, 220 / 255.0),
        edgecolor="white",
        linewidth=0.8,
    )
    ax.add_patch(tri)

    if getattr(env, "_sense_pt", None) is not None:
        sx, sy = float(env._sense_pt[0]), float(env._sense_pt[1])
        ax.plot([ax_x, sx], [ax_y, sy], color=(220 / 255.0, 60 / 255.0, 60 / 255.0), linewidth=2)
        ax.plot(sx, sy, "o", color=(220 / 255.0, 60 / 255.0, 60 / 255.0), markersize=4)

    if title:
        ax.set_title(title)
        ax.title.set_color("white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    frame = buf[:, :, :3].copy()
    plt.close(fig)
    return frame


def show_trajs_complex(env, trajs, title, out_png):
    L = float(env.L)
    field = env.get_initial_field()

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect("equal")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    im = ax.imshow(field, extent=[-L, L, -L, L], origin="lower", cmap="inferno", vmin=0.0, vmax=1.0)
    ax.plot(float(env.src_x), float(env.src_y), "ko")
    circle = plt.Circle((float(env.src_x), float(env.src_y)), float(env.r_goal), color="gray", fill=False)
    ax.add_patch(circle)

    for traj in trajs:
        ax.plot(traj["x"], traj["y"], alpha=0.6)
        ax.plot(traj["x"][0], traj["y"][0], "x")
        ax.plot(traj["x"][-1], traj["y"][-1], "s")

    rets = [traj["return"] for traj in trajs]
    avg_ret = float(np.mean(rets)) if rets else float("nan")
    ax.set_title(f"{title}\nAvg Return: {avg_ret:.2f}")
    ax.title.set_color("white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def generate_gif(env, policy, seed, out_gif, title_prefix):
    obs, _ = env.reset(seed=seed)
    h = None
    done = False
    frames = [render_rollout_frame_complex(env, title=f"{title_prefix} | seed={seed} | step=0")]
    step = 0
    while not done:
        action, h = policy.get_action_deterministic(obs, h)
        obs, _, term, trunc, _ = env.step(action)
        done = bool(term or trunc)
        step += 1
        frames.append(render_rollout_frame_complex(env, title=f"{title_prefix} | seed={seed} | step={step}"))
    imageio.mimsave(out_gif, frames, fps=10, loop=0)


def save_summary(out_json, args, stats, best_seed, ckpt_path):
    payload = {
        "model_type": args.model_type,
        "checkpoint_path": str(ckpt_path),
        "episodes": int(args.episodes),
        "seed_base": int(args.seed_base),
        "best_seed": int(best_seed) if best_seed is not None else None,
        "env": {
            "L": float(args.L),
            "dt": float(args.dt),
            "src_x": float(args.src_x),
            "src_y": float(args.src_y),
            "wind_x": float(args.wind_x),
            "wind_y": float(args.wind_y),
            "sigma_c": float(args.sigma_c),
            "r_goal": float(args.r_goal),
        },
        "turbulent_field": {
            "field_size": int(args.field_size),
            "blur_sigma": float(args.blur_sigma),
            "noise_scale": float(args.noise_scale),
            "noise_bias": float(args.noise_bias),
        },
        "stats": stats,
    }
    out_json.write_text(json.dumps(payload, indent=2))


def make_env_from_args(args, render_mode=None):
    return TurbulentEvalEnv(
        render_mode=render_mode,
        L=args.L,
        dt=args.dt,
        src_x=args.src_x,
        src_y=args.src_y,
        wind_x=args.wind_x,
        wind_y=args.wind_y,
        sigma_c=args.sigma_c,
        r_goal=args.r_goal,
        field_size=args.field_size,
        blur_sigma=args.blur_sigma,
        noise_scale=args.noise_scale,
        noise_bias=args.noise_bias,
    )


def resolve_output_paths(args):
    ckpt_path = Path(args.ckpt_path).expanduser().resolve()
    default_out = ckpt_path.parent.parent / "complex_eval"
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = ckpt_path.stem
    return ckpt_path, out_dir, out_dir / f"{stem}_summary.json", out_dir / f"{stem}_trajectories.png", out_dir / f"{stem}_best.gif"


def evaluate(args):
    ckpt_path, out_dir, summary_path, plot_path, gif_path = resolve_output_paths(args)
    device = torch.device("cpu") if args.force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Evaluating on {device}")
    print(f"[Info] Checkpoint: {ckpt_path}")
    print(f"[Info] Outputs: {out_dir}")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    eval_env = make_env_from_args(args)
    policy = load_policy(ckpt_path, args.model_type, eval_env, device, args.connectome_steps)
    trajectories, stats = rollout(eval_env, policy, args.episodes, args.seed_base)

    best_traj = max(trajectories, key=lambda traj: float(traj["return"])) if trajectories else None
    best_seed = int(best_traj["seed"]) if best_traj is not None else None

    print(f"  > Entry Success: {stats['success_entry_rate'] * 100:.1f}%")
    print(f"  > Hold Success:  {stats['success_hold_rate'] * 100:.1f}%")
    print(f"  > Final In-Goal: {stats['final_in_goal_rate'] * 100:.1f}%")
    print(f"  > Avg Return:    {stats['avg_return']:.2f}")
    print(f"  > Avg Cast Starts: {stats['cast_start_count_mean']:.2f} / episode")
    print(f"  > Avg Cast Steps:  {stats['cast_step_count_mean']:.2f} / episode")
    print(f"  > Cast Step %:  {stats['cast_step_ratio_mean'] * 100:.1f}%")
    print(f"  > Can-Turn %:   {stats['can_turn_ratio_mean'] * 100:.1f}%")

    plot_env = make_env_from_args(args)
    plot_seed = best_seed if best_seed is not None else args.seed_base
    plot_env.reset(seed=plot_seed)
    show_trajs_complex(plot_env, trajectories, f"Complex Eval ({args.model_type})", plot_path)
    plot_env.close()

    if args.save_gif and best_seed is not None:
        gif_env = make_env_from_args(args, render_mode="rgb_array")
        gif_policy = load_policy(ckpt_path, args.model_type, gif_env, device, args.connectome_steps)
        generate_gif(gif_env, gif_policy, best_seed, gif_path, f"Complex Eval ({args.model_type})")
        gif_env.close()
    elif not args.save_gif:
        gif_path = None

    save_summary(summary_path, args, stats, best_seed, ckpt_path)
    print(f"[Info] Summary saved to {summary_path}")
    print(f"[Info] Trajectory plot saved to {plot_path}")
    if gif_path is not None:
        print(f"[Info] GIF saved to {gif_path}")


def build_parser():
    p = argparse.ArgumentParser(description="Evaluate a v1 notebook checkpoint on a complex turbulent plume field.")
    p.add_argument("--model-type", choices=["gru", "connectome", "mlp"], required=True)
    p.add_argument("--ckpt-path", required=True, help="Path to a .pt checkpoint saved by a v1 notebook.")
    p.add_argument("--out-dir", default=None, help="Directory for summary/plot/GIF outputs.")
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seed-base", type=int, default=20000)
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--save-gif", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--L", type=float, default=3.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--src-x", type=float, default=0.0)
    p.add_argument("--src-y", type=float, default=0.0)
    p.add_argument("--wind-x", type=float, default=0.0)
    p.add_argument("--wind-y", type=float, default=0.0)
    p.add_argument("--sigma-c", type=float, default=1.0)
    p.add_argument("--r-goal", type=float, default=0.35)

    p.add_argument("--field-size", type=int, default=100)
    p.add_argument("--blur-sigma", type=float, default=5.0)
    p.add_argument("--noise-scale", type=float, default=1.0)
    p.add_argument("--noise-bias", type=float, default=1.0)
    p.add_argument("--connectome-steps", type=int, default=4)
    return p


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
