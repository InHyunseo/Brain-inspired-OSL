"""Vectorized clean-field OSL environment in torch, for GPU-batched ES.

A lightweight reimplementation of OslEnv's CLEAN dynamics (no bump-field noise)
that runs many episodes in parallel on the GPU. Physics and reward terms mirror
`src/envs/osl_env.py` closely enough for ES fitness; it is NOT a drop-in for the
gym env (no spawn-annulus cue zone, no noise stages) — it exists purely to make
evolution-strategy rollouts fast on Colab GPUs.

Shapes: everything is (B,) batched over parallel episodes. Observation per step is
(B, 6) = [c_left, c_right, dlog, prev_v, prev_body_w, prev_head_w] to match the
policy's expected obs layout.
"""
from __future__ import annotations

import math

import torch


class TorchOSLBatch:
    def __init__(self, batch, device="cpu", seed=0,
                 arena_w=80.0, arena_h=120.0, src_x=40.0, src_y=100.0,
                 sigma=30.0, c_peak=1.0, c_bg=0.0, success_radius=7.5,
                 spawn_min_r=55.0, spawn_max_r=70.0, sensor_spacing=0.15,
                 sensor_forward=1.75, dt=0.1, v_max=1.2,
                 body_w_max_deg=120.0, head_w_max_deg=120.0,
                 reward_log_k=0.15, reward_log_clip=0.5, reward_conc_k=0.02,
                 reward_time=-0.005, reward_goal=20.0, wall_penalty=-2.0, eps=1e-6):
        self.B = int(batch); self.device = device
        self.g = torch.Generator(device="cpu"); self.g.manual_seed(seed)
        for k, v in dict(arena_w=arena_w, arena_h=arena_h, src_x=src_x, src_y=src_y,
                         sigma=sigma, c_peak=c_peak, c_bg=c_bg, success_radius=success_radius,
                         spawn_min_r=spawn_min_r, spawn_max_r=spawn_max_r,
                         sensor_spacing=sensor_spacing, sensor_forward=sensor_forward,
                         dt=dt, v_max=v_max, body_w_max=math.radians(body_w_max_deg),
                         head_w_max=math.radians(head_w_max_deg), reward_log_k=reward_log_k,
                         reward_log_clip=reward_log_clip, reward_conc_k=reward_conc_k,
                         reward_time=reward_time, reward_goal=reward_goal,
                         wall_penalty=wall_penalty, eps=eps).items():
            setattr(self, k, v)

    def _conc(self, x, y):
        dx = x - self.src_x; dy = y - self.src_y
        return self.c_bg + self.c_peak * torch.exp(-(dx*dx + dy*dy) / (2.0 * self.sigma**2))

    def _sensors(self, x, y, heading):
        # head tip then lateral split (matches geometry.sensor_positions forward_mm)
        hx = x + torch.cos(heading) * self.sensor_forward
        hy = y + torch.sin(heading) * self.sensor_forward
        perp = heading + math.pi/2
        off = self.sensor_spacing * 0.5
        lx = hx + torch.cos(perp)*off; ly = hy + torch.sin(perp)*off
        rx = hx - torch.cos(perp)*off; ry = hy - torch.sin(perp)*off
        return self._conc(lx, ly), self._conc(rx, ry)

    def reset(self, seed=None):
        if seed is not None:
            self.g.manual_seed(seed)
        B = self.B
        r = (torch.rand(B, generator=self.g) * (self.spawn_max_r - self.spawn_min_r) + self.spawn_min_r)
        th = torch.rand(B, generator=self.g) * (2*math.pi)
        self.x = (self.src_x + r*torch.cos(th)).to(self.device)
        self.y = (self.src_y + r*torch.sin(th)).to(self.device)
        self.heading = (torch.rand(B, generator=self.g)*(2*math.pi)).to(self.device)
        self.head_rel = torch.zeros(B, device=self.device)
        self.prev_v = torch.zeros(B, device=self.device)
        self.prev_bw = torch.zeros(B, device=self.device)
        self.prev_hw = torch.zeros(B, device=self.device)
        self.prev_logavg = None
        self.done = torch.zeros(B, dtype=torch.bool, device=self.device)
        self.steps = 0
        return self._obs(dlog=torch.zeros(B, device=self.device))

    def _obs(self, dlog):
        cL, cR = self._sensors(self.x, self.y, self.heading)
        return torch.stack([cL, cR, dlog, self.prev_v, self.prev_bw, self.prev_hw], dim=1)

    @torch.no_grad()
    def step(self, action):
        a = torch.clamp(action, -1.0, 1.0)
        raw_v, raw_bw, raw_hw = a[:,0], a[:,1], a[:,2]
        v = (raw_v + 1.0)*0.5*self.v_max
        bw = self.body_w_max*raw_bw; hw = self.head_w_max*raw_hw
        live = (~self.done).float()
        self.x = self.x + v*torch.cos(self.heading)*self.dt*live
        self.y = self.y + v*torch.sin(self.heading)*self.dt*live
        self.heading = self.heading + bw*self.dt*live
        self.head_rel = self.head_rel + hw*self.dt*live
        self.prev_v = v/self.v_max; self.prev_bw = raw_bw; self.prev_hw = raw_hw
        self.steps += 1

        cL, cR = self._sensors(self.x, self.y, self.heading)
        cavg = 0.5*(cL+cR)
        logavg = torch.log(cavg + self.eps)
        if self.prev_logavg is None:
            dlog = torch.zeros_like(cavg)
        else:
            dlog = (logavg - self.prev_logavg)/self.dt
        dlog = torch.clamp(dlog, -self.reward_log_clip, self.reward_log_clip)
        self.prev_logavg = logavg

        dist = torch.sqrt((self.x-self.src_x)**2 + (self.y-self.src_y)**2)
        wall = (self.x < 0)|(self.x > self.arena_w)|(self.y < 0)|(self.y > self.arena_h)
        success = dist <= self.success_radius
        newly_done = (success | wall) & (~self.done)

        reward = (self.reward_log_k*dlog
                  + self.reward_conc_k*(cavg/max(self.c_peak,self.eps))
                  + self.reward_time)
        reward = reward + self.reward_goal*success.float() + self.wall_penalty*wall.float()
        reward = reward * (~self.done).float()             # no reward after done

        self.done = self.done | success | wall
        obs = self._obs(dlog)
        return obs, reward, self.done.clone(), success.clone(), dist
