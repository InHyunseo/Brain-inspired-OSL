"""GPU-batched Evolution Strategy for the spiking connectome (E2E motor policy).

Evaluates the whole ES population in parallel on the GPU: P candidate parameter
vectors x B parallel episodes = P*B simultaneous rollouts, one big batched forward
per env step. No backprop, no critic — fitness is env return; the environment is
the selection pressure.

Policy = torch batched LIF connectome. Each of the P candidates has its own synapse
strengths; we broadcast over the B episodes it is evaluated on. Uses antithetic
sampling + centered-rank shaping (OpenAI-ES).

Requires: src/brain/torch_osl_batch.py (vectorized clean-field env).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.models.connectome import load_connectome_layout
from src.brain.torch_osl_batch import TorchOSLBatch


@dataclass
class ESGpuConfig:
    pop_size: int = 128          # candidates (even, antithetic)
    episodes_per_cand: int = 4   # parallel episodes per candidate
    sigma: float = 0.05
    lr: float = 0.03
    weight_decay: float = 0.005
    max_steps: int = 600
    inner_steps: int = 4
    n_output_nodes: int = 8
    leak: float = 0.9
    v_threshold: float = 0.5
    spike_temp: float = 0.25
    input_gain: float = 3.0
    success_bonus: float = 50.0
    seed: int = 0


class BatchedSpikingConnectome:
    """Vectorized LIF connectome whose parameters carry a leading P (candidate)
    axis. forward maps (P, M, obs) -> (P, M, action) where M = episodes/candidate.
    All weights live on `device`."""

    def __init__(self, cfg: ESGpuConfig, device,
                 weights_csv="assets/connectome/weights.csv",
                 metadata_csv="assets/connectome/metadata.csv", obs_dim=6, action_dim=3):
        conn, layout = load_connectome_layout(weights_csv, metadata_csv, cfg.n_output_nodes)
        self.cfg = cfg; self.device = device
        self.N = layout.total_node_count
        self.obs_dim = obs_dim; self.action_dim = action_dim
        n = self.N
        mask = np.zeros((n, n), dtype=np.float32)
        mask[:layout.base_node_count, :layout.base_node_count] = conn
        if layout.left_orn_indices:  mask[layout.left_sensor_index, layout.left_orn_indices] = 1.0
        if layout.right_orn_indices: mask[layout.right_sensor_index, layout.right_orn_indices] = 1.0
        if layout.mbon_indices:
            for oi in layout.output_indices: mask[layout.mbon_indices, oi] = 1.0
        src, tgt = np.nonzero(mask)
        self.src = torch.as_tensor(src, dtype=torch.long, device=device)
        self.tgt = torch.as_tensor(tgt, dtype=torch.long, device=device)
        self.output_idx = torch.as_tensor(layout.output_indices, dtype=torch.long, device=device)
        self.left_sensor = layout.left_sensor_index
        self.right_sensor = layout.right_sensor_index
        self.E = int(self.src.numel()); self.n_out = int(self.output_idx.numel())

        # parameter slices (flat, per candidate)
        self.shapes = [("w_edge", (self.E,)), ("b_node", (n,)),
                       ("w_in", (obs_dim, n)), ("w_out", (self.n_out, action_dim)),
                       ("b_out", (action_dim,))]
        self.n_params = sum(int(np.prod(s)) for _, s in self.shapes)

    def init_theta(self, seed=0):
        rng = np.random.default_rng(seed)
        scale = 1.0/np.sqrt(max(1, self.E/self.N))
        parts = [rng.standard_normal(self.E)*scale, np.zeros(self.N),
                 rng.standard_normal(self.obs_dim*self.N)*0.1,
                 rng.standard_normal(self.n_out*self.action_dim)*0.3,
                 np.zeros(self.action_dim)]
        return np.concatenate(parts).astype(np.float64)

    def _unpack(self, theta_pb):
        # theta_pb: (P, n_params) tensor -> dict of (P, ...) params
        out = {}; i = 0
        for name, shape in self.shapes:
            k = int(np.prod(shape))
            out[name] = theta_pb[:, i:i+k].reshape(theta_pb.shape[0], *shape); i += k
        return out

    @torch.no_grad()
    def rollout_fitness(self, theta_pb, env_seed):
        """theta_pb: (P, n_params) on device. Returns fitness (P,)."""
        cfg = self.cfg; P = theta_pb.shape[0]; M = cfg.episodes_per_cand
        prm = self._unpack(theta_pb)
        env = TorchOSLBatch(P*M, device=self.device, seed=env_seed,
                            success_radius=7.5, sigma=30.0)
        obs = env.reset(seed=env_seed)                       # (P*M, 6)
        v = torch.zeros(P*M, self.N, device=self.device)
        s = torch.zeros(P*M, self.N, device=self.device)
        ret = torch.zeros(P*M, device=self.device)
        ever_success = torch.zeros(P*M, dtype=torch.bool, device=self.device)

        # expand per-candidate params to per-rollout (P*M)
        def ex(x): return x.repeat_interleave(M, dim=0)
        w_edge = ex(prm["w_edge"]); b_node = ex(prm["b_node"])
        w_in = ex(prm["w_in"]); w_out = ex(prm["w_out"]); b_out = ex(prm["b_out"])

        for _ in range(cfg.max_steps):
            drive = torch.bmm(obs.unsqueeze(1), w_in).squeeze(1)        # (P*M, N)
            drive[:, self.left_sensor] += obs[:, 0]
            drive[:, self.right_sensor] += obs[:, 1]
            for _ in range(cfg.inner_steps):
                msg = s[:, self.src]*w_edge                            # (P*M, E)
                inp = torch.zeros_like(v).index_add_(1, self.tgt, msg)
                inp = inp + drive + b_node
                v = cfg.leak*v + inp
                s = torch.sigmoid((v - cfg.v_threshold)/cfg.spike_temp)
                v = v - s*cfg.v_threshold
            rate = s[:, self.output_idx]                              # (P*M, n_out)
            act = torch.tanh(torch.bmm(rate.unsqueeze(1), w_out).squeeze(1) + b_out)
            obs, rew, done, success, dist = env.step(act)
            ret = ret + rew
            ever_success = ever_success | success
        ret = ret + cfg.success_bonus*ever_success.float()           # bonus for reaching source
        fit = ret.reshape(P, M).mean(dim=1)
        succ = ever_success.reshape(P, M).float().mean(dim=1)        # true success rate
        return fit, succ


def _centered_ranks(x):
    r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x)); r /= (len(x)-1); return r-0.5


class ESGpu:
    def __init__(self, cfg: ESGpuConfig, device):
        self.cfg = cfg; self.device = device
        self.policy = BatchedSpikingConnectome(cfg, device)
        self.theta = self.policy.init_theta(cfg.seed)
        self.n_params = self.policy.n_params
        self.rng = np.random.default_rng(cfg.seed)
        self.generation = 0; self.history = []
        if cfg.pop_size % 2: raise ValueError("pop_size must be even")

    def step(self):
        cfg = self.cfg; half = cfg.pop_size//2
        eps = self.rng.standard_normal((half, self.n_params))
        noises = np.concatenate([eps, -eps], 0)                       # (P, n_params)
        cand = self.theta[None, :] + cfg.sigma*noises                # (P, n_params)
        theta_pb = torch.tensor(cand, dtype=torch.float32, device=self.device)
        fit, succ = self.policy.rollout_fitness(theta_pb, env_seed=cfg.seed + self.generation*101)
        succ = succ.cpu().numpy()
        fit = fit.cpu().numpy().astype(np.float64)
        shaped = _centered_ranks(fit)
        grad = (noises.T @ shaped)/(cfg.pop_size*cfg.sigma)
        self.theta = self.theta + cfg.lr*grad - cfg.weight_decay*self.theta
        self.generation += 1
        log = {"generation": self.generation, "fit_mean": float(fit.mean()),
               "fit_max": float(fit.max()), "success_mean": float(succ.mean()),
               "success_max": float(succ.max())}
        self.history.append(log); return log

    def train(self, generations, log_every=1):
        import time
        for _ in range(generations):
            t0 = time.time(); log = self.step()
            if log["generation"] % log_every == 0:
                print(f"[gen {log['generation']:4d}] fit_mean {log['fit_mean']:8.2f} "
                      f"fit_max {log['fit_max']:8.2f} | success {log['success_mean']*100:4.1f}% "
                      f"(max {log['success_max']*100:3.0f}%) | {time.time()-t0:.1f}s")
        return self.history
