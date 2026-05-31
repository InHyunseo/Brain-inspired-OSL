"""Evolution-Strategy trainer for the spiking connectome — gradient-free, no critic.

Natural-selection-style learning: perturb the brain's flat parameter vector with
Gaussian noise, evaluate each variant's fitness on the OSL task, and move the
mean toward variants that did well. This sidesteps the failure mode of PPO on the
sparse 6-hop connectome (weak/slow gradients) and is compatible with the
non-differentiable spiking dynamics.

Uses OpenAI-ES tricks: antithetic sampling (mirror each noise to cut variance)
and rank-based fitness shaping (robust to reward outliers). The fitness function
is just env return — the *environment* plays the role of the critic.

Parallel rollouts via a process pool; falls back to serial if disabled.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class ESConfig:
    pop_size: int = 64                 # number of perturbations (must be even for antithetic)
    sigma: float = 0.05                # noise std
    lr: float = 0.03                   # step size for the mean update
    weight_decay: float = 0.005        # L2 pull toward 0 (keeps weights bounded)
    episodes_per_eval: int = 2         # rollouts averaged per candidate (reduce variance)
    max_steps: int = 600               # per-episode cap
    seed: int = 0
    rank_shaping: bool = True          # centered-rank fitness transform
    success_bonus: float = 50.0        # added to return on success (sharpen the signal)


def _centered_ranks(x: np.ndarray) -> np.ndarray:
    """Map fitnesses to centered ranks in [-0.5, 0.5] (OpenAI-ES utility)."""
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[np.argsort(x)] = np.arange(len(x))
    ranks /= (len(x) - 1)
    return ranks - 0.5


class ESTrainer:
    """ES over a flat parameter vector. `make_policy` builds a fresh policy with
    `set_params`/`act`; `make_env` builds an OSL env. Both are zero-arg factories
    so this stays decoupled from concrete classes (and picklable for pools)."""

    def __init__(self, make_policy: Callable, make_env: Callable, cfg: ESConfig):
        self.make_policy = make_policy
        self.make_env = make_env
        self.cfg = cfg
        if cfg.pop_size % 2 != 0:
            raise ValueError("pop_size must be even for antithetic sampling")
        probe = make_policy()
        self.n_params = probe.n_params
        self.theta = probe.get_params().astype(np.float64)
        self.rng = np.random.default_rng(cfg.seed)
        self.generation = 0
        self.history: list[dict] = []

    # ---- fitness of one parameter vector ----
    def evaluate(self, flat: np.ndarray, seed_base: int) -> dict:
        policy = self.make_policy()
        policy.set_params(flat.astype(np.float32))
        rets, succs, dists = [], [], []
        for k in range(self.cfg.episodes_per_eval):
            env = self.make_env()
            obs, _ = env.reset(seed=seed_base + k)
            policy.reset_state()
            ret = 0.0
            info = {}
            for _ in range(self.cfg.max_steps):
                a = policy.act(obs)
                obs, r, term, trunc, info = env.step(a)
                ret += float(r)
                if term or trunc:
                    break
            success = bool(info.get("success", False))
            if success:
                ret += self.cfg.success_bonus
            rets.append(ret)
            succs.append(1.0 if success else 0.0)
            dists.append(float(info.get("distance_to_source_mm", np.nan)))
        return {"fitness": float(np.mean(rets)),
                "success": float(np.mean(succs)),
                "distance": float(np.nanmean(dists))}

    # ---- one ES generation ----
    def step(self, pool=None) -> dict:
        cfg = self.cfg
        half = cfg.pop_size // 2
        eps = self.rng.standard_normal((half, self.n_params))
        noises = np.concatenate([eps, -eps], axis=0)        # antithetic
        gseed = cfg.seed + self.generation * 100003

        candidates = [self.theta + cfg.sigma * noises[i] for i in range(cfg.pop_size)]
        seeds = [gseed + (i % half) * 7 for i in range(cfg.pop_size)]  # mirror pair shares seed

        if pool is not None:
            results = pool.starmap(_eval_static,
                                   [(self.make_policy, self.make_env, cfg, c, s)
                                    for c, s in zip(candidates, seeds)])
        else:
            results = [self.evaluate(c, s) for c, s in zip(candidates, seeds)]

        fit = np.array([r["fitness"] for r in results], dtype=np.float64)
        shaped = _centered_ranks(fit) if cfg.rank_shaping else (fit - fit.mean()) / (fit.std() + 1e-8)

        # ES gradient estimate: (1/(n*sigma)) * sum_i shaped_i * noise_i
        grad = (noises.T @ shaped) / (cfg.pop_size * cfg.sigma)
        self.theta += cfg.lr * grad - cfg.weight_decay * self.theta

        self.generation += 1
        log = {
            "generation": self.generation,
            "fit_mean": float(fit.mean()), "fit_max": float(fit.max()),
            "success_mean": float(np.mean([r["success"] for r in results])),
            "success_max": float(np.max([r["success"] for r in results])),
            "distance_mean": float(np.nanmean([r["distance"] for r in results])),
            "grad_norm": float(np.linalg.norm(grad)),
        }
        self.history.append(log)
        return log

    def get_theta(self) -> np.ndarray:
        return self.theta.copy()

    def train(self, generations: int, pool=None, log_every: int = 1) -> list[dict]:
        for _ in range(generations):
            t0 = time.time()
            log = self.step(pool=pool)
            if log["generation"] % log_every == 0:
                print(f"[gen {log['generation']:4d}] fit {log['fit_mean']:8.2f} "
                      f"(max {log['fit_max']:8.2f}) | success {log['success_mean']*100:5.1f}% "
                      f"(max {log['success_max']*100:4.0f}%) | dist {log['distance_mean']:6.1f}mm "
                      f"| {time.time()-t0:.1f}s")
        return self.history


def _eval_static(make_policy, make_env, cfg, flat, seed_base):
    """Top-level fn so a process pool can pickle it."""
    policy = make_policy()
    policy.set_params(flat.astype(np.float32))
    rets, succs, dists = [], [], []
    for k in range(cfg.episodes_per_eval):
        env = make_env()
        obs, _ = env.reset(seed=seed_base + k)
        policy.reset_state()
        ret = 0.0
        info = {}
        for _ in range(cfg.max_steps):
            a = policy.act(obs)
            obs, r, term, trunc, info = env.step(a)
            ret += float(r)
            if term or trunc:
                break
        success = bool(info.get("success", False))
        if success:
            ret += cfg.success_bonus
        rets.append(ret)
        succs.append(1.0 if success else 0.0)
        dists.append(float(info.get("distance_to_source_mm", np.nan)))
    return {"fitness": float(np.mean(rets)), "success": float(np.mean(succs)),
            "distance": float(np.nanmean(dists))}
