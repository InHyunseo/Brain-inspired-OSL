"""Generate a supervised valence dataset and train the connectome to regress dlog.

Question: can the connectome, given only the bilateral sensor stream, reconstruct
dlog (the temporal up-gradient signal)? We compare against two baselines to make
the result meaningful:
  - const-0       : predict 0 always (MSE = variance of dlog) — the trivial floor.
  - instantaneous : a linear map of the current (c_L, c_R, c_L-c_R) — can a purely
                    spatial readout fake the temporal derivative? (no memory)
The connectome carries membrane potential across time, so it *can* form a temporal
derivative; the baselines cannot. Beating them shows the recurrence is doing real
temporal integration.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.envs.osl_env import OslEnv, EnvConfig


def make_dataset(n_episodes=400, ep_len=120, seed=0, env_kw=None):
    """Roll out a random policy; collect (sensor_seq (T,2), dlog_seq (T,)) per ep."""
    env_kw = env_kw or dict(success_radius_mm=7.5, gaussian_sigma_mm=30.0, episode_seconds=120.0)
    rng = np.random.default_rng(seed)
    S, D = [], []
    for e in range(n_episodes):
        env = OslEnv(EnvConfig.from_dict(env_kw))
        obs, _ = env.reset(seed=seed + e)
        sens, dl = [], []
        for _ in range(ep_len):
            a = rng.uniform(-1, 1, 3).astype(np.float32)
            obs, r, term, trunc, info = env.step(a)
            sens.append(obs[:2].copy()); dl.append(float(obs[2]))
            if term or trunc:
                break
        if len(sens) >= 8:
            S.append(np.asarray(sens, dtype=np.float32))
            D.append(np.asarray(dl, dtype=np.float32))
    return S, D


def _pad(seqs, T):
    out = np.zeros((len(seqs), T, seqs[0].shape[-1] if seqs[0].ndim > 1 else 1), dtype=np.float32)
    mask = np.zeros((len(seqs), T), dtype=np.float32)
    for i, s in enumerate(seqs):
        L = min(len(s), T)
        out[i, :L] = s[:L].reshape(L, -1)
        mask[i, :L] = 1.0
    return out, mask


def to_tensors(S, D, T=120, device="cpu"):
    Sx, m = _pad(S, T)
    Dy, _ = _pad([d.reshape(-1, 1) for d in D], T)
    return (torch.tensor(Sx, device=device), torch.tensor(Dy, device=device),
            torch.tensor(m, device=device))


def masked_mse(pred, target, mask):
    m = mask.unsqueeze(-1)
    return ((pred - target) ** 2 * m).sum() / m.sum().clamp(min=1)


def baseline_scores(Sx, Dy, mask):
    """const-0 and best instantaneous-linear MSE (closed-form ridge on [cL,cR,cL-cR,1])."""
    m = mask.bool()
    y = Dy[..., 0][m].numpy()
    var = float(np.mean(y ** 2))                          # const-0 MSE
    cL, cR = Sx[..., 0][m].numpy(), Sx[..., 1][m].numpy()
    X = np.stack([cL, cR, cL - cR, np.ones_like(cL)], 1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    inst = float(np.mean((X @ coef - y) ** 2))            # instantaneous-linear MSE
    return {"const0_mse": var, "instantaneous_mse": inst}


def train(model, Sx, Dy, mask, epochs=60, lr=1e-2, batch=64, val_frac=0.2, seed=0, verbose=True):
    n = Sx.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = int(n * val_frac)
    vi, ti = perm[:n_val], perm[n_val:]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = float("inf"); hist = []
    for ep in range(epochs):
        model.train()
        bperm = ti[torch.randperm(len(ti), generator=g)]
        for j in range(0, len(bperm), batch):
            idx = bperm[j:j + batch]
            opt.zero_grad()
            pred = model(Sx[idx])
            loss = masked_mse(pred, Dy[idx], mask[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = masked_mse(model(Sx[vi]), Dy[vi], mask[vi]).item()
        best = min(best, vloss); hist.append(vloss)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}  val_mse {vloss:.5f}  (best {best:.5f})")
    return {"best_val_mse": best, "history": hist}
