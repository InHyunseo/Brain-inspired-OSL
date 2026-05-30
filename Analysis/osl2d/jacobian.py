"""Per-step Jacobian of hidden-state dynamics.

For each (obs_t, h_t):  J_t = d h_{t+1} / d h_t, where the underlying map is
``adapter.forward(obs_t, h_t)`` returning ``(mean, log_std, h_next)``.

Ported from osl_analysis ``utils/jacobian.py`` — sim-agnostic; works with the
:class:`Analysis.osl2d.policy_adapter.Policy2DAdapter` interface.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.autograd.functional import jacobian as autograd_jacobian


def jacobian_at(adapter, obs_t: np.ndarray, h_t: np.ndarray) -> np.ndarray:
    """Compute J_t at a single (obs, h). Returns (state_size, state_size)."""
    device = adapter.device
    obs_tensor = torch.from_numpy(np.asarray(obs_t, dtype=np.float32)).to(device)
    h_tensor = torch.from_numpy(np.asarray(h_t, dtype=np.float32)).to(device)

    def step(h_vec):
        _, _, h_next = adapter.forward(obs_tensor, h_vec)
        return h_next

    J = autograd_jacobian(step, h_tensor, create_graph=False, vectorize=True)
    J_np = J.detach().cpu().numpy()
    # autograd can return (1, H, 1, H) when forward keeps the batch dim.
    while J_np.ndim > 2:
        J_np = J_np.squeeze()
    if J_np.ndim == 1:
        J_np = np.diag(J_np)
    return J_np


def jacobian_sequence(adapter, obs: np.ndarray, h: np.ndarray, stride: int = 1) -> np.ndarray:
    """Compute J at every (obs_t, h_t) along an episode. Returns (T', N, N)."""
    T = len(obs)
    idx = list(range(0, max(T - 1, 0), stride))
    if not idx:
        return np.zeros((0, h.shape[1], h.shape[1]), dtype=np.float32)
    out = np.zeros((len(idx), h.shape[1], h.shape[1]), dtype=np.float32)
    for k, t in enumerate(idx):
        out[k] = jacobian_at(adapter, obs[t], h[t])
    return out


def block_decompose(J: np.ndarray, group_indices: dict) -> dict:
    out = {}
    for name, idx in group_indices.items():
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            continue
        sub = J[np.ix_(idx, idx)]
        w, v = np.linalg.eig(sub)
        out[name] = {"block": sub, "eigvals": w, "eigvecs": v}
    return out


def dominant_summary(eigvals: np.ndarray) -> dict:
    eigvals = np.asarray(eigvals).ravel()
    if eigvals.size == 0:
        return {
            "top_abs": 0.0, "top_real": 0.0, "top_imag": 0.0, "top_arg": 0.0,
            "is_oscillatory": False, "n_slow_real": 0, "n_oscillatory": 0,
        }
    abs_w = np.abs(eigvals)
    top = eigvals[int(np.argmax(abs_w))]
    is_complex = abs(top.imag) > 1e-6
    return {
        "top_abs": float(np.abs(top)),
        "top_real": float(top.real),
        "top_imag": float(top.imag),
        "top_arg": float(np.angle(top)),
        "is_oscillatory": bool(is_complex),
        "n_slow_real": int(np.sum((np.abs(eigvals.imag) < 1e-6) & (abs_w > 0.9))),
        "n_oscillatory": int(np.sum(np.abs(eigvals.imag) > 1e-6) // 2),
    }
