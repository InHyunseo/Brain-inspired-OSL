"""Per-step Jacobian of hidden-state dynamics.

For each timestep t we compute:
    J_t = d h_{t+1} / d h_t   evaluated at (obs_t, h_t)
where the underlying map is `policy.forward(obs_t, h_t)[3]` (h_next).

Sizes: state_size ~ 32 → J_t is 32×32 dense; autograd jacobian is fine.

`block_decompose` partitions J by hidden-state index groups (motor/command/
inter) and returns per-block submatrices and their eigendecompositions.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch.autograd.functional import jacobian as autograd_jacobian


def jacobian_at(policy, obs_t: np.ndarray, h_t: np.ndarray) -> np.ndarray:
    """Compute J_t at a single (obs, h). Returns (state_size, state_size) np array."""
    device = next(policy.parameters()).device
    obs_t_t = torch.from_numpy(obs_t).float().unsqueeze(0).to(device)
    h_t_t = torch.from_numpy(h_t).float().unsqueeze(0).to(device).requires_grad_(False)

    def step(h):
        _, _, _, h_next = policy.forward(obs_t_t, h.unsqueeze(0))
        return h_next.squeeze(0)

    J = autograd_jacobian(step, h_t_t.squeeze(0), create_graph=False, vectorize=True)
    return J.detach().cpu().numpy()


def jacobian_sequence(policy, obs: np.ndarray, h: np.ndarray,
                      stride: int = 1) -> np.ndarray:
    """Compute J at every (obs_t, h_t) along an episode. Returns (T', N, N)."""
    Ts = list(range(0, len(obs) - 1, stride))
    out = np.zeros((len(Ts), h.shape[1], h.shape[1]), dtype=np.float32)
    for i, t in enumerate(Ts):
        out[i] = jacobian_at(policy, obs[t], h[t])
    return out


def block_decompose(J: np.ndarray, group_indices: dict[str, np.ndarray],
                    keep: Iterable[str] = ("motor", "command", "inter")):
    """Slice J into within-group blocks and eigendecompose each.

    Returns: dict[group_name] = {"block": (k,k), "eigvals": (k,), "eigvecs": (k,k)}
    """
    out = {}
    for g in keep:
        idx = group_indices[g]
        if len(idx) == 0:
            continue
        sub = J[np.ix_(idx, idx)]
        w, v = np.linalg.eig(sub)
        out[g] = {"block": sub, "eigvals": w, "eigvecs": v}
    return out


def eig_full(J: np.ndarray):
    w, v = np.linalg.eig(J)
    return w, v


def dominant_summary(eigvals: np.ndarray) -> dict:
    """Single-Jacobian summary: top |λ|, slow (real, |λ|>0.9), oscillatory (complex)."""
    abs_w = np.abs(eigvals)
    top_idx = int(np.argmax(abs_w))
    top = eigvals[top_idx]
    is_complex = abs(top.imag) > 1e-6
    return {
        "top_abs": float(abs_w[top_idx]),
        "top_real": float(top.real),
        "top_imag": float(top.imag),
        "top_arg": float(np.angle(top)),  # radians per step
        "is_oscillatory": bool(is_complex),
        "n_slow_real": int(np.sum((np.abs(eigvals.imag) < 1e-6) & (abs_w > 0.9))),
        "n_oscillatory": int(np.sum(np.abs(eigvals.imag) > 1e-6) // 2),
    }
