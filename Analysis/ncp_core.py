"""NCP-based actor-critic for POMDP Pendulum.

This module defines the *standard analysis interface* that Phase 1-3 scripts
depend on. When migrating to the OSL connectome policy later, only this file
(and `pendulum_pomdp.py`) needs to be replaced; the phase scripts consume the
same surface:

  forward(obs, h, patch=None) -> (action_mean, action_logstd, value, h_next)
  group_indices: dict[str, np.ndarray]   # hidden-state index sets per group
  step_forward(obs, h, patch=None)       # convenience: 1-step batched eval

Group semantics (AutoNCP):
- 'motor'   : output neurons (== first `output_size` units of hidden state)
- 'command' : command-layer units
- 'inter'   : inter-layer units
- 'sensory' : NOT a hidden-state group — these are input feature indices
              (kept for completeness, but never used as a hidden mask).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ncps.torch import CfC
from ncps.wirings import AutoNCP


@dataclass
class NCPCoreConfig:
    obs_dim: int = 2
    action_dim: int = 1
    units: int = 32
    output_size: int = 8           # AutoNCP motor neurons; we map to action via head
    sparsity_level: float = 0.5
    log_std_init: float = -0.5
    wiring_seed: int = 12345


def _apply_patch(h: torch.Tensor, patch: Optional[dict]) -> torch.Tensor:
    """Apply activation patching to hidden state.

    patch: None or dict with keys
        indices: 1D LongTensor of hidden-state positions to modify
        value:   "zero" | "mean" | "flip" | float | Tensor
    The op is differentiable-safe (uses index_copy / arithmetic, no in-place
    on a leaf). Returns a new tensor.
    """
    if patch is None:
        return h
    idx = patch["indices"]
    if not torch.is_tensor(idx):
        idx = torch.as_tensor(idx, dtype=torch.long, device=h.device)
    val = patch["value"]
    h_new = h.clone()
    sel = h_new.index_select(1, idx)
    if val == "zero":
        rep = torch.zeros_like(sel)
    elif val == "mean":
        rep = sel.mean(dim=1, keepdim=True).expand_as(sel)
    elif val == "flip":
        rep = -sel
    elif torch.is_tensor(val):
        rep = val.to(h.device).expand_as(sel)
    else:
        rep = torch.full_like(sel, float(val))
    h_new[:, idx] = rep
    return h_new


class NCPCore(nn.Module):
    """CfC + AutoNCP actor-critic with patch hook and group metadata."""

    def __init__(self, cfg: NCPCoreConfig | None = None):
        super().__init__()
        self.cfg = cfg or NCPCoreConfig()
        c = self.cfg

        self.wiring = AutoNCP(
            units=c.units,
            output_size=c.output_size,
            sparsity_level=c.sparsity_level,
            seed=c.wiring_seed,
        )
        # CfC owns a single recurrent backbone; we attach two MLP heads
        # (actor mean, critic value) downstream of motor-neuron outputs.
        self.backbone = CfC(input_size=c.obs_dim, units=self.wiring, batch_first=True)
        # `wiring.build(input_size)` is called inside CfC's __init__.
        self.actor_head = nn.Linear(c.output_size, c.action_dim)
        self.critic_head = nn.Linear(c.output_size, 1)
        self.log_std = nn.Parameter(torch.full((c.action_dim,), c.log_std_init))

        # Cache group indices into hidden state.
        # AutoNCP layout: motor (0..output_size-1) | command (...) | inter (...)
        self._group_indices = {
            "motor": np.asarray(self.wiring._motor_neurons, dtype=np.int64),
            "command": np.asarray(self.wiring._command_neurons, dtype=np.int64),
            "inter": np.asarray(self.wiring._inter_neurons, dtype=np.int64),
            # Sensory refers to *input feature indices*, not hidden state:
            "sensory_input": np.asarray(self.wiring._sensory_neurons, dtype=np.int64),
        }

    @property
    def state_size(self) -> int:
        return self.backbone.state_size

    @property
    def group_indices(self) -> dict:
        return self._group_indices

    def initial_state(self, batch_size: int, device=None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        return torch.zeros(batch_size, self.state_size, device=device)

    def forward(
        self,
        obs: torch.Tensor,           # (B, F)
        h: torch.Tensor,             # (B, state_size)
        patch: Optional[dict] = None,
    ):
        """One-step batched forward.

        Returns:
            action_mean: (B, action_dim)
            action_logstd: (action_dim,)  broadcast
            value: (B, 1)
            h_next: (B, state_size)
        """
        if obs.dim() == 2:
            x = obs.unsqueeze(1)  # (B, 1, F)
        else:
            x = obs
        y, h_next = self.backbone(x, h)   # y: (B, 1, output_size)
        h_next = _apply_patch(h_next, patch)
        motor = y[:, -1, :]               # (B, output_size)
        action_mean = self.actor_head(motor)
        value = self.critic_head(motor)
        return action_mean, self.log_std, value, h_next

    @torch.no_grad()
    def act(self, obs: torch.Tensor, h: torch.Tensor, deterministic: bool = False,
            patch: Optional[dict] = None):
        mean, log_std, value, h_next = self.forward(obs, h, patch=patch)
        if deterministic:
            action = mean
            log_prob = torch.zeros(mean.shape[0], device=mean.device)
        else:
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value.squeeze(-1), h_next

    def evaluate_actions(self, obs_seq: torch.Tensor, h0: torch.Tensor,
                         actions: torch.Tensor):
        """For PPO: evaluate stored (obs_seq, actions) under current policy.

        obs_seq: (B, T, F), h0: (B, state_size), actions: (B, T, action_dim)
        Returns log_probs (B, T), values (B, T), entropy (B, T).
        """
        y, _ = self.backbone(obs_seq, h0)        # (B, T, output_size)
        motor = y
        mean = self.actor_head(motor)            # (B, T, action_dim)
        value = self.critic_head(motor).squeeze(-1)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_probs, value, entropy


def action_to_env(action: torch.Tensor) -> np.ndarray:
    """Map policy action (unbounded) to Pendulum action range [-2, 2]."""
    return (2.0 * torch.tanh(action)).detach().cpu().numpy()
