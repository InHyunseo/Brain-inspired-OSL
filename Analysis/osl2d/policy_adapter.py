"""Backbone-agnostic adapter over the 2D OSL policy for analysis.

Wraps a trained PPO :class:`src.models.policy.Policy` or SAC
:class:`src.agents.sac_agent.SACPolicy` behind a single interface the
Jacobian / fixed-point / ablation phases use, mirroring osl_analysis
``policy_adapter.RSACPolicyAdapter``:

    forward(obs_tensor, h_tensor, patch=None) -> (mean, log_std, h_next_flat)
    step_patched(obs_np, h_flat_np, patch=None) -> (action_np, h_next_flat_np)
    .device, .state_size, .action_dim, .group_indices

The actor hidden state is the backbone state ``(state_size,)`` carried across
env steps. ``patch`` follows the connectome ``_apply_patch`` convention
(``{"indices": ..., "value": "zero"|"mean"|"flip"|float}``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.envs.osl_env import ACTION_DIM


def _build_policy_from_ckpt(ckpt_path: str | Path, device: torch.device):
    """Reconstruct a (policy, agent_type) pair from a training checkpoint.

    Reads ``agent_config`` (which now carries ``backbone``/``gru_hidden``) and
    rebuilds the matching policy class, then loads weights.
    """
    from src.models.policy import remap_legacy_backbone_keys

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(payload["agent_config"])
    # Legacy checkpoints stored the connectome under `connectome.*`; the backbone
    # was renamed to `backbone.*`. Remap so old connectome runs still load.
    state = remap_legacy_backbone_keys(payload["policy_state_dict"])
    # SAC checkpoints carry twin-Q critics (q1/q2); PPO carries a `critic` MLP.
    is_sac = any(k.startswith("q1.") for k in state)

    backbone = cfg.get("backbone", "connectome")
    gru_hidden = int(cfg.get("gru_hidden", 421))
    common = dict(
        weights_csv=cfg.get("weights_csv"),
        metadata_csv=cfg.get("metadata_csv"),
        latent_dim=int(cfg.get("latent_dim", 32)),
        message_passing_steps=int(cfg.get("message_passing_steps", 6)),
        backbone=backbone,
        gru_hidden=gru_hidden,
    )
    if is_sac:
        from src.agents.sac_agent import SACPolicy
        policy = SACPolicy(
            critic_hidden=tuple(cfg.get("critic_hidden", (128, 128))),
            log_std_init=float(cfg.get("log_std_init", -0.5)),
            log_std_min=float(cfg.get("log_std_min", -5.0)),
            log_std_max=float(cfg.get("log_std_max", 2.0)),
            **common,
        )
        agent_type = "sac"
    else:
        from src.models.policy import Policy
        policy = Policy(log_std_init=float(cfg.get("log_std_init", -0.5)), **common)
        agent_type = "ppo"
    policy.load_state_dict(state)
    policy.to(device)
    policy.eval()
    return policy, agent_type


class Policy2DAdapter:
    def __init__(self, policy, device: torch.device, agent_type: str = "ppo"):
        self.policy = policy
        self.device = device
        self.agent_type = agent_type
        self.action_dim = ACTION_DIM
        self.state_size = int(policy.actor_state_size)
        self.backbone_kind = getattr(policy, "backbone_kind", "connectome")
        # D-dim connectome: each node carries `feature_dim` channels, so the raw
        # hidden state is (N*D,). For neuron-level analysis we pool the D channels
        # back to one scalar per node via L2 norm. feature_dim==1 (or GRU) => no-op.
        backbone = getattr(policy, "backbone", None)
        self.feature_dim = int(getattr(backbone, "feature_dim", 1) or 1)
        self.n_nodes = int(getattr(backbone, "n_total", self.state_size))

    def pool_hidden(self, h: np.ndarray) -> np.ndarray:
        """Reduce a raw hidden state/sequence to one scalar per neuron.

        `h` is (..., N*D); returns (..., N) by L2 norm over each node's D
        channels. For feature_dim==1 this is identity (up to abs value), which is
        fine — analysis only ever used |activation| anyway. GRU backbones (no
        feature_dim) pass through unchanged.
        """
        if self.feature_dim <= 1:
            return h
        h = np.asarray(h, dtype=np.float32)
        lead = h.shape[:-1]
        hr = h.reshape(*lead, self.n_nodes, self.feature_dim)
        return np.linalg.norm(hr, axis=-1)

    @property
    def node_group_indices(self) -> dict[str, list[int]]:
        """Group indices in *node* space (post-pool), for neuron-level analysis.

        The policy's `group_indices` are in flattened N*D slot space (so live
        ablation can zero whole nodes). After pooling, traces live in node space,
        so divide every slot index by feature_dim and dedup back to node ids.
        """
        if self.feature_dim <= 1:
            return self.policy.group_indices
        D = self.feature_dim
        out: dict[str, list[int]] = {}
        for name, slots in self.policy.group_indices.items():
            nodes = sorted({int(i) // D for i in slots})
            out[name] = nodes
        return out

    @classmethod
    def from_checkpoint(cls, ckpt_path: str | Path, device: str | torch.device | None = None):
        dev = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        policy, agent_type = _build_policy_from_ckpt(ckpt_path, dev)
        return cls(policy, dev, agent_type)

    @property
    def group_indices(self) -> dict[str, list[int]]:
        return self.policy.group_indices

    def _mean_logstd(self, obs: torch.Tensor, h: torch.Tensor, patch=None):
        """Return (mean, log_std, h_next) for a (B, F)/(B, H) batch."""
        mask = torch.ones(obs.shape[0], 1, device=obs.device, dtype=obs.dtype)
        if self.agent_type == "sac":
            mean, log_std, h_next = self.policy._actor_forward(obs, h, mask, patch=patch)
            # SAC squashes with tanh at sampling; for analysis we use the raw mean.
            return mean, log_std, h_next
        dist, h_next = self.policy._actor_distribution(obs, h, mask, patch=patch)
        log_std = self.policy.actor_log_std.expand_as(dist.mean)
        return dist.mean, log_std, h_next

    def forward(self, obs_tensor: torch.Tensor, h_tensor: torch.Tensor, patch=None):
        """Differentiable forward used by the Jacobian. Keeps a batch dim of 1.

        Accepts ``(F,)``/``(H,)`` or ``(B, F)``/``(B, H)``. Returns tensors with
        the batch dim preserved so autograd produces a clean (H, H) Jacobian.
        """
        squeeze = obs_tensor.dim() == 1
        obs = obs_tensor.unsqueeze(0) if squeeze else obs_tensor
        h = h_tensor.unsqueeze(0) if h_tensor.dim() == 1 else h_tensor
        mean, log_std, h_next = self._mean_logstd(obs, h, patch=patch)
        return mean, log_std, h_next

    @torch.no_grad()
    def step_patched(self, obs_np: np.ndarray, h_flat_np: np.ndarray, patch=None):
        """Deterministic single step for online ablation rollouts.

        Returns ``(action_np (A,), h_next_flat_np (H,))``.
        """
        obs = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)).to(self.device).unsqueeze(0)
        h = torch.from_numpy(np.asarray(h_flat_np, dtype=np.float32)).to(self.device).unsqueeze(0)
        mean, _, h_next = self._mean_logstd(obs, h, patch=patch)
        if self.agent_type == "sac":
            action = torch.tanh(mean)
        else:
            action = mean.clamp(-1.0, 1.0)
        return action.squeeze(0).cpu().numpy(), h_next.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def step_stochastic(
        self,
        obs_np: np.ndarray,
        h_flat_np: np.ndarray,
        patch=None,
        generator: torch.Generator | None = None,
    ):
        """Stochastic single step — samples from the policy's action distribution.

        Mirrors how the agent actually acts at training time:
          - SAC: reparameterized squashed-Gaussian, ``tanh(mean + sigma * eps)``.
            The squash must wrap ``mean + sigma*eps`` (NOT ``tanh(mean) + ...``),
            matching ``SACPolicy`` sampling.
          - PPO: ``Normal(mean, sigma).sample()`` then clamp to the action box.

        The hidden-state update is identical to the deterministic path (the
        backbone is driven by ``obs``, not by the sampled action), so the
        returned ``h_next`` is consistent with ``step_patched``. Uses ``mean``
        for the recurrent rollout via the same forward; only the *emitted*
        action is stochastic.

        Returns ``(action_np (A,), h_next_flat_np (H,))``.
        """
        obs = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)).to(self.device).unsqueeze(0)
        h = torch.from_numpy(np.asarray(h_flat_np, dtype=np.float32)).to(self.device).unsqueeze(0)
        mean, log_std, h_next = self._mean_logstd(obs, h, patch=patch)
        std = log_std.exp()
        eps = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
        if self.agent_type == "sac":
            action = torch.tanh(mean + std * eps)
        else:
            action = (mean + std * eps).clamp(-1.0, 1.0)
        return action.squeeze(0).cpu().numpy(), h_next.squeeze(0).cpu().numpy()

    def initial_state(self) -> np.ndarray:
        return np.zeros(self.state_size, dtype=np.float32)
