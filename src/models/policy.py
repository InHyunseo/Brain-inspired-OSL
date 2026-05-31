"""Actor-critic policy (backbone-agnostic).

Actor backbone is selectable via `backbone=`:

- ``"connectome"`` — `Connectome` consumes sensor channels (obs[:, 0:2]); its
  latent is concatenated with the dlog + efference-copy channels (obs[:, 2:6])
  before a Linear head emits the Gaussian mean.
- ``"gru"`` — `GRUBackbone` consumes the full 6-D obs; its hidden state is the
  latent and the head consumes it directly (no re-concat).

Critic: stateless MLP over the full 6-D obs producing a scalar value.

Sequence operations follow the convention: tensors are `(T, B, D)` for both
`obs_seq` and `mask_seq`; states are `(B, state_size)` carried across env steps
and zeroed via mask at episode boundaries.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.distributions import Normal

from src.envs.osl_env import ACTION_DIM, OBS_DIM
from src.models.connectome import Connectome
from src.models.gru_backbone import GRUBackbone


SENSOR_INDICES = (0, 1)
# dlog (idx 2) + efference copy v/body_omega/head_omega (idx 3,4,5). dlog is a
# temporal-gradient cue and the efference channels let the head interpret its
# own cast (head_omega) when reading dlog. Re-concatenated onto the connectome
# latent before the actor head; the sensor-only connectome backbone never sees
# these directly.
HEAD_EXTRA_INDICES = (2, 3, 4, 5)


def _gather(obs: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor:
    return obs[..., list(indices)]


def remap_legacy_backbone_keys(state: dict) -> dict:
    """Rename legacy ``connectome.*`` state-dict keys to ``backbone.*``.

    Checkpoints saved before the backbone abstraction stored the connectome
    under ``connectome.*``; the actor backbone is now ``backbone.*``. Returns a
    possibly-new dict (input unchanged) so old connectome runs still load.
    """
    if not any(k.startswith("connectome.") for k in state):
        return state
    return {
        ("backbone." + k[len("connectome."):] if k.startswith("connectome.") else k): v
        for k, v in state.items()
    }


class Policy(nn.Module):
    def __init__(
        self,
        weights_csv: str | Path | None = None,
        metadata_csv: str | Path | None = None,
        latent_dim: int = 32,
        message_passing_steps: int = 6,
        critic_hidden: tuple[int, ...] = (64, 64),
        log_std_init: float = -0.5,
        backbone: str = "connectome",
        gru_hidden: int = 421,
    ):
        super().__init__()
        self.backbone_kind = str(backbone)
        if self.backbone_kind == "connectome":
            if weights_csv is None or metadata_csv is None:
                raise ValueError("connectome backbone requires weights_csv and metadata_csv")
            self.backbone = Connectome(
                weights_csv=weights_csv,
                metadata_csv=metadata_csv,
                latent_dim=latent_dim,
                message_passing_steps=message_passing_steps,
                activation="tanh",
            )
            head_in_dim = self.backbone.latent_dim + len(HEAD_EXTRA_INDICES)
        elif self.backbone_kind == "gru":
            self.backbone = GRUBackbone(input_size=OBS_DIM, hidden=gru_hidden)
            head_in_dim = self.backbone.latent_dim
        else:
            raise ValueError(f"Unknown backbone {backbone!r}; expected 'connectome' or 'gru'")

        self.actor_mean = nn.Linear(head_in_dim, ACTION_DIM)
        self.actor_log_std = nn.Parameter(torch.full((ACTION_DIM,), float(log_std_init)))

        critic_layers: list[nn.Module] = []
        last = OBS_DIM
        for h in critic_hidden:
            critic_layers.append(nn.Linear(last, h))
            critic_layers.append(nn.Tanh())
            last = h
        critic_layers.append(nn.Linear(last, 1))
        self.critic = nn.Sequential(*critic_layers)

        self.actor_state_size = self.backbone.state_size
        self.critic_state_size = 0  # critic is stateless; placeholder kept for buffer/runner symmetry

    @property
    def group_indices(self) -> dict[str, list[int]]:
        return self.backbone.group_indices

    def _backbone_input(self, obs: torch.Tensor) -> torch.Tensor:
        """Slice the obs the backbone consumes (sensor-only for connectome, full for gru)."""
        if self.backbone_kind == "connectome":
            return _gather(obs, SENSOR_INDICES)
        return obs

    def _head_input(self, latent: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Compose the actor-head input from the backbone latent."""
        if self.backbone_kind == "connectome":
            head_extra = _gather(obs, HEAD_EXTRA_INDICES)
            return torch.cat([latent, head_extra], dim=-1)
        return latent

    def actor_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.backbone.parameters()
        yield from self.actor_mean.parameters()
        yield self.actor_log_std

    def critic_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.critic.parameters()

    def initial_states(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actor_state = self.backbone.initial_state(batch_size, device)
        critic_state = torch.zeros(batch_size, 0, device=device)
        return actor_state, critic_state

    def _actor_distribution(
        self,
        obs: torch.Tensor,
        actor_state: torch.Tensor,
        mask: torch.Tensor,
        patch: dict[str, Any] | None = None,
    ):
        latent, next_actor_state = self.backbone.forward_step(
            self._backbone_input(obs), actor_state, mask, patch=patch
        )
        head_in = self._head_input(latent, obs)
        mean = self.actor_mean(head_in)
        std = self.actor_log_std.exp().expand_as(mean)
        return Normal(mean, std), next_actor_state

    def act(
        self,
        obs: torch.Tensor,
        actor_state: torch.Tensor,
        critic_state: torch.Tensor,
        mask: torch.Tensor,
        deterministic: bool = False,
        patch: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, next_actor_state = self._actor_distribution(obs, actor_state, mask, patch=patch)
        action = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        value = self.critic(obs)
        return action, log_prob, value, next_actor_state, critic_state

    def predict_value(
        self, obs: torch.Tensor, critic_state: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.critic(obs), critic_state

    def evaluate_actions_sequence(
        self,
        obs_seq: torch.Tensor,
        mask_seq: torch.Tensor,
        action_seq: torch.Tensor,
        actor_state0: torch.Tensor,
        critic_state0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_seq, _ = self.backbone.forward_sequence(
            self._backbone_input(obs_seq), actor_state0, mask_seq
        )
        head_in_seq = self._head_input(latent_seq, obs_seq)

        steps, batch = obs_seq.shape[:2]
        flat_head_in = head_in_seq.reshape(steps * batch, -1)
        flat_actions = action_seq.reshape(steps * batch, -1)
        flat_obs = obs_seq.reshape(steps * batch, -1)

        mean = self.actor_mean(flat_head_in)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(flat_actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        values = self.critic(flat_obs)

        return (
            values.reshape(steps, batch, 1),
            log_prob.reshape(steps, batch, 1),
            entropy.reshape(steps, batch, 1),
        )
