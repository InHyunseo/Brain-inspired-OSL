"""Actor-critic policy (backbone-agnostic).

Actor backbone is selectable via `backbone=`:

- ``"connectome"`` — `Connectome` consumes sensor channels (obs[:, 0:2]); its
  latent is concatenated with the dlog + efference-copy channels (obs[:, 2:6])
  before a Linear head emits the Gaussian mean.
- ``"gru"`` — `GRUBackbone` consumes the full 6-D obs; its hidden state is the
  latent and the head consumes it directly (no re-concat).

Critic: selectable via `critic_type=`:

- ``"mlp"`` — stateless 2-layer MLP over the full 6-D obs.
- ``"recurrent"`` — a single `nn.GRUCell` over the full 6-D obs followed by a
  Linear value head.

The critic is a separate parallel network (it does NOT share the actor
backbone). It is discarded at deployment; only the actor backbone is the
"biological" model.

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


def _normalize_hidden_sizes(hidden: int | str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(hidden, int):
        return (int(hidden),)
    if isinstance(hidden, str):
        return tuple(int(part.strip()) for part in hidden.split(",") if part.strip())
    return tuple(int(width) for width in hidden)


def _normalize_critic_type(critic_type: str) -> str:
    critic = str(critic_type).lower()
    if critic in {"mlp", "stateless", "feedforward", "ff"}:
        return "mlp"
    if critic in {"recurrent", "gru", "rnn"}:
        return "recurrent"
    raise ValueError(f"Unknown critic_type {critic_type!r}; expected 'mlp' or 'recurrent'")


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


class CriticGRU(nn.Module):
    """Recurrent value network: a single GRUCell over obs + a Linear value head.

    Parallel to (not sharing) the actor backbone. Mirrors the backbone's
    step/sequence interface so the policy can carry a critic hidden state the
    same way it carries the actor state (zeroed at episode boundaries via mask).
    """

    def __init__(self, obs_dim: int, hidden: int):
        super().__init__()
        self.hidden = int(hidden)
        self.cell = nn.GRUCell(input_size=int(obs_dim), hidden_size=self.hidden)
        self.value_head = nn.Linear(self.hidden, 1)
        self.state_size = self.hidden

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.state_size, device=device)

    def forward_step(
        self, obs: torch.Tensor, state: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_next = self.cell(obs, state * mask)
        return self.value_head(h_next), h_next

    def forward_sequence(
        self, obs_seq: torch.Tensor, state0: torch.Tensor, mask_seq: torch.Tensor
    ) -> torch.Tensor:
        state = state0
        values = []
        for step in range(obs_seq.shape[0]):
            value, state = self.forward_step(obs_seq[step], state, mask_seq[step])
            values.append(value)
        return torch.stack(values, dim=0)


class CriticMLP(nn.Sequential):
    """Stateless value network: MLP over obs with the critic-state API shim.

    The rest of PPO carries a critic state tensor so recurrent critics can train
    through sequences. For the MLP critic this state is a one-column zero
    placeholder and is intentionally ignored.
    """

    def __init__(self, obs_dim: int, hidden: Iterable[int]):
        widths = _normalize_hidden_sizes(hidden)
        layers: list[nn.Module] = []
        in_dim = int(obs_dim)
        for width in widths:
            layers.extend([nn.Linear(in_dim, int(width)), nn.Tanh()])
            in_dim = int(width)
        layers.append(nn.Linear(in_dim, 1))
        super().__init__(*layers)
        self.state_size = 1

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.state_size, device=device)

    def forward_step(
        self, obs: torch.Tensor, state: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del mask
        return self(obs), torch.zeros_like(state)

    def forward_sequence(
        self, obs_seq: torch.Tensor, state0: torch.Tensor, mask_seq: torch.Tensor
    ) -> torch.Tensor:
        del state0, mask_seq
        steps, batch = obs_seq.shape[:2]
        flat_obs = obs_seq.reshape(steps * batch, -1)
        return self(flat_obs).reshape(steps, batch, 1)


class Policy(nn.Module):
    def __init__(
        self,
        weights_csv: str | Path | None = None,
        metadata_csv: str | Path | None = None,
        latent_dim: int = 32,
        message_passing_steps: int = 6,
        critic_hidden: int | str | Iterable[int] = (64, 64),
        critic_type: str = "recurrent",
        log_std_init: float = -0.5,
        backbone: str = "connectome",
        gru_hidden: int = 421,
        feature_dim: int = 8,
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
                feature_dim=feature_dim,
            )
            head_in_dim = self.backbone.latent_dim + len(HEAD_EXTRA_INDICES)
        elif self.backbone_kind == "gru":
            self.backbone = GRUBackbone(input_size=OBS_DIM, hidden=gru_hidden)
            head_in_dim = self.backbone.latent_dim
        else:
            raise ValueError(f"Unknown backbone {backbone!r}; expected 'connectome' or 'gru'")

        self.actor_mean = nn.Linear(head_in_dim, ACTION_DIM)
        self.actor_log_std = nn.Parameter(torch.full((ACTION_DIM,), float(log_std_init)))

        self.critic_type = _normalize_critic_type(critic_type)
        critic_hidden = _normalize_hidden_sizes(critic_hidden)
        if self.critic_type == "mlp":
            self.critic = CriticMLP(obs_dim=OBS_DIM, hidden=critic_hidden)
        else:
            # Recurrent critic uses the first hidden width as its GRU state size.
            critic_gru_hidden = int(critic_hidden[0]) if critic_hidden else 64
            self.critic = CriticGRU(obs_dim=OBS_DIM, hidden=critic_gru_hidden)

        self.actor_state_size = self.backbone.state_size
        self.critic_state_size = self.critic.state_size

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
        critic_state = self.critic.initial_state(batch_size, device)
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
        value, next_critic_state = self.critic.forward_step(obs, critic_state, mask)
        return action, log_prob, value, next_actor_state, next_critic_state

    def predict_value(
        self, obs: torch.Tensor, critic_state: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.critic.forward_step(obs, critic_state, mask)

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

        mean = self.actor_mean(flat_head_in)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(flat_actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        values = self.critic.forward_sequence(obs_seq, critic_state0, mask_seq)

        return (
            values,
            log_prob.reshape(steps, batch, 1),
            entropy.reshape(steps, batch, 1),
        )
