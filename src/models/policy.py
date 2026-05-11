"""Actor-critic policy.

Actor: Connectome consumes sensor channels (obs[:, 0:2]); its latent is
concatenated with efference-copy channels (obs[:, 2:5]) before a Linear head
emits the Gaussian mean. Action is tanh-squashed to [-1, 1].

Critic: stateless MLP over the full 5-D obs producing a scalar value.

Sequence operations follow the bySY convention: tensors are `(T, B, D)` for
both `obs_seq` and `mask_seq`; states are `(B, state_size)` carried across
env steps and zeroed via mask at episode boundaries.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.distributions import Normal

from src.envs.osl_env import ACTION_DIM, OBS_DIM
from src.models.connectome import Connectome


SENSOR_INDICES = (0, 1)
EFFERENCE_INDICES = (2, 3, 4)


def _gather(obs: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor:
    return obs[..., list(indices)]


class Policy(nn.Module):
    def __init__(
        self,
        weights_csv: str | Path,
        metadata_csv: str | Path,
        latent_dim: int = 32,
        message_passing_steps: int = 6,
        critic_hidden: tuple[int, ...] = (64, 64),
        log_std_init: float = -0.5,
    ):
        super().__init__()
        self.connectome = Connectome(
            weights_csv=weights_csv,
            metadata_csv=metadata_csv,
            latent_dim=latent_dim,
            message_passing_steps=message_passing_steps,
            activation="tanh",
        )
        head_in_dim = latent_dim + len(EFFERENCE_INDICES)
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

        self.actor_state_size = self.connectome.state_size
        self.critic_state_size = 0  # critic is stateless; placeholder kept for buffer/runner symmetry

    def actor_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.connectome.parameters()
        yield from self.actor_mean.parameters()
        yield self.actor_log_std

    def critic_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.critic.parameters()

    def initial_states(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actor_state = self.connectome.initial_state(batch_size, device)
        critic_state = torch.zeros(batch_size, 0, device=device)
        return actor_state, critic_state

    def _actor_distribution(self, obs: torch.Tensor, actor_state: torch.Tensor, mask: torch.Tensor):
        sensor = _gather(obs, SENSOR_INDICES)
        efference = _gather(obs, EFFERENCE_INDICES)
        latent, next_actor_state = self.connectome.forward_step(sensor, actor_state, mask)
        head_in = torch.cat([latent, efference], dim=-1)
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, next_actor_state = self._actor_distribution(obs, actor_state, mask)
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
        sensor_seq = _gather(obs_seq, SENSOR_INDICES)
        efference_seq = _gather(obs_seq, EFFERENCE_INDICES)
        latent_seq, _ = self.connectome.forward_sequence(sensor_seq, actor_state0, mask_seq)
        head_in_seq = torch.cat([latent_seq, efference_seq], dim=-1)

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
