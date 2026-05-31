"""RSAC backbones.

Hybrid SAC actors with three backbone choices, all producing a tanh-squashed
Gaussian over `[v, body_omega, head_omega]`:

- `GRUActor`        — GRU recurrent backbone
- `ConnectomeActor` — real larva connectome (`Connectome` branch) + efference concat
- `MLPActor`        — feed-forward MLP (no recurrence)

`QCritic` is the GRU-based twin-critic component shared across backbones.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from src.envs.osl_env import ACTION_DIM
from src.models.connectome import Connectome
from src.models.policy import HEAD_EXTRA_INDICES, SENSOR_INDICES, _gather


def _gaussian_sample(mu, log_std, action_low, action_high):
    std = log_std.exp()
    normal = Normal(mu, std)
    x = normal.rsample()
    y = torch.tanh(x)
    scale = (action_high - action_low) * 0.5
    bias = (action_high + action_low) * 0.5
    action = y * scale + bias
    log_prob = normal.log_prob(x) - torch.log(scale * (1.0 - y.pow(2)) + 1e-6)
    log_prob = log_prob.sum(dim=-1, keepdim=True)
    return action, log_prob.squeeze(-1), mu


def _gaussian_deterministic(mu, action_low, action_high):
    y = torch.tanh(mu)
    scale = (action_high - action_low) * 0.5
    bias = (action_high + action_low) * 0.5
    return y * scale + bias


class _GaussianHead(nn.Module):
    def __init__(self, hidden, act_dim, log_std_min=-5.0, log_std_max=2.0):
        super().__init__()
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def forward(self, y):
        mu = self.mu(y)
        log_std = torch.clamp(self.log_std(y), self.log_std_min, self.log_std_max)
        return mu, log_std


class GRUActor(nn.Module):
    def __init__(self, obs_dim, act_dim=ACTION_DIM, hidden=147):
        super().__init__()
        self.rnn = nn.GRU(input_size=obs_dim, hidden_size=hidden, batch_first=True)
        self.head = _GaussianHead(hidden, act_dim)

    def forward(self, obs, h=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        y, h2 = self.rnn(obs, h)
        mu, log_std = self.head(y)
        return mu, log_std, h2

    def sample(self, obs, action_low, action_high, h=None):
        mu, log_std, h2 = self.forward(obs, h)
        action, log_prob, mu_out = _gaussian_sample(mu, log_std, action_low, action_high)
        return action, log_prob, h2, mu_out

    def deterministic(self, obs, action_low, action_high, h=None):
        mu, _, h2 = self.forward(obs, h)
        return _gaussian_deterministic(mu, action_low, action_high), h2


class MLPActor(nn.Module):
    """Stateless MLP actor. The `h` arg is accepted/returned (None) for API symmetry."""

    def __init__(self, obs_dim, act_dim=ACTION_DIM, hidden=147):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = _GaussianHead(hidden, act_dim)

    def forward(self, obs, h=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        y = self.mlp(obs)
        mu, log_std = self.head(y)
        return mu, log_std, None

    def sample(self, obs, action_low, action_high, h=None):
        mu, log_std, _ = self.forward(obs, h)
        action, log_prob, mu_out = _gaussian_sample(mu, log_std, action_low, action_high)
        return action, log_prob, None, mu_out

    def deterministic(self, obs, action_low, action_high, h=None):
        mu, _, _ = self.forward(obs, h)
        return _gaussian_deterministic(mu, action_low, action_high), None


class ConnectomeActor(nn.Module):
    """RSAC actor wrapping the larva `Connectome` branch.

    Sensor channels (`obs[:, 0:2]`) feed the connectome; the dlog + efference-copy
    channels (`obs[:, 2:6]`) are concatenated with the latent before the Gaussian
    head. Hidden state `h` is the connectome's `(B, state_size)` activation.
    """

    def __init__(
        self,
        obs_dim,
        act_dim=ACTION_DIM,
        weights_csv: str | Path = "assets/connectome/weights.csv",
        metadata_csv: str | Path = "assets/connectome/metadata.csv",
        latent_dim: int = 32,
        message_passing_steps: int = 6,
        feature_dim: int = 8,
    ):
        super().__init__()
        self.connectome = Connectome(
            weights_csv=weights_csv,
            metadata_csv=metadata_csv,
            latent_dim=latent_dim,
            message_passing_steps=message_passing_steps,
            activation="tanh",
            feature_dim=feature_dim,
        )
        # actual latent is the D-dim output-node states flattened (n_out * D),
        # exposed as connectome.latent_dim -- not the output-node count.
        head_in = self.connectome.latent_dim + len(HEAD_EXTRA_INDICES)
        self.head = _GaussianHead(head_in, act_dim)
        self.state_size = self.connectome.state_size

    def _step(self, obs, h):
        # obs: (B, D) for online sampling, (B, T, D) for sequence updates.
        # Online (B, D): one connectome step, returns mu/log_std (B, D_act), h2 (B, S).
        # Sequence (B, T, D): T connectome steps, returns mu/log_std (B, T, D_act), h2 (B, S).
        if obs.dim() == 3:
            bsz = obs.shape[0]
            if h is None:
                h = self.connectome.initial_state(bsz, obs.device)
            sensor_seq = _gather(obs, SENSOR_INDICES).transpose(0, 1)       # (T, B, 2)
            head_extra_seq = _gather(obs, HEAD_EXTRA_INDICES)                # (B, T, E)
            mask_seq = torch.ones(sensor_seq.shape[0], bsz, 1,
                                  device=obs.device, dtype=obs.dtype)
            latent_seq, h2 = self.connectome.forward_sequence(sensor_seq, h, mask_seq)
            latent = latent_seq.transpose(0, 1)                              # (B, T, K)
            head_in = torch.cat([latent, head_extra_seq], dim=-1)
            mu, log_std = self.head(head_in)
            return mu, log_std, h2

        if h is None:
            h = self.connectome.initial_state(obs.shape[0], obs.device)
        mask = torch.ones(obs.shape[0], 1, device=obs.device, dtype=obs.dtype)
        sensor = _gather(obs, SENSOR_INDICES)
        head_extra = _gather(obs, HEAD_EXTRA_INDICES)
        latent, h2 = self.connectome.forward_step(sensor, h, mask)
        head_in = torch.cat([latent, head_extra], dim=-1)
        mu, log_std = self.head(head_in)
        return mu, log_std, h2

    def forward(self, obs, h=None):
        return self._step(obs, h)

    def sample(self, obs, action_low, action_high, h=None):
        mu, log_std, h2 = self._step(obs, h)
        action, log_prob, mu_out = _gaussian_sample(mu, log_std, action_low, action_high)
        return action, log_prob, h2, mu_out

    def deterministic(self, obs, action_low, action_high, h=None):
        mu, _, h2 = self._step(obs, h)
        return _gaussian_deterministic(mu, action_low, action_high), h2


class QCritic(nn.Module):
    """GRU-based Q(s, a). Used twice (q1, q2) for SAC twin-critic."""

    def __init__(self, obs_dim, act_dim, hidden=147):
        super().__init__()
        self.rnn = nn.GRU(input_size=obs_dim, hidden_size=hidden, batch_first=True)
        self.fc1 = nn.Linear(hidden + act_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.q = nn.Linear(hidden, 1)

    def forward(self, obs, act, h=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        if act.dim() == 2:
            act = act.unsqueeze(1)
        y, h2 = self.rnn(obs, h)
        z = torch.cat([y, act], dim=-1)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        return self.q(z).squeeze(-1), h2
