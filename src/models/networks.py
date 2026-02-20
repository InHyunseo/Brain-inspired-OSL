import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Bernoulli

class RQNet(nn.Module):
    """Recurrent Q-Network (GRU based)"""
    def __init__(self, obs_dim, act_dim, hidden=147):
        super().__init__()
        self.gru = nn.GRU(input_size=obs_dim, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, act_dim)

    def forward(self, x, h=None):
        # x: (B, T, obs_dim) or (B, obs_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        y, h2 = self.gru(x, h)          # y: (B, T, H)
        q = self.head(y)                # q: (B, T, A)
        return q, h2

    def forward_last(self, x, h=None):
        q, h2 = self.forward(x, h)
        return q[:, -1], h2             # (B, A), (1, B, H)

class DQN(nn.Module):
    """Simple MLP DQN"""
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.layer1 = nn.Linear(obs_dim, hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.layer3 = nn.Linear(hidden, act_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def _build_rnn(cell_type, input_size, hidden_size):
    if cell_type == "rnn":
        return nn.RNN(input_size=input_size, hidden_size=hidden_size, nonlinearity="tanh", batch_first=True)
    return nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)


class RecurrentGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, action_low, action_high, hidden=147, cell_type="gru", log_std_min=-5.0, log_std_max=2.0):
        super().__init__()
        self.rnn = _build_rnn(cell_type, obs_dim, hidden)
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale", (high - low) * 0.5)
        self.register_buffer("action_bias", (high + low) * 0.5)

    def forward(self, obs, h=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        y, h2 = self.rnn(obs, h)
        mu = self.mu(y)
        log_std = torch.clamp(self.log_std(y), self.log_std_min, self.log_std_max)
        return mu, log_std, h2

    def sample(self, obs, h=None):
        mu, log_std, h2 = self.forward(obs, h)
        std = log_std.exp()
        normal = Normal(mu, std)
        x = normal.rsample()
        y = torch.tanh(x)
        action = y * self.action_scale + self.action_bias

        # tanh-squashed Gaussian log-prob with change-of-variables correction
        log_prob = normal.log_prob(x) - torch.log(self.action_scale * (1.0 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        return action, log_prob, h2, mu

    def deterministic(self, obs, h=None):
        mu, _, h2 = self.forward(obs, h)
        y = torch.tanh(mu)
        action = y * self.action_scale + self.action_bias
        return action, h2


class RecurrentQCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=147, cell_type="gru"):
        super().__init__()
        self.rnn = _build_rnn(cell_type, obs_dim, hidden)
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
        q = self.q(z).squeeze(-1)
        return q, h2


class MLPQCritic(nn.Module):
    """
    Non-recurrent Q critic with the same call signature as RecurrentQCritic.
    Returns (q, None) so RSAC code path can stay unchanged.
    """
    def __init__(self, obs_dim, act_dim, hidden=147, cell_type="gru"):
        super().__init__()
        del cell_type  # kept for constructor compatibility
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.q = nn.Linear(hidden, 1)

    def forward(self, obs, act, h=None):
        del h
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        if act.dim() == 2:
            act = act.unsqueeze(1)
        z = torch.cat([obs, act], dim=-1)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        q = self.q(z).squeeze(-1)
        return q, None


class RecurrentHybridActor(nn.Module):
    """
    Hybrid policy for [v, omega, cast]:
    - continuous part: tanh-squashed Gaussian for [v, omega]
    - discrete part: Bernoulli for cast start decision
    """
    def __init__(self, obs_dim, cont_act_dim, hidden=147, cell_type="gru", log_std_min=-5.0, log_std_max=2.0):
        super().__init__()
        self.rnn = _build_rnn(cell_type, obs_dim, hidden)
        self.mu = nn.Linear(hidden, cont_act_dim)
        self.log_std = nn.Linear(hidden, cont_act_dim)
        self.cast_logit = nn.Linear(hidden, 1)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def forward(self, obs, h=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        y, h2 = self.rnn(obs, h)
        mu = self.mu(y)
        log_std = torch.clamp(self.log_std(y), self.log_std_min, self.log_std_max)
        cast_logit = self.cast_logit(y)
        return mu, log_std, cast_logit, h2

    def sample(self, obs, action_low, action_high, h=None):
        mu, log_std, cast_logit, h2 = self.forward(obs, h)
        std = log_std.exp()
        normal = Normal(mu, std)
        x = normal.rsample()
        y = torch.tanh(x)

        cont_low = action_low[:2]
        cont_high = action_high[:2]
        cont_scale = (cont_high - cont_low) * 0.5
        cont_bias = (cont_high + cont_low) * 0.5
        cont_action = y * cont_scale + cont_bias

        cont_log_prob = normal.log_prob(x) - torch.log(cont_scale * (1.0 - y.pow(2)) + 1e-6)
        cont_log_prob = cont_log_prob.sum(dim=-1, keepdim=True)

        bern = Bernoulli(logits=cast_logit)
        cast_action = bern.sample()
        disc_log_prob = bern.log_prob(cast_action)

        action = torch.cat([cont_action, cast_action], dim=-1)
        log_prob = (cont_log_prob + disc_log_prob).squeeze(-1)
        cast_prob = torch.sigmoid(cast_logit)
        return action, log_prob, h2, mu, cast_prob

    def deterministic(self, obs, action_low, action_high, h=None):
        mu, _, cast_logit, h2 = self.forward(obs, h)
        y = torch.tanh(mu)

        cont_low = action_low[:2]
        cont_high = action_high[:2]
        cont_scale = (cont_high - cont_low) * 0.5
        cont_bias = (cont_high + cont_low) * 0.5
        cont_action = y * cont_scale + cont_bias

        cast_action = (cast_logit > 0.0).float()
        action = torch.cat([cont_action, cast_action], dim=-1)
        return action, h2
