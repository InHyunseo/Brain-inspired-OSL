"""DRQN/DQN unified Q-net (recurrent toggle)."""
from __future__ import annotations

import torch.nn as nn


class QNet(nn.Module):
    """Q-net for the unified DRQN/DQN agent.

    - recurrent=True : GRU backbone (DRQN). API: forward(x, h) where x is
      (B, T, obs_dim) or (B, obs_dim). Returns (q, h2) with q shape (B, T, A).
    - recurrent=False: MLP backbone (DQN). API: forward(x, h=None). Returns
      (q, None) with q shape matching x (per-timestep if x is (B, T, D)).

    `forward_last` returns the Q-values of the final timestep only, for use in
    online action selection.
    """

    def __init__(self, obs_dim, n_actions, hidden=147, recurrent=True):
        super().__init__()
        self.recurrent = bool(recurrent)
        if self.recurrent:
            self.rnn = nn.GRU(input_size=obs_dim, hidden_size=hidden, batch_first=True)
            self.head = nn.Linear(hidden, n_actions)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
            )
            self.head = nn.Linear(hidden, n_actions)

    def forward(self, x, h=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if self.recurrent:
            y, h2 = self.rnn(x, h)
            return self.head(y), h2
        return self.head(self.mlp(x)), None

    def forward_last(self, x, h=None):
        q, h2 = self.forward(x, h)
        return q[:, -1], h2
