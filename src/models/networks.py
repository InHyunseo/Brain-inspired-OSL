import torch
import torch.nn as nn
import torch.nn.functional as F

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