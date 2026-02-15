import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.models.networks import DQN


class DQNAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        device,
        hidden=256,
        lr=1e-4,
        gamma=0.99,
        max_grad_norm=10.0,
    ):
        self.device = device
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.act_dim = act_dim

        self.q = DQN(obs_dim, act_dim, hidden=hidden).to(device)
        self.tq = DQN(obs_dim, act_dim, hidden=hidden).to(device)
        self.tq.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def get_action(self, obs, h=None, epsilon=0.0):
        del h
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            qvals = self.q(obs_t)

        if np.random.random() < epsilon:
            return np.random.randint(0, self.act_dim), None
        return int(torch.argmax(qvals, dim=1).item()), None

    def update(self, batch):
        obs_seq, act_seq, rew_seq, done_seq = batch

        obs_seq = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device)   # (B, T+1, D)
        act_seq = torch.as_tensor(act_seq, dtype=torch.int64, device=self.device)     # (B, T)
        rew_seq = torch.as_tensor(rew_seq, dtype=torch.float32, device=self.device)   # (B, T)
        done_seq = torch.as_tensor(done_seq, dtype=torch.float32, device=self.device) # (B, T)

        obs_t = obs_seq[:, :-1, :].reshape(-1, obs_seq.shape[-1])
        next_obs_t = obs_seq[:, 1:, :].reshape(-1, obs_seq.shape[-1])
        act_t = act_seq.reshape(-1)
        rew_t = rew_seq.reshape(-1)
        done_t = done_seq.reshape(-1)

        q_all = self.q(obs_t)
        qsa = q_all.gather(1, act_t.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_a = torch.argmax(self.q(next_obs_t), dim=1, keepdim=True)
            next_q = self.tq(next_obs_t).gather(1, next_a).squeeze(-1)
            y = rew_t + self.gamma * (1.0 - done_t) * next_q

        loss = self.loss_fn(qsa, y)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        self.tq.load_state_dict(self.q.state_dict())

    def save(self, path):
        torch.save(self.q.state_dict(), path)

    def load(self, path):
        self.q.load_state_dict(torch.load(path, map_location=self.device))
        self.tq.load_state_dict(self.q.state_dict())
