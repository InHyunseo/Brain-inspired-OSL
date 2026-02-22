import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.models.networks import RQNet

class DRQNAgent:
    def __init__(self, obs_dim, act_dim, device, rnn_hidden=147, lr=1e-4, gamma=0.99, max_grad_norm=10.0):
        self.device = device
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.act_dim = act_dim
        self.rnn_hidden = rnn_hidden

        # Networks
        self.q = RQNet(obs_dim, act_dim, hidden=rnn_hidden).to(device)
        self.tq = RQNet(obs_dim, act_dim, hidden=rnn_hidden).to(device)
        self.tq.load_state_dict(self.q.state_dict())
        
        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def get_action(self, obs, h=None, epsilon=0.0):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            qvals, h2 = self.q.forward_last(obs_t, h)
        
        if np.random.random() < epsilon:
            return np.random.randint(0, self.act_dim), h2
        else:
            return int(torch.argmax(qvals, dim=1).item()), h2

    def update(self, batch):
        obs_seq, act_seq, rew_seq, terminal_seq = batch
        
        obs_seq = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device)   # (B, T+1, D)
        act_seq = torch.as_tensor(act_seq, dtype=torch.int64, device=self.device)     # (B, T)
        rew_seq = torch.as_tensor(rew_seq, dtype=torch.float32, device=self.device)   # (B, T)
        terminal_seq = torch.as_tensor(terminal_seq, dtype=torch.float32, device=self.device) # (B, T)

        # Current Q
        q_all, _ = self.q(obs_seq, None) # (B, T+1, A)
        qsa = q_all[:, :-1, :].gather(2, act_seq.unsqueeze(-1)).squeeze(-1) # (B, T)

        # Target Q (Double DQN)
        with torch.no_grad():
            tq_all, _ = self.tq(obs_seq, None)
            next_a = torch.argmax(q_all[:, 1:, :], dim=2, keepdim=True)
            next_q = tq_all[:, 1:, :].gather(2, next_a).squeeze(-1)
            y = rew_seq + self.gamma * (1.0 - terminal_seq) * next_q

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
