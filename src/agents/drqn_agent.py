"""Unified DRQN/DQN agent for the continuous-action OSL env.

The 2D OSL env exposes Box([v, omega, cast]) actions. This agent runs over a
discrete action set {RUN, CAST, TURN_L, TURN_R} and an internal `to_env_action`
adapter maps each discrete choice to the continuous env action. Toggle the
backbone between recurrent (DRQN) and feed-forward (DQN) via `recurrent`.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.networks import QNet


A_RUN, A_CAST, A_TURN_L, A_TURN_R = 0, 1, 2, 3
N_ACTIONS = 4


class DRQNAgent:
    def __init__(
        self,
        obs_dim,
        action_low,
        action_high,
        device,
        rnn_hidden=147,
        lr=1e-4,
        gamma=0.99,
        recurrent=True,
    ):
        self.device = device
        self.gamma = float(gamma)
        self.recurrent = bool(recurrent)

        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.v_max = float(self.action_high[0])
        self.omega_max = float(self.action_high[1])

        self.q = QNet(obs_dim, N_ACTIONS, hidden=rnn_hidden, recurrent=self.recurrent).to(device)
        self.tq = QNet(obs_dim, N_ACTIONS, hidden=rnn_hidden, recurrent=self.recurrent).to(device)
        self.tq.load_state_dict(self.q.state_dict())

        self.opt = optim.Adam(self.q.parameters(), lr=float(lr))
        self.loss_fn = nn.SmoothL1Loss()

    def to_env_action(self, a_idx):
        """Map a discrete action index into the env's Box(3,) action."""
        if a_idx == A_RUN:
            return np.array([self.v_max, 0.0, 0.0], dtype=np.float32)
        if a_idx == A_CAST:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if a_idx == A_TURN_L:
            return np.array([self.v_max, +self.omega_max, 0.0], dtype=np.float32)
        if a_idx == A_TURN_R:
            return np.array([self.v_max, -self.omega_max, 0.0], dtype=np.float32)
        raise ValueError(f"Unknown discrete action index {a_idx}.")

    def get_action(self, obs, h=None, epsilon=0.0):
        if np.random.random() < float(epsilon):
            return int(np.random.randint(N_ACTIONS)), h
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_last, h2 = self.q.forward_last(obs_t, h)
        return int(q_last.argmax(dim=-1).item()), h2

    def get_action_deterministic(self, obs, h=None):
        return self.get_action(obs, h=h, epsilon=0.0)

    def update(self, batch):
        obs_seq, act_seq, rew_seq, terminal_seq = batch

        obs_seq = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device)
        act_seq = torch.as_tensor(act_seq, dtype=torch.int64, device=self.device)
        rew_seq = torch.as_tensor(rew_seq, dtype=torch.float32, device=self.device)
        terminal_seq = torch.as_tensor(terminal_seq, dtype=torch.float32, device=self.device)

        obs_t = obs_seq[:, :-1, :]
        next_obs_t = obs_seq[:, 1:, :]

        q_all, _ = self.q(obs_t, None)
        q_taken = q_all.gather(-1, act_seq.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_q_all, _ = self.tq(next_obs_t, None)
            next_q_max = next_q_all.max(dim=-1).values
            y = rew_seq + self.gamma * (1.0 - terminal_seq) * next_q_max

        loss = self.loss_fn(q_taken, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return {"loss": float(loss.item())}

    def sync_target(self):
        self.tq.load_state_dict(self.q.state_dict())

    def save(self, path):
        torch.save({
            "q": self.q.state_dict(),
            "tq": self.tq.state_dict(),
            "recurrent": self.recurrent,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q.load_state_dict(ckpt["q"])
        self.tq.load_state_dict(ckpt.get("tq", ckpt["q"]))
