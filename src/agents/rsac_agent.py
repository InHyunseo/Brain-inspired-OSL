"""Recurrent SAC for the 2D OSL env (3D Gaussian over [v, body_omega, head_omega])."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.envs.osl_env import ACTION_DIM
from src.models.networks import ConnectomeActor, GRUActor, MLPActor, QCritic


_BACKBONES = {"gru", "connectome", "mlp"}


class RSACAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        action_low,
        action_high,
        device,
        rnn_hidden=147,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        gamma=0.99,
        tau=0.005,
        actor_backbone="connectome",
        connectome_latent_dim=32,
        connectome_message_passing_steps=6,
        connectome_weights_csv="assets/connectome/weights.csv",
        connectome_metadata_csv="assets/connectome/metadata.csv",
    ):
        self.device = device
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.act_dim = int(act_dim)
        self.actor_backbone = str(actor_backbone).lower()
        if self.actor_backbone not in _BACKBONES:
            raise ValueError(f"Unsupported actor_backbone {self.actor_backbone}; pick from {_BACKBONES}.")
        if self.act_dim != ACTION_DIM:
            raise ValueError(f"RSACAgent expects action dim exactly {ACTION_DIM}: [v, body_omega, head_omega].")

        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=device)

        if self.actor_backbone == "gru":
            self.actor = GRUActor(obs_dim, act_dim=ACTION_DIM, hidden=rnn_hidden).to(device)
        elif self.actor_backbone == "connectome":
            self.actor = ConnectomeActor(
                obs_dim,
                act_dim=ACTION_DIM,
                weights_csv=connectome_weights_csv,
                metadata_csv=connectome_metadata_csv,
                latent_dim=connectome_latent_dim,
                message_passing_steps=connectome_message_passing_steps,
            ).to(device)
        else:  # mlp
            self.actor = MLPActor(obs_dim, act_dim=ACTION_DIM, hidden=rnn_hidden).to(device)

        self.q1 = QCritic(obs_dim, act_dim, hidden=rnn_hidden).to(device)
        self.q2 = QCritic(obs_dim, act_dim, hidden=rnn_hidden).to(device)
        self.tq1 = QCritic(obs_dim, act_dim, hidden=rnn_hidden).to(device)
        self.tq2 = QCritic(obs_dim, act_dim, hidden=rnn_hidden).to(device)
        self.tq1.load_state_dict(self.q1.state_dict())
        self.tq2.load_state_dict(self.q2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr_critic
        )

        self.log_alpha = torch.zeros(1, device=device, requires_grad=True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr_alpha)
        self.target_entropy = -float(ACTION_DIM)

        self.loss_fn = nn.SmoothL1Loss()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _flatten_action(self, action: torch.Tensor) -> np.ndarray:
        # ConnectomeActor returns (B, D); GRU/MLP return (B, T, D). Take last step.
        flat = action[:, -1, :] if action.dim() == 3 else action
        return flat.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def get_action(self, obs, h=None):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, h2, _ = self.actor.sample(obs_t, self.action_low, self.action_high, h)
        return self._flatten_action(action), h2

    def get_action_deterministic(self, obs, h=None):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, h2 = self.actor.deterministic(obs_t, self.action_low, self.action_high, h)
        return self._flatten_action(action), h2

    def update(self, batch):
        obs_seq, act_seq, rew_seq, terminal_seq = batch

        obs_seq = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device)
        act_seq = torch.as_tensor(act_seq, dtype=torch.float32, device=self.device)
        rew_seq = torch.as_tensor(rew_seq, dtype=torch.float32, device=self.device)
        terminal_seq = torch.as_tensor(terminal_seq, dtype=torch.float32, device=self.device)

        obs_t = obs_seq[:, :-1, :]
        next_obs_t = obs_seq[:, 1:, :]

        with torch.no_grad():
            next_a, next_logp, _, _ = self.actor.sample(next_obs_t, self.action_low, self.action_high, None)
            next_q1, _ = self.tq1(next_obs_t, next_a, None)
            next_q2, _ = self.tq2(next_obs_t, next_a, None)
            next_q = torch.min(next_q1, next_q2) - self.alpha.detach() * next_logp
            y = rew_seq + self.gamma * (1.0 - terminal_seq) * next_q

        q1, _ = self.q1(obs_t, act_seq, None)
        q2, _ = self.q2(obs_t, act_seq, None)
        critic_loss = self.loss_fn(q1, y) + self.loss_fn(q2, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        pi_a, logp, _, _ = self.actor.sample(obs_t, self.action_low, self.action_high, None)
        q1_pi, _ = self.q1(obs_t, pi_a, None)
        q2_pi, _ = self.q2(obs_t, pi_a, None)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * logp - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self.soft_update()

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
        }

    def soft_update(self):
        with torch.no_grad():
            for p, tp in zip(self.q1.parameters(), self.tq1.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
            for p, tp in zip(self.q2.parameters(), self.tq2.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "actor_backbone": self.actor_backbone,
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "tq1": self.tq1.state_dict(),
            "tq2": self.tq2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        ckpt_backbone = str(ckpt.get("actor_backbone", "gru")).lower()
        if ckpt_backbone != self.actor_backbone:
            raise ValueError(
                f"Checkpoint actor_backbone={ckpt_backbone} but agent actor_backbone={self.actor_backbone}."
            )
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.tq1.load_state_dict(ckpt.get("tq1", ckpt["q1"]))
        self.tq2.load_state_dict(ckpt.get("tq2", ckpt["q2"]))
        if "log_alpha" in ckpt:
            self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))
