import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.networks import RecurrentHybridActor, RecurrentQCritic


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
        cell_type="gru",
    ):
        self.device = device
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.act_dim = int(act_dim)
        if self.act_dim < 3:
            raise ValueError("RSACAgent expects hybrid action dim 3: [v, omega, cast].")

        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=device)

        self.actor = RecurrentHybridActor(
            obs_dim,
            cont_act_dim=2,
            hidden=rnn_hidden,
            cell_type=cell_type,
        ).to(device)

        self.q1 = RecurrentQCritic(obs_dim, act_dim, hidden=rnn_hidden, cell_type=cell_type).to(device)
        self.q2 = RecurrentQCritic(obs_dim, act_dim, hidden=rnn_hidden, cell_type=cell_type).to(device)
        self.tq1 = RecurrentQCritic(obs_dim, act_dim, hidden=rnn_hidden, cell_type=cell_type).to(device)
        self.tq2 = RecurrentQCritic(obs_dim, act_dim, hidden=rnn_hidden, cell_type=cell_type).to(device)
        self.tq1.load_state_dict(self.q1.state_dict())
        self.tq2.load_state_dict(self.q2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr_critic)

        self.log_alpha = torch.zeros(1, device=device, requires_grad=True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr_alpha)
        self.target_entropy = -float(act_dim)*0.5

        self.loss_fn = nn.MSELoss()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def get_action(self, obs, h=None, epsilon=1.0):
        del epsilon
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, h2, _, _ = self.actor.sample(obs_t, self.action_low, self.action_high, h)
        a = action[:, -1, :].squeeze(0).detach().cpu().numpy().astype(np.float32)
        return a, h2

    def get_action_deterministic(self, obs, h=None):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, h2 = self.actor.deterministic(obs_t, self.action_low, self.action_high, h)
        a = action[:, -1, :].squeeze(0).detach().cpu().numpy().astype(np.float32)
        return a, h2

    def update(self, batch):
        obs_seq, act_seq, rew_seq, done_seq = batch

        obs_seq = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device)    # (B, T+1, D)
        act_seq = torch.as_tensor(act_seq, dtype=torch.float32, device=self.device)    # (B, T, A)
        rew_seq = torch.as_tensor(rew_seq, dtype=torch.float32, device=self.device)    # (B, T)
        done_seq = torch.as_tensor(done_seq, dtype=torch.float32, device=self.device)  # (B, T)

        obs_t = obs_seq[:, :-1, :]
        next_obs_t = obs_seq[:, 1:, :]

        with torch.no_grad():
            next_a, next_logp, _, _, _ = self.actor.sample(next_obs_t, self.action_low, self.action_high, None)
            next_q1, _ = self.tq1(next_obs_t, next_a, None)
            next_q2, _ = self.tq2(next_obs_t, next_a, None)
            next_q = torch.min(next_q1, next_q2) - self.alpha.detach() * next_logp
            y = rew_seq + self.gamma * (1.0 - done_seq) * next_q

        q1, _ = self.q1(obs_t, act_seq, None)
        q2, _ = self.q2(obs_t, act_seq, None)
        critic_loss = self.loss_fn(q1, y) + self.loss_fn(q2, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        pi_a, logp, _, _, _ = self.actor.sample(obs_t, self.action_low, self.action_high, None)
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

        td_err = (y - q1).abs().mean().item()
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "td_abs": float(td_err),
        }

    def soft_update(self):
        with torch.no_grad():
            for p, tp in zip(self.q1.parameters(), self.tq1.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
            for p, tp in zip(self.q2.parameters(), self.tq2.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def sync_target(self):
        # compatibility with existing training loop; SAC uses soft updates each step.
        return None

    def save(self, path):
        ckpt = {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "tq1": self.tq1.state_dict(),
            "tq2": self.tq2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }
        torch.save(ckpt, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.tq1.load_state_dict(ckpt.get("tq1", ckpt["q1"]))
        self.tq2.load_state_dict(ckpt.get("tq2", ckpt["q2"]))
        if "log_alpha" in ckpt:
            self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))
