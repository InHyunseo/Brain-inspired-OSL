"""SAC trainer — mirror of PPOTrainer with the algorithm swapped to off-policy SAC.

Reuses the entire PPO pipeline (parallel runner, curriculum via `set_noise_stage`,
connectome actor, recurrent state, logging / TB / checkpoint / eval). Only the
buffer (rollout → replay) and update rule (clipped surrogate → twin-Q SAC with
auto-α) differ.

Curriculum usage from the notebook is identical to PPO:
    trainer.runner.set_noise_stage(stage, strength)
    trainer.train(phase_timesteps=N)
"""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from src.envs.osl_env import OBS_DIM, ACTION_DIM, OslEnv
from src.envs.parallel_runner import ParallelRunner, VectorRunner
from src.models.connectome import Connectome


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SACConfig:
    # Rollout / parallelism — same shape as PPOConfig so the notebook stays
    # symmetric. `rollout_steps` is how many env steps we collect per update
    # cycle before doing `gradient_steps` minibatch updates from the replay.
    rollout_steps: int = 32
    num_envs: int = 16
    parallel_envs: bool = True
    gradient_steps: int = 32           # SAC updates per rollout cycle
    batch_size: int = 256              # transitions per minibatch
    learning_starts_steps: int = 5000  # collect random data before learning
    # SAC hyperparameters
    gamma: float = 0.99
    tau: float = 0.005                 # target-Q Polyak rate
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_log_alpha: float = math.log(0.2)
    target_entropy: float | None = None  # default: -action_dim
    actor_max_grad_norm: float = 0.5
    critic_max_grad_norm: float = 0.5
    log_std_init: float = -0.5
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    # Replay buffer
    buffer_capacity: int = 200_000     # total transitions across envs
    # Model
    latent_dim: int = 32
    message_passing_steps: int = 6
    weights_csv: str = "assets/connectome/weights.csv"
    metadata_csv: str = "assets/connectome/metadata.csv"
    critic_hidden: tuple[int, ...] = (128, 128)
    # Eval / log / ckpt — same names as PPOConfig
    eval_interval_updates: int = 10
    eval_episodes: int = 3
    log_every_updates: int = 1
    checkpoint_every_timesteps: int = 100_000
    recent_stats_window: int = 50
    seed: int = 7
    device: str = "auto"

    @classmethod
    def from_args(cls, args) -> "SACConfig":
        kw = {}
        for f in cls.__dataclass_fields__:
            if hasattr(args, f) and getattr(args, f) is not None:
                kw[f] = getattr(args, f)
        return cls(**kw)


# ---------------------------------------------------------------------------
# Policy — twin Q critics + tanh-squashed Gaussian actor over connectome.
# Mirrors src/models/policy.Policy but uses Q(s, a) instead of V(s).
# ---------------------------------------------------------------------------


SENSOR_INDICES = (0, 1)
EFFERENCE_INDICES = (2, 3, 4)


def _gather(obs: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor:
    return obs[..., list(indices)]


class _QNet(nn.Module):
    def __init__(self, hidden: Iterable[int]):
        super().__init__()
        layers: list[nn.Module] = []
        last = OBS_DIM + ACTION_DIM
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


class SACPolicy(nn.Module):
    """Actor uses the connectome (sensor channels) + efference copy + linear
    heads for mean and log_std. Twin Q critics are stateless MLPs over (s, a).

    Action distribution: tanh-squashed Gaussian (SAC standard). Provides
    reparameterised samples and a log-prob with the tanh-Jacobian correction.
    """

    def __init__(
        self,
        weights_csv: str | Path,
        metadata_csv: str | Path,
        latent_dim: int = 32,
        message_passing_steps: int = 6,
        critic_hidden: tuple[int, ...] = (128, 128),
        log_std_init: float = -0.5,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
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
        self.actor_log_std = nn.Linear(head_in_dim, ACTION_DIM)
        # Bias the log_std head toward `log_std_init` at init.
        nn.init.constant_(self.actor_log_std.bias, float(log_std_init))
        nn.init.zeros_(self.actor_log_std.weight)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        self.q1 = _QNet(critic_hidden)
        self.q2 = _QNet(critic_hidden)
        self.q1_target = _QNet(critic_hidden)
        self.q2_target = _QNet(critic_hidden)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        for p in self.q1_target.parameters():
            p.requires_grad_(False)
        for p in self.q2_target.parameters():
            p.requires_grad_(False)

        self.actor_state_size = self.connectome.state_size

    def actor_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.connectome.parameters()
        yield from self.actor_mean.parameters()
        yield from self.actor_log_std.parameters()

    def critic_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.q1.parameters()
        yield from self.q2.parameters()

    def initial_states(self, batch_size: int, device: torch.device):
        actor_state = self.connectome.initial_state(batch_size, device)
        critic_state = torch.zeros(batch_size, 0, device=device)
        return actor_state, critic_state

    def _actor_forward(
        self, obs: torch.Tensor, actor_state: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sensor = _gather(obs, SENSOR_INDICES)
        efference = _gather(obs, EFFERENCE_INDICES)
        latent, next_actor_state = self.connectome.forward_step(sensor, actor_state, mask)
        head_in = torch.cat([latent, efference], dim=-1)
        mean = self.actor_mean(head_in)
        log_std = self.actor_log_std(head_in).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std, next_actor_state

    def _sample_action(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        std = log_std.exp()
        if deterministic:
            x = mean
            action = torch.tanh(x)
            # log_prob undefined for deterministic eval; return zero.
            log_prob = torch.zeros(action.shape[:-1] + (1,), device=action.device)
            return action, log_prob
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        # log-prob with tanh Jacobian correction. Sum over action dim.
        log_prob = normal.log_prob(x) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def act(
        self,
        obs: torch.Tensor,
        actor_state: torch.Tensor,
        critic_state: torch.Tensor,
        mask: torch.Tensor,
        deterministic: bool = False,
    ):
        """Single-step action for env interaction.

        Returns (action, log_prob, next_actor_state, next_critic_state) — same
        signature shape the PPO eval cell expects (ignoring the value channel)."""
        mean, log_std, next_actor_state = self._actor_forward(obs, actor_state, mask)
        action, log_prob = self._sample_action(mean, log_std, deterministic)
        return action, log_prob, next_actor_state, critic_state

    def sample_with_logprob(
        self, obs: torch.Tensor, actor_state: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """For SAC actor update — reparameterised sample + log-prob."""
        mean, log_std, next_actor_state = self._actor_forward(obs, actor_state, mask)
        action, log_prob = self._sample_action(mean, log_std, deterministic=False)
        return action, log_prob, next_actor_state

    @torch.no_grad()
    def soft_update_targets(self, tau: float) -> None:
        for tp, p in zip(self.q1_target.parameters(), self.q1.parameters()):
            tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)
        for tp, p in zip(self.q2_target.parameters(), self.q2.parameters()):
            tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


# ---------------------------------------------------------------------------
# Replay buffer — ring buffer over `num_envs * rollout_steps` transitions.
# Stores actor_state and mask alongside the transition so the actor's recurrent
# state at time t is available during off-policy updates (we use a single-step
# update rule: no BPTT through the replay).
# ---------------------------------------------------------------------------


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        actor_state_dim: int,
        device: torch.device,
    ):
        self.capacity = int(capacity)
        self.device = device
        self.obs = torch.zeros(self.capacity, obs_dim, device=device)
        self.actions = torch.zeros(self.capacity, action_dim, device=device)
        self.rewards = torch.zeros(self.capacity, 1, device=device)
        self.next_obs = torch.zeros(self.capacity, obs_dim, device=device)
        self.dones = torch.zeros(self.capacity, 1, device=device)
        self.actor_states = torch.zeros(self.capacity, actor_state_dim, device=device)
        self.next_actor_states = torch.zeros(self.capacity, actor_state_dim, device=device)
        self.masks = torch.zeros(self.capacity, 1, device=device)
        self._idx = 0
        self._full = False

    def __len__(self) -> int:
        return self.capacity if self._full else self._idx

    def add_batch(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        actor_state: torch.Tensor,
        next_actor_state: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        n = obs.shape[0]
        # Wrap-around write.
        end = self._idx + n
        if end <= self.capacity:
            sl = slice(self._idx, end)
            self.obs[sl] = obs
            self.actions[sl] = action
            self.rewards[sl] = reward
            self.next_obs[sl] = next_obs
            self.dones[sl] = done
            self.actor_states[sl] = actor_state
            self.next_actor_states[sl] = next_actor_state
            self.masks[sl] = mask
        else:
            first = self.capacity - self._idx
            sl1 = slice(self._idx, self.capacity)
            self.obs[sl1] = obs[:first]
            self.actions[sl1] = action[:first]
            self.rewards[sl1] = reward[:first]
            self.next_obs[sl1] = next_obs[:first]
            self.dones[sl1] = done[:first]
            self.actor_states[sl1] = actor_state[:first]
            self.next_actor_states[sl1] = next_actor_state[:first]
            self.masks[sl1] = mask[:first]
            sl2 = slice(0, n - first)
            self.obs[sl2] = obs[first:]
            self.actions[sl2] = action[first:]
            self.rewards[sl2] = reward[first:]
            self.next_obs[sl2] = next_obs[first:]
            self.dones[sl2] = done[first:]
            self.actor_states[sl2] = actor_state[first:]
            self.next_actor_states[sl2] = next_actor_state[first:]
            self.masks[sl2] = mask[first:]
            self._full = True
        self._idx = (self._idx + n) % self.capacity
        if self._idx == 0 and end >= self.capacity:
            self._full = True

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        n = len(self)
        idx = torch.randint(0, n, (batch_size,), device=self.device)
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
            "actor_states": self.actor_states[idx],
            "next_actor_states": self.next_actor_states[idx],
            "masks": self.masks[idx],
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def _resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


class SACTrainer:
    def __init__(self, env_config: dict[str, Any], cfg: SACConfig, run_dir: str | Path):
        self.env_config = dict(env_config)
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        use_parallel = bool(cfg.parallel_envs and cfg.num_envs > 1)
        runner_cls = ParallelRunner if use_parallel else VectorRunner
        self.runner = runner_cls(self.env_config, cfg.num_envs, cfg.seed)
        self.use_parallel_envs = use_parallel

        obs_dim = int(np.prod(self.runner.observation_shape))
        action_dim = int(np.prod(self.runner.action_shape))

        self.policy = SACPolicy(
            weights_csv=cfg.weights_csv,
            metadata_csv=cfg.metadata_csv,
            latent_dim=cfg.latent_dim,
            message_passing_steps=cfg.message_passing_steps,
            critic_hidden=cfg.critic_hidden,
            log_std_init=cfg.log_std_init,
            log_std_min=cfg.log_std_min,
            log_std_max=cfg.log_std_max,
        ).to(self.device)

        self.actor_params = list(self.policy.actor_parameters())
        self.critic_params = list(self.policy.critic_parameters())
        self.actor_optimizer = torch.optim.Adam(self.actor_params, lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_params, lr=cfg.critic_lr)
        # Auto-tuned entropy temperature α.
        self.log_alpha = torch.tensor(float(cfg.init_log_alpha), requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.target_entropy = float(cfg.target_entropy) if cfg.target_entropy is not None else -float(action_dim)

        self.buffer = ReplayBuffer(
            capacity=cfg.buffer_capacity,
            obs_dim=obs_dim,
            action_dim=action_dim,
            actor_state_dim=self.policy.actor_state_size,
            device=self.device,
        )

        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "training_log.jsonl"
        self._write_config()

        self.tb_dir = self.run_dir / "tb"
        self._tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb_dir.mkdir(parents=True, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=str(self.tb_dir))
        except Exception as exc:  # noqa: BLE001
            print(f"[SACTrainer] TensorBoard disabled ({exc}); writing JSONL only.")

        # persistent rollout state across phases (same as PPOTrainer)
        self._obs: torch.Tensor | None = None
        self._actor_state: torch.Tensor | None = None
        self._critic_state: torch.Tensor | None = None
        self._mask: torch.Tensor | None = None
        self._episode_returns = np.zeros(cfg.num_envs, dtype=np.float64)
        self._episode_lengths = np.zeros(cfg.num_envs, dtype=np.int64)
        self._episode_casts = np.zeros(cfg.num_envs, dtype=np.int64)
        self._completed_returns: list[float] = []
        self._completed_lengths: list[int] = []
        self._recent_returns: deque[float] = deque(maxlen=max(1, cfg.recent_stats_window))
        self._recent_lengths: deque[int] = deque(maxlen=max(1, cfg.recent_stats_window))
        self._recent_successes: deque[float] = deque(maxlen=max(1, cfg.recent_stats_window))
        self._recent_casts: deque[float] = deque(maxlen=max(1, cfg.recent_stats_window))
        self._total_steps = 0
        self._update_count = 0
        self._next_checkpoint_step = max(1, cfg.checkpoint_every_timesteps)

    # ---- bookkeeping helpers — identical surface to PPOTrainer -----------
    def _write_config(self) -> None:
        payload = {"env_config": self.env_config, "agent_config": asdict(self.cfg)}
        with (self.run_dir / "config.json").open("w", encoding="utf-8") as h:
            json.dump(payload, h, indent=2)

    def _checkpoint_payload(self, summary: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "policy_state_dict": self.policy.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "env_config": self.env_config,
            "agent_config": asdict(self.cfg),
            "training_state": {
                "total_steps": int(self._total_steps),
                "updates": int(self._update_count),
            },
            "summary": summary or {},
        }

    def save_checkpoint(self, filename: str, summary: dict[str, Any] | None = None) -> Path:
        path = self.checkpoint_dir / filename
        torch.save(self._checkpoint_payload(summary), path)
        return path

    def _append_log(self, payload: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as h:
            h.write(json.dumps(payload, ensure_ascii=True) + "\n")
        if self._tb_writer is not None:
            step = int(payload.get("total_steps", self._total_steps))
            for k, v in payload.items():
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    continue
                if k == "total_steps":
                    continue
                try:
                    self._tb_writer.add_scalar(k, float(v), step)
                except Exception:
                    pass
            self._tb_writer.flush()

    def _print_progress(self, payload: dict[str, Any]) -> None:
        print(
            f"[Train] step {payload['total_steps']} "
            f"| update {payload['updates']} "
            f"| eps {payload['completed_episodes']} "
            f"| ret {payload['recent_return_mean']:.3f} "
            f"| len {payload['recent_episode_length_mean']:.1f} "
            f"| success {payload['recent_success_rate'] * 100.0:.1f}% "
            f"| casts {payload.get('recent_cast_mean', 0.0):.1f} "
            f"| actor {payload['actor_loss']:.4f} "
            f"| critic {payload['critic_loss']:.4f} "
            f"| α {payload['alpha']:.4f}"
        )

    def _ensure_rollout_state(self) -> None:
        if self._obs is not None:
            return
        obs_np, _ = self.runner.reset()
        self._obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        self._actor_state, self._critic_state = self.policy.initial_states(self.cfg.num_envs, self.device)
        self._mask = torch.zeros(self.cfg.num_envs, 1, device=self.device)

    def _evaluate(self, episodes: int) -> dict[str, float]:
        env = OslEnv({**self.env_config, "seed": self.cfg.seed + 10_000})
        returns, successes = [], []
        for episode_idx in range(episodes):
            obs, _ = env.reset(seed=self.cfg.seed + 10_000 + episode_idx)
            actor_state, critic_state = self.policy.initial_states(1, self.device)
            mask = torch.zeros(1, 1, device=self.device)
            ep_return = 0.0
            done = False
            success = False
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action, _, next_actor_state, next_critic_state = self.policy.act(
                        obs_t, actor_state, critic_state, mask, deterministic=True
                    )
                obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
                ep_return += reward
                done = bool(terminated or truncated)
                if done:
                    success = bool(info.get("success", False))
                mask.fill_(0.0 if done else 1.0)
                actor_state = next_actor_state * mask
                critic_state = next_critic_state * mask
            returns.append(ep_return)
            successes.append(1.0 if success else 0.0)
        env.close()
        return {
            "eval_return_mean": float(np.mean(returns)) if returns else 0.0,
            "eval_return_std": float(np.std(returns)) if returns else 0.0,
            "eval_success_rate": float(np.mean(successes)) if successes else 0.0,
        }

    # ---- SAC update -----------------------------------------------------
    def _update_policy(self) -> dict[str, float]:
        if len(self.buffer) < max(self.cfg.batch_size, self.cfg.learning_starts_steps):
            return {"actor_loss": 0.0, "critic_loss": 0.0, "alpha": float(self.log_alpha.exp().detach()),
                    "entropy_target": self.target_entropy, "q_mean": 0.0}

        actor_losses, critic_losses, alpha_vals, q_means = [], [], [], []
        for _ in range(self.cfg.gradient_steps):
            batch = self.buffer.sample(self.cfg.batch_size)
            obs = batch["obs"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            next_obs = batch["next_obs"]
            dones = batch["dones"]
            next_actor_states = batch["next_actor_states"]
            # Recurrent state used to recompute action samples at obs (used in
            # the actor update). We use the stored actor_state at time t.
            actor_states = batch["actor_states"]
            ones_mask = torch.ones(obs.shape[0], 1, device=self.device)

            # ---- Critic update ----
            with torch.no_grad():
                # mask=1 (non-terminal continuation) for the recurrent step on
                # next_obs — the buffer's next_actor_state already accounts for
                # episode boundaries (we zeroed it in the rollout when done).
                next_action, next_log_prob, _ = self.policy.sample_with_logprob(
                    next_obs, next_actor_states, ones_mask
                )
                target_q1 = self.policy.q1_target(next_obs, next_action)
                target_q2 = self.policy.q2_target(next_obs, next_action)
                target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_prob
                target = rewards + self.cfg.gamma * (1.0 - dones) * target_q

            q1 = self.policy.q1(obs, actions)
            q2 = self.policy.q2(obs, actions)
            critic_loss = 0.5 * ((q1 - target).pow(2).mean() + (q2 - target).pow(2).mean())

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic_params, self.cfg.critic_max_grad_norm)
            self.critic_optimizer.step()

            # ---- Actor update ----
            new_action, new_log_prob, _ = self.policy.sample_with_logprob(
                obs, actor_states, ones_mask
            )
            q1_new = self.policy.q1(obs, new_action)
            q2_new = self.policy.q2(obs, new_action)
            q_new = torch.min(q1_new, q2_new)
            alpha = self.log_alpha.exp().detach()
            actor_loss = (alpha * new_log_prob - q_new).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_params, self.cfg.actor_max_grad_norm)
            self.actor_optimizer.step()

            # ---- α update ----
            alpha_loss = -(self.log_alpha * (new_log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # ---- Target soft update ----
            self.policy.soft_update_targets(self.cfg.tau)

            actor_losses.append(float(actor_loss.detach().cpu()))
            critic_losses.append(float(critic_loss.detach().cpu()))
            alpha_vals.append(float(self.log_alpha.exp().detach().cpu()))
            q_means.append(float(q_new.detach().mean().cpu()))

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "alpha": float(np.mean(alpha_vals)),
            "entropy_target": self.target_entropy,
            "q_mean": float(np.mean(q_means)),
        }

    # ---- Main train loop ------------------------------------------------
    def train(self, phase_timesteps: int) -> dict[str, Any]:
        """Run rollout/update loop until `phase_timesteps` env-steps have been collected
        in this phase. Persistent state (rollout state, optimizer, buffer) is preserved
        across phases — call multiple times to advance a curriculum."""
        self._ensure_rollout_state()
        phase_start_step = self._total_steps
        phase_target = phase_start_step + int(phase_timesteps)
        latest_metrics: dict[str, Any] = {
            "actor_loss": 0.0, "critic_loss": 0.0,
            "alpha": float(self.log_alpha.exp().detach()), "q_mean": 0.0,
        }
        log_payload: dict[str, Any] = {}

        print(
            f"[PhaseStart] target_steps={phase_target} (this phase +{phase_timesteps}) "
            f"| device={self.device} | runner={'parallel' if self.use_parallel_envs else 'local'}"
        )

        while self._total_steps < phase_target:
            # ---- Rollout: collect `rollout_steps` transitions per env. ----
            for _ in range(self.cfg.rollout_steps):
                obs_before = self._obs
                actor_state_before = self._actor_state
                mask_before = self._mask
                with torch.no_grad():
                    action, _log_prob, next_actor_state, next_critic_state = self.policy.act(
                        obs_before, actor_state_before, self._critic_state, mask_before,
                        deterministic=False,
                    )
                next_obs_np, reward_np, done_np, infos = self.runner.step(action.cpu().numpy())
                reward = torch.as_tensor(reward_np, dtype=torch.float32, device=self.device)
                next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=self.device)
                done_t = torch.as_tensor(done_np, dtype=torch.float32, device=self.device)
                next_mask = 1.0 - done_t

                # Persist transition. We store the actor_state used at this
                # step and the post-step actor_state (already masked at done).
                masked_next_actor_state = next_actor_state * next_mask
                self.buffer.add_batch(
                    obs=obs_before,
                    action=action,
                    reward=reward,
                    next_obs=next_obs,
                    done=done_t,
                    actor_state=actor_state_before,
                    next_actor_state=masked_next_actor_state,
                    mask=mask_before,
                )

                # Episode bookkeeping (identical to PPOTrainer).
                self._episode_returns += reward_np[:, 0]
                self._episode_lengths += 1
                for env_idx in range(self.cfg.num_envs):
                    if infos[env_idx].get("event_is_high_cast_like", False):
                        self._episode_casts[env_idx] += 1
                done_flags = done_np[:, 0] > 0.5
                for env_idx in np.flatnonzero(done_flags):
                    ep_return = float(self._episode_returns[env_idx])
                    ep_length = int(self._episode_lengths[env_idx])
                    ep_casts = int(self._episode_casts[env_idx])
                    success = bool(infos[env_idx].get("success", False))
                    self._completed_returns.append(ep_return)
                    self._completed_lengths.append(ep_length)
                    self._recent_returns.append(ep_return)
                    self._recent_lengths.append(ep_length)
                    self._recent_successes.append(1.0 if success else 0.0)
                    self._recent_casts.append(float(ep_casts))
                    self._episode_returns[env_idx] = 0.0
                    self._episode_lengths[env_idx] = 0
                    self._episode_casts[env_idx] = 0

                self._obs = next_obs
                self._actor_state = masked_next_actor_state
                self._critic_state = next_critic_state
                self._mask = next_mask
                self._total_steps += self.cfg.num_envs
                if self._total_steps >= phase_target:
                    break

            # ---- Update phase ----
            latest_metrics = self._update_policy()
            self._update_count += 1

            if self._update_count % self.cfg.eval_interval_updates == 0:
                latest_metrics.update(self._evaluate(self.cfg.eval_episodes))

            log_payload = {
                "total_steps": int(self._total_steps),
                "updates": int(self._update_count),
                "completed_episodes": len(self._completed_returns),
                "recent_return_mean": float(np.mean(self._recent_returns)) if self._recent_returns else 0.0,
                "recent_episode_length_mean": float(np.mean(self._recent_lengths)) if self._recent_lengths else 0.0,
                "recent_success_rate": float(np.mean(self._recent_successes)) if self._recent_successes else 0.0,
                "recent_cast_mean": float(np.mean(self._recent_casts)) if self._recent_casts else 0.0,
                **latest_metrics,
            }
            if self._update_count % max(1, self.cfg.log_every_updates) == 0:
                self._print_progress(log_payload)
                self._append_log(log_payload)

            while self._total_steps >= self._next_checkpoint_step:
                ckpt_name = f"step_{self._next_checkpoint_step:09d}.pt"
                ckpt_path = self.save_checkpoint(ckpt_name, summary=log_payload)
                print(f"[Checkpoint] {ckpt_path}")
                self._next_checkpoint_step += max(1, self.cfg.checkpoint_every_timesteps)

        summary = {
            "total_steps": int(self._total_steps),
            "updates": int(self._update_count),
            "completed_episodes": len(self._completed_returns),
            "mean_episode_return": float(np.mean(self._completed_returns)) if self._completed_returns else 0.0,
            "median_episode_return": float(np.median(self._completed_returns)) if self._completed_returns else 0.0,
            "mean_episode_length": float(np.mean(self._completed_lengths)) if self._completed_lengths else 0.0,
            "recent_return_mean": float(np.mean(self._recent_returns)) if self._recent_returns else 0.0,
            "recent_success_rate": float(np.mean(self._recent_successes)) if self._recent_successes else 0.0,
            **latest_metrics,
        }
        return summary

    def save_final(self, summary: dict[str, Any]) -> Path:
        path = self.run_dir / "ckpt_final.pt"
        torch.save(self._checkpoint_payload(summary), path)
        with (self.run_dir / "summary.json").open("w", encoding="utf-8") as h:
            json.dump(summary, h, indent=2)
        return path

    def close(self) -> None:
        self.runner.close()
        if self._tb_writer is not None:
            try:
                self._tb_writer.close()
            except Exception:
                pass
            self._tb_writer = None
