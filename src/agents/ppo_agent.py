"""PPO trainer.

Custom on-policy PPO with separate actor/critic optimizers, sequence-based
updates, and recurrent actor state managed across env steps. Curriculum
phases are driven externally by calling `set_noise_stage` on the runner
between `train(phase_timesteps=...)` calls; the same trainer/policy/optimizer/
buffer instances are reused across phases.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.envs.osl_env import OslEnv
from src.envs.parallel_runner import ParallelRunner, VectorRunner
from src.models.policy import Policy


@dataclass
class PPOConfig:
    rollout_steps: int = 128
    num_envs: int = 16
    parallel_envs: bool = True
    update_epochs: int = 4
    minibatch_envs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.005
    value_loss_coef: float = 0.5
    normalize_advantages: bool = True
    clip_value_loss: bool = True
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    actor_max_grad_norm: float = 0.5
    critic_max_grad_norm: float = 0.5
    log_std_init: float = -0.5
    backbone: str = "connectome"       # "connectome" | "gru"
    gru_hidden: int = 421              # GRU hidden (≈ connectome's 423-node state, for scale parity)
    latent_dim: int = 32
    message_passing_steps: int = 6
    weights_csv: str = "assets/connectome/weights.csv"
    metadata_csv: str = "assets/connectome/metadata.csv"
    eval_interval_updates: int = 10
    eval_episodes: int = 3
    log_every_updates: int = 1
    checkpoint_every_timesteps: int = 100_000
    recent_stats_window: int = 50
    seed: int = 7
    device: str = "auto"

    @classmethod
    def from_args(cls, args) -> "PPOConfig":
        kw = {}
        for f in cls.__dataclass_fields__:
            if hasattr(args, f) and getattr(args, f) is not None:
                kw[f] = getattr(args, f)
        return cls(**kw)


class RolloutBuffer:
    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        actor_state_dim: int,
        device: torch.device,
    ):
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.obs = torch.zeros(rollout_steps, num_envs, obs_dim, device=device)
        self.masks = torch.zeros(rollout_steps, num_envs, 1, device=device)
        self.actions = torch.zeros(rollout_steps, num_envs, action_dim, device=device)
        self.log_probs = torch.zeros(rollout_steps, num_envs, 1, device=device)
        self.rewards = torch.zeros(rollout_steps, num_envs, 1, device=device)
        self.values = torch.zeros(rollout_steps, num_envs, 1, device=device)
        self.returns = torch.zeros(rollout_steps, num_envs, 1, device=device)
        self.advantages = torch.zeros(rollout_steps, num_envs, 1, device=device)
        self.initial_actor_states = torch.zeros(num_envs, actor_state_dim, device=device)

    def begin_rollout(self, actor_states: torch.Tensor) -> None:
        self.initial_actor_states = actor_states.clone()

    def insert(self, step, obs, mask, action, log_prob, reward, value):
        self.obs[step].copy_(obs)
        self.masks[step].copy_(mask)
        self.actions[step].copy_(action)
        self.log_probs[step].copy_(log_prob)
        self.rewards[step].copy_(reward)
        self.values[step].copy_(value)

    def compute_returns(self, next_value, next_mask, gamma, gae_lambda):
        gae = torch.zeros_like(next_value)
        for step in reversed(range(self.rollout_steps)):
            nv = next_value if step == self.rollout_steps - 1 else self.values[step + 1]
            nm = next_mask if step == self.rollout_steps - 1 else self.masks[step + 1]
            delta = self.rewards[step] + gamma * nv * nm - self.values[step]
            gae = delta + gamma * gae_lambda * nm * gae
            self.advantages[step] = gae
        self.returns.copy_(self.advantages + self.values)

    def iter_minibatches(self, minibatch_envs):
        indices = torch.randperm(self.num_envs, device=self.obs.device)
        for start in range(0, self.num_envs, minibatch_envs):
            env_ids = indices[start : start + minibatch_envs]
            yield {
                "obs": self.obs[:, env_ids],
                "masks": self.masks[:, env_ids],
                "actions": self.actions[:, env_ids],
                "log_probs": self.log_probs[:, env_ids],
                "values": self.values[:, env_ids],
                "returns": self.returns[:, env_ids],
                "advantages": self.advantages[:, env_ids],
                "actor_state0": self.initial_actor_states[env_ids],
            }


def _resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


class PPOTrainer:
    def __init__(self, env_config: dict[str, Any], cfg: PPOConfig, run_dir: str | Path):
        self.env_config = dict(env_config)
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        use_parallel = bool(cfg.parallel_envs and cfg.num_envs > 1)
        runner_cls = ParallelRunner if use_parallel else VectorRunner
        self.runner = runner_cls(self.env_config, cfg.num_envs, cfg.seed)
        self.use_parallel_envs = use_parallel

        obs_dim = int(np.prod(self.runner.observation_shape))
        action_dim = int(np.prod(self.runner.action_shape))

        self.policy = Policy(
            weights_csv=cfg.weights_csv,
            metadata_csv=cfg.metadata_csv,
            latent_dim=cfg.latent_dim,
            message_passing_steps=cfg.message_passing_steps,
            log_std_init=cfg.log_std_init,
            backbone=cfg.backbone,
            gru_hidden=cfg.gru_hidden,
        ).to(self.device)

        self.actor_params = list(self.policy.actor_parameters())
        self.critic_params = list(self.policy.critic_parameters())
        self.actor_optimizer = torch.optim.Adam(self.actor_params, lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_params, lr=cfg.critic_lr)
        self.buffer = RolloutBuffer(
            rollout_steps=cfg.rollout_steps,
            num_envs=cfg.num_envs,
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

        # TensorBoard mirror of training_log.jsonl. Optional dep — fall back
        # silently if `tensorboard` isn't installed.
        self.tb_dir = self.run_dir / "tb"
        self._tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb_dir.mkdir(parents=True, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=str(self.tb_dir))
        except Exception as exc:  # noqa: BLE001
            print(f"[PPOTrainer] TensorBoard disabled ({exc}); writing JSONL only.")

        # persistent rollout state across phases
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

    def _write_config(self) -> None:
        payload = {"env_config": self.env_config, "agent_config": asdict(self.cfg)}
        with (self.run_dir / "config.json").open("w", encoding="utf-8") as h:
            json.dump(payload, h, indent=2)

    def _checkpoint_payload(self, summary: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "policy_state_dict": self.policy.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
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

    def load_checkpoint(self, path: str | Path, load_optimizer: bool = True) -> int:
        """Restore policy, optimizers and step counter from a checkpoint to resume.

        Returns the restored `total_steps` so the caller (e.g. the notebook
        curriculum loop) can skip phases already completed. Rolling-stat deques
        are intentionally NOT restored (they are cosmetic recent-window logs);
        only the cumulative step counter that drives the curriculum is.
        """
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(payload["policy_state_dict"])
        if load_optimizer:
            self.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
        ts = payload.get("training_state", {})
        self._total_steps = int(ts.get("total_steps", 0))
        self._update_count = int(ts.get("updates", 0))
        # Resume checkpointing from the next interval boundary past where we are.
        interval = max(1, self.cfg.checkpoint_every_timesteps)
        self._next_checkpoint_step = (self._total_steps // interval + 1) * interval
        print(f"[Resume] loaded {path} -> total_steps={self._total_steps} "
              f"updates={self._update_count} (optimizer={'yes' if load_optimizer else 'no'})")
        return self._total_steps

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
            f"| dist {payload.get('eval_final_distance_mean', float('nan')):.1f}mm "
            f"| casts {payload.get('recent_cast_mean', 0.0):.1f} "
            f"| actor {payload['actor_loss']:.4f} "
            f"| critic {payload['critic_loss']:.4f} "
            f"| ent {payload['entropy']:.4f}"
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
        returns = []
        successes = []
        final_dists = []
        for episode_idx in range(episodes):
            obs, _ = env.reset(seed=self.cfg.seed + 10_000 + episode_idx)
            actor_state, critic_state = self.policy.initial_states(1, self.device)
            mask = torch.zeros(1, 1, device=self.device)
            ep_return = 0.0
            done = False
            success = False
            final_dist = float("nan")
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action, _, _, next_actor_state, next_critic_state = self.policy.act(
                        obs_t, actor_state, critic_state, mask, deterministic=True
                    )
                obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
                ep_return += reward
                done = bool(terminated or truncated)
                if done:
                    success = bool(info.get("success", False))
                    # Distance between the final agent position and the source.
                    final_dist = float(info.get("distance_to_source_mm", float("nan")))
                mask.fill_(0.0 if done else 1.0)
                actor_state = next_actor_state * mask
                critic_state = next_critic_state * mask
            returns.append(ep_return)
            successes.append(1.0 if success else 0.0)
            final_dists.append(final_dist)
        env.close()
        return {
            "eval_return_mean": float(np.mean(returns)) if returns else 0.0,
            "eval_return_std": float(np.std(returns)) if returns else 0.0,
            "eval_success_rate": float(np.mean(successes)) if successes else 0.0,
            "eval_final_distance_mean": float(np.nanmean(final_dists)) if final_dists else float("nan"),
        }

    def _update_policy(self) -> dict[str, float]:
        advantages = self.buffer.advantages
        if self.cfg.normalize_advantages:
            adv_mean = advantages.mean()
            adv_std = advantages.std().clamp_min(1e-6)
            self.buffer.advantages = (advantages - adv_mean) / adv_std
        actor_losses, critic_losses, entropies = [], [], []
        for _ in range(self.cfg.update_epochs):
            for batch in self.buffer.iter_minibatches(self.cfg.minibatch_envs):
                values, log_probs, entropy = self.policy.evaluate_actions_sequence(
                    obs_seq=batch["obs"],
                    mask_seq=batch["masks"],
                    action_seq=batch["actions"],
                    actor_state0=batch["actor_state0"],
                    critic_state0=torch.zeros(batch["actor_state0"].shape[0], 0, device=self.device),
                )
                ratio = torch.exp(log_probs - batch["log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon) * batch["advantages"]
                actor_loss = -torch.min(surr1, surr2).mean() - self.cfg.entropy_coef * entropy.mean()

                if self.cfg.clip_value_loss:
                    clipped_values = batch["values"] + (values - batch["values"]).clamp(
                        -self.cfg.clip_epsilon, self.cfg.clip_epsilon
                    )
                    value_loss = 0.5 * torch.max(
                        (values - batch["returns"]).pow(2),
                        (clipped_values - batch["returns"]).pow(2),
                    ).mean()
                else:
                    value_loss = 0.5 * (values - batch["returns"]).pow(2).mean()
                critic_loss = self.cfg.value_loss_coef * value_loss

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor_params, self.cfg.actor_max_grad_norm)
                self.actor_optimizer.step()

                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_params, self.cfg.critic_max_grad_norm)
                self.critic_optimizer.step()

                actor_losses.append(float(actor_loss.detach().cpu()))
                critic_losses.append(float(critic_loss.detach().cpu()))
                entropies.append(float(entropy.mean().detach().cpu()))
        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
        }

    def train(self, phase_timesteps: int) -> dict[str, Any]:
        """Run rollout/update loop until `phase_timesteps` env-steps have been collected
        in this phase. Persistent state (rollout state, optimizer, buffer) is preserved
        across phases — call multiple times to advance a curriculum."""
        self._ensure_rollout_state()
        phase_start_step = self._total_steps
        phase_target = phase_start_step + int(phase_timesteps)
        latest_metrics: dict[str, Any] = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
        log_payload: dict[str, Any] = {}

        print(
            f"[PhaseStart] target_steps={phase_target} (this phase +{phase_timesteps}) "
            f"| device={self.device} | runner={'parallel' if self.use_parallel_envs else 'local'}"
        )

        while self._total_steps < phase_target:
            self.buffer.begin_rollout(self._actor_state)
            for step in range(self.cfg.rollout_steps):
                with torch.no_grad():
                    action, log_prob, value, next_actor_state, next_critic_state = self.policy.act(
                        self._obs, self._actor_state, self._critic_state, self._mask, deterministic=False
                    )
                next_obs_np, reward_np, done_np, infos = self.runner.step(action.cpu().numpy())
                reward = torch.as_tensor(reward_np, dtype=torch.float32, device=self.device)
                next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=self.device)
                next_mask = torch.as_tensor(1.0 - done_np, dtype=torch.float32, device=self.device)

                self.buffer.insert(step, self._obs, self._mask, action, log_prob, reward, value)

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
                self._actor_state = next_actor_state * next_mask
                self._critic_state = next_critic_state * next_mask
                self._mask = next_mask
                self._total_steps += self.cfg.num_envs
                if self._total_steps >= phase_target:
                    break

            with torch.no_grad():
                next_value, _ = self.policy.predict_value(self._obs, self._critic_state, self._mask)
            self.buffer.compute_returns(next_value, self._mask, self.cfg.gamma, self.cfg.gae_lambda)
            latest_metrics = self._update_policy()
            self._update_count += 1

            if self._update_count % self.cfg.eval_interval_updates == 0:
                eval_metrics = self._evaluate(self.cfg.eval_episodes)
                latest_metrics.update(eval_metrics)
                # Cache eval-only metrics so every log line can show the most
                # recent value (eval runs only every eval_interval_updates).
                self._last_eval_metrics = eval_metrics

            log_payload = {
                "total_steps": int(self._total_steps),
                "updates": int(self._update_count),
                "completed_episodes": len(self._completed_returns),
                "recent_return_mean": float(np.mean(self._recent_returns)) if self._recent_returns else 0.0,
                "recent_episode_length_mean": float(np.mean(self._recent_lengths)) if self._recent_lengths else 0.0,
                "recent_success_rate": float(np.mean(self._recent_successes)) if self._recent_successes else 0.0,
                "recent_cast_mean": float(np.mean(self._recent_casts)) if self._recent_casts else 0.0,
                **getattr(self, "_last_eval_metrics", {}),
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
