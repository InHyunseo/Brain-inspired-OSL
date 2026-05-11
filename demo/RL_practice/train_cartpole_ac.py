import os
import csv
import time
import json

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# --- plotting / gif ---
import matplotlib
matplotlib.use("Agg")  # headless 저장용
import matplotlib.pyplot as plt

# --- 상대경로를 통한 결과 저장 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(SCRIPT_DIR, "runs")


# -----------------------
# Model
# -----------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, act_dim)   # logits
        self.critic = nn.Linear(hidden, 1)        # V(s)

    def forward(self, x):
        h = self.body(x)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value


# -----------------------
# Reward shaping
# -----------------------
def shaped_reward(env, obs, terminated, truncated):
    u = env.unwrapped
    x_th = getattr(u, "x_threshold", 2.4)
    th_th = getattr(u, "theta_threshold_radians", 12 * np.pi / 180.0)

    x = float(obs[0])
    th = float(obs[2])

    r = 1.0 - (th / th_th) ** 2 - 0.01 * (x / x_th) ** 2

    if terminated:
        r -= 1.0
    if truncated and (not terminated):
        r += 100.0
    return r


# -----------------------
# Utils: moving average / plots / gif
# -----------------------
def moving_avg(x, window=50):
    if len(x) == 0:
        return np.array([])
    w = max(1, int(window))
    x = np.asarray(x, dtype=np.float32)
    if len(x) < w:
        return x
    kernel = np.ones(w, dtype=np.float32) / w
    return np.convolve(x, kernel, mode="valid")


def save_training_figure(run_dir, episodes, steps, returns, actor_losses, critic_losses, window=50):
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    ep = np.asarray(episodes, dtype=np.int32)
    steps = np.asarray(steps, dtype=np.float32)
    returns = np.asarray(returns, dtype=np.float32)
    actor_losses = np.asarray(actor_losses, dtype=np.float32)
    critic_losses = np.asarray(critic_losses, dtype=np.float32)

    fig = plt.figure(figsize=(10, 8))

    # (1) 학습곡선: steps + return (같은 축에 두기 애매해서 twin axis)
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(ep, steps, alpha=0.35, label="steps/episode")
    ma_steps = moving_avg(steps, window)
    if len(ma_steps) > 0:
        ax1.plot(ep[len(ep) - len(ma_steps):], ma_steps, label=f"steps MA({window})")
    ax1.set_xlabel("episode")
    ax1.set_ylabel("steps")
    ax1.grid(True, alpha=0.3)

    ax1b = ax1.twinx()
    ax1b.plot(ep, returns, alpha=0.25, linestyle="--", label="shaped_return")
    ma_ret = moving_avg(returns, window)
    if len(ma_ret) > 0:
        ax1b.plot(ep[len(ep) - len(ma_ret):], ma_ret, linestyle="--", label=f"return MA({window})")
    ax1b.set_ylabel("shaped_return")

    # 범례 합치기
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    # (2) 손실곡선: actor / critic
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(ep, actor_losses, alpha=0.4, label="actor_loss")
    ax2.plot(ep, critic_losses, alpha=0.4, label="critic_loss")
    ma_a = moving_avg(actor_losses, window)
    ma_c = moving_avg(critic_losses, window)
    if len(ma_a) > 0:
        ax2.plot(ep[len(ep) - len(ma_a):], ma_a, label=f"actor MA({window})")
    if len(ma_c) > 0:
        ax2.plot(ep[len(ep) - len(ma_c):], ma_c, label=f"critic MA({window})")
    ax2.set_xlabel("episode")
    ax2.set_ylabel("loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    out_path = os.path.join(plots_dir, "training_curves.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_gif(frames, path, fps=30):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    duration = 1.0 / float(fps)

    # imageio 우선, 없으면 PIL로 fallback
    try:
        import imageio.v2 as imageio
        imageio.mimsave(path, frames, duration=duration)
        return
    except Exception:
        pass

    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=int(1000 * duration),
        loop=0,
    )


@torch.no_grad()
def rollout_video_gif(env_id, model, device, seed, out_gif_path, fps=30, deterministic=True):
    env = gym.make(env_id, render_mode="rgb_array")
    obs, info = env.reset(seed=seed)

    frames = []
    frame = env.render()
    if frame is not None:
        frames.append(frame)

    done = False
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        logits, value = model(obs_t)

        if deterministic:
            action = torch.argmax(logits).item()
        else:
            dist = Categorical(logits=logits)
            action = int(dist.sample().item())

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        obs = next_obs

    env.close()
    save_gif(frames, out_gif_path, fps=fps)
    return frames


# -----------------------
# Train
# -----------------------
def train(
    env_id="CartPole-v1",
    seed=0,
    total_episodes=2000,
    gamma=0.99,
    lr=3e-4,
    value_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    reward_scale=0.01,

    # 저장 관련
    out_dir=DEFAULT_OUT_DIR,
    run_name=None,
    save_mid_episode=None,
    log_every=50,

    # 시각화 관련
    plot_ma_window=50,
    video_fps=30,
    video_eval_seed=123,

    force_cpu=False,
):
    device = torch.device("cpu") if force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # run dir
    if run_name is None:
        run_name = time.strftime("cartpole_ac_%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    media_dir = os.path.join(run_dir, "media")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(media_dir, exist_ok=True)

    # config 저장
    if save_mid_episode is None:
        save_mid_episode = total_episodes // 2

    config = dict(
        env_id=env_id, seed=seed, total_episodes=total_episodes, gamma=gamma, lr=lr,
        value_coef=value_coef, entropy_coef=entropy_coef, max_grad_norm=max_grad_norm,
        reward_scale=reward_scale, force_cpu=force_cpu,
        save_mid_episode=save_mid_episode,
        plot_ma_window=plot_ma_window,
        video_fps=video_fps,
        video_eval_seed=video_eval_seed,
    )
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # metrics.csv 준비 (+ total_loss 추가)
    metrics_path = os.path.join(run_dir, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "episode",
            "endured_step",
            "shaped_return",
            "success",
            "actor_loss",
            "critic_loss",
            "total_loss",
        ])

    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    huber = nn.SmoothL1Loss()

    # 체크포인트: 초기(학습 전) 저장 -> init 영상용
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "init.pt"))

    # in-memory hist (plot용)
    episodes = []
    ep_returns = []
    ep_steps_hist = []
    actor_loss_hist = []
    critic_loss_hist = []
    total_loss_hist = []

    for ep in range(1, total_episodes + 1):
        obs, info = env.reset(seed=seed + ep)

        log_probs = []
        values = []
        rewards = []
        entropies = []

        endured_step = 0
        done = False
        terminated = False
        truncated = False

        while not done:
            endured_step += 1
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

            logits, value = model(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            next_obs, _, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated

            r = shaped_reward(env, next_obs, terminated, truncated) * reward_scale

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor(r, dtype=torch.float32, device=device))
            entropies.append(entropy)

            obs = next_obs

        # episode metrics
        ep_return = float(torch.stack(rewards).sum().item())
        success = int(truncated and (not terminated))

        # returns G_t (Monte-Carlo)
        returns = []
        G = torch.tensor(0.0, device=device)
        for r in reversed(rewards):
            G = r + gamma * G
            returns.append(G)
        returns.reverse()
        returns = torch.stack(returns).detach()

        values_t = torch.stack(values)
        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)

        adv = (returns - values_t).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        actor_loss = -(log_probs_t * adv).mean()
        critic_loss = huber(values_t, returns)
        entropy_bonus = entropies_t.mean()

        total_loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy_bonus

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # csv append
        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                ep,
                endured_step,
                ep_return,
                success,
                float(actor_loss.item()),
                float(critic_loss.item()),
                float(total_loss.item()),
            ])

        # hist append
        episodes.append(ep)
        ep_returns.append(ep_return)
        ep_steps_hist.append(endured_step)
        actor_loss_hist.append(float(actor_loss.item()))
        critic_loss_hist.append(float(critic_loss.item()))
        total_loss_hist.append(float(total_loss.item()))

        # checkpoints
        if ep == save_mid_episode:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "mid.pt"))
        if ep == total_episodes:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "final.pt"))

        if ep % log_every == 0:
            mean_ret = np.mean(ep_returns[-log_every:])
            mean_steps = np.mean(ep_steps_hist[-log_every:])
            print(
                f"Episode {ep:4d} | mean_steps({log_every})={mean_steps:6.1f} | "
                f"mean_return({log_every})={mean_ret: .3f} | "
                f"actor={actor_loss.item():.4f} critic={critic_loss.item():.4f}"
            )

    env.close()

    # --------- (A) 학습곡선 + 손실곡선 figure 저장 ---------
    save_training_figure(
        run_dir,
        episodes=episodes,
        steps=ep_steps_hist,
        returns=ep_returns,
        actor_losses=actor_loss_hist,
        critic_losses=critic_loss_hist,
        window=plot_ma_window,
    )

    # --------- (B) init / mid / final 정책 rollout GIF 저장 ---------
    def load_ckpt(path):
        m = ActorCritic(obs_dim, act_dim).to(device)
        sd = torch.load(path, map_location=device)
        m.load_state_dict(sd)
        m.eval()
        return m

    init_model = load_ckpt(os.path.join(ckpt_dir, "init.pt"))
    mid_model = load_ckpt(os.path.join(ckpt_dir, "mid.pt"))
    final_model = load_ckpt(os.path.join(ckpt_dir, "final.pt"))

    frames_init = rollout_video_gif(
        env_id, init_model, device, seed=video_eval_seed,
        out_gif_path=os.path.join(media_dir, "init.gif"),
        fps=video_fps, deterministic=False
    )
    frames_mid = rollout_video_gif(
        env_id, mid_model, device, seed=video_eval_seed,
        out_gif_path=os.path.join(media_dir, "mid.gif"),
        fps=video_fps, deterministic=False
    )
    frames_final = rollout_video_gif(
        env_id, final_model, device, seed=video_eval_seed,
        out_gif_path=os.path.join(media_dir, "final.gif"),
        fps=video_fps, deterministic=False
    )

    return model, run_dir


if __name__ == "__main__":
    train()
