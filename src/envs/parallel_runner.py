from __future__ import annotations

import multiprocessing as mp
from typing import Any

import numpy as np

from src.envs.osl_env import OslEnv


def _env_worker(remote, env_config: dict[str, Any], seed: int) -> None:
    cfg = dict(env_config)
    cfg["seed"] = int(seed)
    env = OslEnv(cfg)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                obs, info = env.reset(seed=int(data))
                remote.send((np.asarray(obs, dtype=np.float32), info))
                continue
            if cmd == "step":
                obs, reward, terminated, truncated, info = env.step(np.asarray(data, dtype=np.float32))
                done = bool(terminated or truncated)
                if done:
                    info = dict(info)
                    info["terminal_observation"] = np.asarray(obs, dtype=np.float32).tolist()
                    obs, reset_info = env.reset()
                    info["reset_info"] = reset_info
                remote.send((np.asarray(obs, dtype=np.float32), float(reward), done, info))
                continue
            if cmd == "set_noise_stage":
                stage, strength = data
                env.set_noise_stage(int(stage), float(strength))
                remote.send(True)
                continue
            if cmd == "set_spawn_radius":
                min_r, max_r = data
                env.set_spawn_radius(float(min_r), float(max_r))
                remote.send(True)
                continue
            if cmd == "close":
                remote.close()
                break
            raise RuntimeError(f"Unknown worker command: {cmd}")
    finally:
        env.close()


class ParallelRunner:
    """Multi-process env runner. One subprocess per env, fork-based pipe IPC."""

    def __init__(self, env_config: dict[str, Any], num_envs: int, seed: int):
        self.num_envs = int(num_envs)
        self.base_seed = int(seed)
        probe_env = OslEnv({**dict(env_config), "seed": self.base_seed})
        self._observation_shape = probe_env.observation_space.shape
        self._action_shape = probe_env.action_space.shape
        probe_env.close()

        self.closed = False
        self.remotes = []
        self.processes = []
        ctx = mp.get_context("fork")
        for env_idx in range(self.num_envs):
            parent_remote, child_remote = ctx.Pipe()
            proc = ctx.Process(
                target=_env_worker,
                args=(child_remote, dict(env_config), self.base_seed + env_idx),
            )
            proc.daemon = True
            proc.start()
            child_remote.close()
            self.remotes.append(parent_remote)
            self.processes.append(proc)

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return self._observation_shape

    @property
    def action_shape(self) -> tuple[int, ...]:
        return self._action_shape

    def reset(self) -> tuple[np.ndarray, list[dict[str, Any]]]:
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("reset", self.base_seed + env_idx))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return np.asarray(obs, dtype=np.float32), list(infos)

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", np.asarray(action, dtype=np.float32)))
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, infos = zip(*results)
        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32).reshape(self.num_envs, 1),
            np.asarray(dones, dtype=np.float32).reshape(self.num_envs, 1),
            list(infos),
        )

    def set_noise_stage(self, stage: int, strength: float) -> None:
        for remote in self.remotes:
            remote.send(("set_noise_stage", (int(stage), float(strength))))
        for remote in self.remotes:
            remote.recv()

    def set_spawn_radius(self, min_radius_mm: float, max_radius_mm: float) -> None:
        for remote in self.remotes:
            remote.send(("set_spawn_radius", (float(min_radius_mm), float(max_radius_mm))))
        for remote in self.remotes:
            remote.recv()

    def close(self) -> None:
        if self.closed:
            return
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for proc in self.processes:
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
        self.closed = True


class VectorRunner:
    """In-process serial env runner. Same API as ParallelRunner."""

    def __init__(self, env_config: dict[str, Any], num_envs: int, seed: int):
        self.num_envs = int(num_envs)
        self.base_seed = int(seed)
        self.envs: list[OslEnv] = []
        for env_idx in range(self.num_envs):
            cfg = dict(env_config)
            cfg["seed"] = self.base_seed + env_idx
            self.envs.append(OslEnv(cfg))

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return self.envs[0].observation_space.shape

    @property
    def action_shape(self) -> tuple[int, ...]:
        return self.envs[0].action_space.shape

    def reset(self) -> tuple[np.ndarray, list[dict[str, Any]]]:
        obs_list = []
        info_list = []
        for env_idx, env in enumerate(self.envs):
            obs, info = env.reset(seed=self.base_seed + env_idx)
            obs_list.append(obs)
            info_list.append(info)
        return np.asarray(obs_list, dtype=np.float32), info_list

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        next_obs = []
        rewards = []
        dones = []
        infos = []
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
            done = bool(terminated or truncated)
            if done:
                info = dict(info)
                info["terminal_observation"] = np.asarray(obs, dtype=np.float32).tolist()
                obs, reset_info = env.reset()
                info["reset_info"] = reset_info
            next_obs.append(obs)
            rewards.append([reward])
            dones.append([1.0 if done else 0.0])
            infos.append(info)
        return (
            np.asarray(next_obs, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            infos,
        )

    def set_noise_stage(self, stage: int, strength: float) -> None:
        for env in self.envs:
            env.set_noise_stage(int(stage), float(strength))

    def set_spawn_radius(self, min_radius_mm: float, max_radius_mm: float) -> None:
        for env in self.envs:
            env.set_spawn_radius(float(min_radius_mm), float(max_radius_mm))

    def close(self) -> None:
        for env in self.envs:
            env.close()
