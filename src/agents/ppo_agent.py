"""sb3 RecurrentPPO wrapper for the 2D OSL connectome policy.

Mirrors the curriculum loop in ipynb/PPO_framework.ipynb: each phase calls
`learn_phase(env_kind, total_timesteps)` which (re)builds a 16-way
SubprocVecEnv + VecNormalize wrapped around StaticEnv/DynamicEnv, while the
underlying RecurrentPPO model + LSTM hidden state are preserved across phases.
"""
import os

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize

from src.models.networks import ConnectomeExtractor
from src.utils.factory import make_env_fn


class PPOAgent:
    def __init__(
        self,
        features_dim=180,
        lr=3e-4,
        batch_size=256,
        n_steps=128,
        ent_coef=0.01,
        tb_log_dir=None,
        seed=42,
    ):
        self.features_dim = int(features_dim)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.n_steps = int(n_steps)
        self.ent_coef = float(ent_coef)
        self.tb_log_dir = tb_log_dir
        self.seed = int(seed)

        self.policy_kwargs = dict(
            features_extractor_class=ConnectomeExtractor,
            features_extractor_kwargs=dict(features_dim=self.features_dim),
            net_arch=dict(pi=[128], qf=[128]),
        )

        self.model = None
        self.vec_env = None
        self._last_vecnorm_path = None

    def _build_vec_env(self, env_kind, n_envs, vecnorm_load_path=None):
        env_fns = [make_env_fn(env_kind, monitor=True) for _ in range(n_envs)]
        vec = SubprocVecEnv(env_fns)
        if vecnorm_load_path is not None and os.path.exists(vecnorm_load_path):
            vec = VecNormalize.load(vecnorm_load_path, vec)
            vec.training = True
        else:
            vec = VecNormalize(vec, norm_obs=False, norm_reward=True, clip_reward=10.0)
        return vec

    def learn_phase(
        self,
        env_kind,
        total_timesteps,
        n_envs=16,
        callback=None,
        vecnorm_load_path=None,
        reset_num_timesteps=None,
    ):
        """Run one curriculum phase. The first call constructs the model;
        subsequent calls swap the env and keep training."""
        vec = self._build_vec_env(env_kind, n_envs, vecnorm_load_path)

        if self.model is None:
            self.model = RecurrentPPO(
                "MlpLstmPolicy", vec,
                learning_rate=self.lr,
                batch_size=self.batch_size,
                n_steps=self.n_steps,
                ent_coef=self.ent_coef,
                policy_kwargs=self.policy_kwargs,
                verbose=1,
                tensorboard_log=self.tb_log_dir,
                seed=self.seed,
            )
            if reset_num_timesteps is None:
                reset_num_timesteps = True
        else:
            self.model.set_env(vec)
            if reset_num_timesteps is None:
                reset_num_timesteps = False

        self.vec_env = vec
        self.model.learn(
            total_timesteps=int(total_timesteps),
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
        )

    def save(self, model_path, vecnorm_path=None):
        if self.model is None:
            raise RuntimeError("PPOAgent.save called before any learn_phase.")
        self.model.save(model_path)
        if vecnorm_path is not None and self.vec_env is not None:
            self.vec_env.save(vecnorm_path)
            self._last_vecnorm_path = vecnorm_path

    def load(self, model_path, vecnorm_path=None, env_kind="dynamic_1.0", n_envs=1):
        """Load a model + VecNormalize stats for evaluation (single env)."""
        eval_vec = DummyVecEnv([make_env_fn(env_kind, monitor=True) for _ in range(n_envs)])
        if vecnorm_path is not None and os.path.exists(vecnorm_path):
            eval_vec = VecNormalize.load(vecnorm_path, eval_vec)
            eval_vec.training = False
            eval_vec.norm_reward = False
        self.vec_env = eval_vec
        self.model = RecurrentPPO.load(model_path, env=eval_vec)

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        if self.model is None:
            raise RuntimeError("PPOAgent.predict called before any learn_phase or load.")
        return self.model.predict(
            obs, state=state, episode_start=episode_start, deterministic=deterministic
        )

    @property
    def last_vecnorm_path(self):
        return self._last_vecnorm_path
