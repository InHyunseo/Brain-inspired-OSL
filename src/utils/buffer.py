"""Episode replay buffer for RSAC and DRQN sequence sampling."""
import random
from collections import deque

import numpy as np


class EpisodeReplayBuffer:
    """Stores complete episodes and samples random length-seq_len chunks.

    Each transition is (obs, action, reward, next_obs, terminal). RSAC stores
    `action` as a float vector, DRQN stores it as an int — sampling preserves
    whatever dtype is in the tuple via `sample_continuous` / `sample`.
    """

    def __init__(self, cap_steps=150_000):
        self.cap_steps = int(cap_steps)
        self.episodes = deque()
        self.lengths = deque()
        self.n_steps = 0

    def add_episode(self, ep):
        if not ep:
            return
        self.episodes.append(ep)
        self.lengths.append(len(ep))
        self.n_steps += len(ep)
        while self.n_steps > self.cap_steps and self.episodes:
            self.n_steps -= self.lengths.popleft()
            self.episodes.popleft()

    def __len__(self):
        return self.n_steps

    def _sample_chunks(self, batch_size, seq_len, act_dtype):
        # Pick episodes proportionally to (length - seq_len + 1).
        eligible_idx = [i for i, L in enumerate(self.lengths) if L >= seq_len]
        if not eligible_idx:
            raise RuntimeError("Buffer insufficient for sampling sequences.")

        obs_seqs, act_seqs, rew_seqs, terminal_seqs = [], [], [], []
        for _ in range(batch_size):
            idx = random.choice(eligible_idx)
            ep = self.episodes[idx]
            s = random.randint(0, len(ep) - seq_len)
            chunk = ep[s:s + seq_len]

            o0 = chunk[0][0]
            obs_seq = [o0] + [tr[3] for tr in chunk]

            obs_seqs.append(np.asarray(obs_seq, dtype=np.float32))
            act_seqs.append(np.asarray([tr[1] for tr in chunk], dtype=act_dtype))
            rew_seqs.append(np.asarray([tr[2] for tr in chunk], dtype=np.float32))
            terminal_seqs.append(np.asarray([tr[4] for tr in chunk], dtype=np.float32))

        return (
            np.stack(obs_seqs, axis=0),
            np.stack(act_seqs, axis=0),
            np.stack(rew_seqs, axis=0),
            np.stack(terminal_seqs, axis=0),
        )

    def sample(self, batch_size, seq_len):
        """Sample with int64 actions (DRQN discrete)."""
        return self._sample_chunks(batch_size, seq_len, np.int64)

    def sample_continuous(self, batch_size, seq_len):
        """Sample with float32 actions (RSAC continuous + cast bit)."""
        return self._sample_chunks(batch_size, seq_len, np.float32)
