import random
import numpy as np

class EpisodeReplayBuffer:
    def __init__(self, cap_steps=150000):
        self.cap_steps = int(cap_steps)
        self.episodes = [] 
        self.n_steps = 0

    def add_episode(self, ep):
        """ep: list of (obs, action, reward, next_obs, done)"""
        if not ep: return
        self.episodes.append(ep)
        self.n_steps += len(ep)
        while self.n_steps > self.cap_steps and self.episodes:
            old = self.episodes.pop(0)
            self.n_steps -= len(old)

    def __len__(self):
        return self.n_steps

    def sample(self, batch_size, seq_len):
        candidates = [ep for ep in self.episodes if len(ep) >= seq_len]
        if not candidates:
            raise RuntimeError("Buffer insufficient for sampling sequences.")

        obs_seqs, act_seqs, rew_seqs, done_seqs = [], [], [], []

        for _ in range(batch_size):
            ep = random.choice(candidates)
            s = random.randint(0, len(ep) - seq_len)
            chunk = ep[s:s + seq_len]

            o0 = chunk[0][0]
            obs_seq = [o0] + [tr[3] for tr in chunk] # Next obs including
            
            obs_seqs.append(np.asarray(obs_seq, dtype=np.float32))
            act_seqs.append(np.asarray([tr[1] for tr in chunk], dtype=np.int64))
            rew_seqs.append(np.asarray([tr[2] for tr in chunk], dtype=np.float32))
            done_seqs.append(np.asarray([tr[4] for tr in chunk], dtype=np.float32))

        return (
            np.stack(obs_seqs, axis=0),
            np.stack(act_seqs, axis=0),
            np.stack(rew_seqs, axis=0),
            np.stack(done_seqs, axis=0),
        )