"""Networks.

Backbones for the 2D OSL stack:

- RSAC actors (continuous [v, omega] Gaussian + cast Bernoulli):
    - GRUActor       — GRU backbone
    - ConnectomeActor — 5-population connectome (ORN/PN/LN/KC/MBON, 24:7:4:54:1)
    - MLPActor       — feed-forward MLP (no recurrence)

- RSAC critic:
    - QCritic        — GRU-based recurrent twin-critic component

- DRQN/DQN Q-net:
    - QNet           — GRU (recurrent) or MLP (feed-forward) Q-net for discrete actions

- PPO feature extractor (sb3):
    - ConnectomeExtractor — same connectome math, as a BaseFeaturesExtractor
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Normal


# ---------------------------------------------------------------------------
# Connectome population sizing (ORN:PN:LN:KC:MBON = 24:7:4:54:1, base unit 90)
# ---------------------------------------------------------------------------

CONNECTOME_BASE = 90
CONNECTOME_RATIOS = (24, 7, 4, 54, 1)  # ORN, PN, LN, KC, MBON


def _population_sizes(total_size):
    """Return (n_orn, n_pn, n_ln, n_kc, n_mbon) for a connectome with the
    standard 24:7:4:54:1 ratio. total_size must be a positive multiple of 90."""
    if total_size % CONNECTOME_BASE != 0:
        raise ValueError(
            f"connectome hidden_size must be a multiple of {CONNECTOME_BASE}; got {total_size}."
        )
    k = total_size // CONNECTOME_BASE
    if k < 1:
        raise ValueError(f"connectome hidden_size must be >= {CONNECTOME_BASE}; got {total_size}.")
    return tuple(r * k for r in CONNECTOME_RATIOS)


# ---------------------------------------------------------------------------
# Shared connectome step (used by both ConnectomeActor and ConnectomeExtractor)
# ---------------------------------------------------------------------------


class _ConnectomeCell(nn.Module):
    """One outer step = `inner_steps` tanh updates over ORN/PN/LN/KC/MBON.

    Connections (matching ipynb/PPO_framework.ipynb):
        ORN  <- W_oto(ORN)  + W_lto(LN)  + x_t
        PN   <- W_otp(ORN') + W_ltp(LN)  + W_ptp(PN)
        LN   <- W_otl(ORN') + W_ptl(PN') + W_ltl(LN)
        KC   <- W_ktk(KC)   + W_mtk(MBON)+ W_ptk(PN')
        MBON <- W_ktm(KC')
    Each population has a bias term and tanh activation.
    """

    def __init__(self, obs_dim, hidden_size, inner_steps=4):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.inner_steps = int(max(1, inner_steps))

        n_orn, n_pn, n_ln, n_kc, n_mbon = _population_sizes(self.hidden_size)
        self.n_orn, self.n_pn, self.n_ln, self.n_kc, self.n_mbon = n_orn, n_pn, n_ln, n_kc, n_mbon

        self.in_orn = nn.Linear(obs_dim, n_orn)

        self.W_oto = nn.Linear(n_orn, n_orn, bias=False)
        self.W_lto = nn.Linear(n_ln, n_orn, bias=False)

        self.W_otp = nn.Linear(n_orn, n_pn, bias=False)
        self.W_ltp = nn.Linear(n_ln, n_pn, bias=False)
        self.W_ptp = nn.Linear(n_pn, n_pn, bias=False)

        self.W_otl = nn.Linear(n_orn, n_ln, bias=False)
        self.W_ptl = nn.Linear(n_pn, n_ln, bias=False)
        self.W_ltl = nn.Linear(n_ln, n_ln, bias=False)

        self.W_ktk = nn.Linear(n_kc, n_kc, bias=False)
        self.W_mtk = nn.Linear(n_mbon, n_kc, bias=False)
        self.W_ptk = nn.Linear(n_pn, n_kc, bias=False)

        self.W_ktm = nn.Linear(n_kc, n_mbon, bias=False)

        self.b_orn = nn.Parameter(torch.zeros(n_orn))
        self.b_pn = nn.Parameter(torch.zeros(n_pn))
        self.b_ln = nn.Parameter(torch.zeros(n_ln))
        self.b_kc = nn.Parameter(torch.zeros(n_kc))
        self.b_mbon = nn.Parameter(torch.zeros(n_mbon))

    def init_state(self, batch_size, device, dtype):
        z = lambda n: torch.zeros(batch_size, n, device=device, dtype=dtype)
        return z(self.n_orn), z(self.n_pn), z(self.n_ln), z(self.n_kc), z(self.n_mbon)

    def step_inner(self, h_orn, h_pn, h_ln, h_kc, h_mbon, x_t):
        for _ in range(self.inner_steps):
            orn = torch.tanh(self.W_oto(h_orn) + self.W_lto(h_ln) + x_t + self.b_orn)
            pn = torch.tanh(self.W_otp(orn) + self.W_ltp(h_ln) + self.W_ptp(h_pn) + self.b_pn)
            ln = torch.tanh(self.W_otl(orn) + self.W_ptl(pn) + self.W_ltl(h_ln) + self.b_ln)
            kc = torch.tanh(self.W_ktk(h_kc) + self.W_mtk(h_mbon) + self.W_ptk(pn) + self.b_kc)
            mbon = torch.tanh(self.W_ktm(kc) + self.b_mbon)
            h_orn, h_pn, h_ln, h_kc, h_mbon = orn, pn, ln, kc, mbon
        return h_orn, h_pn, h_ln, h_kc, h_mbon


class _ConnectomeBackbone(nn.Module):
    """Sequence wrapper around `_ConnectomeCell`.

    forward(obs, h=None) where obs is (B, T, obs_dim) or (B, obs_dim).
    Returns (y, h2) with y = readout(h_mbon) per timestep, shape (B, T, hidden_size).
    Hidden state h is a packed (1, B, hidden_size) tensor for API compatibility.
    """

    def __init__(self, obs_dim, hidden_size, inner_steps=4):
        super().__init__()
        self.cell = _ConnectomeCell(obs_dim, hidden_size, inner_steps)
        self.hidden_size = self.cell.hidden_size
        self.readout = nn.Linear(self.cell.n_mbon, self.hidden_size)

    def _split(self, h_flat):
        c = self.cell
        i0, i1, i2, i3 = c.n_orn, c.n_orn + c.n_pn, c.n_orn + c.n_pn + c.n_ln, c.n_orn + c.n_pn + c.n_ln + c.n_kc
        return (h_flat[:, :i0], h_flat[:, i0:i1], h_flat[:, i1:i2],
                h_flat[:, i2:i3], h_flat[:, i3:])

    def _pack(self, h_orn, h_pn, h_ln, h_kc, h_mbon):
        return torch.cat([h_orn, h_pn, h_ln, h_kc, h_mbon], dim=-1)

    def _parse_hidden(self, h, batch_size, device, dtype):
        if h is None:
            return self.cell.init_state(batch_size, device, dtype)
        if h.dim() == 3:
            if h.size(0) != 1:
                raise ValueError("connectome hidden state expects first dim 1.")
            h_flat = h[0]
        elif h.dim() == 2:
            h_flat = h
        else:
            raise ValueError("connectome hidden state must be rank-2 or rank-3.")
        if h_flat.size(0) != batch_size or h_flat.size(1) != self.hidden_size:
            raise ValueError(
                f"connectome hidden state shape mismatch: got {tuple(h_flat.shape)}, "
                f"expected ({batch_size}, {self.hidden_size})."
            )
        return self._split(h_flat.to(device=device, dtype=dtype))

    def forward(self, obs, h=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        bsz, seq_len, _ = obs.shape
        h_orn, h_pn, h_ln, h_kc, h_mbon = self._parse_hidden(h, bsz, obs.device, obs.dtype)

        outputs = []
        for t in range(seq_len):
            x_t = self.cell.in_orn(obs[:, t, :])
            h_orn, h_pn, h_ln, h_kc, h_mbon = self.cell.step_inner(h_orn, h_pn, h_ln, h_kc, h_mbon, x_t)
            outputs.append(self.readout(h_mbon).unsqueeze(1))

        y = torch.cat(outputs, dim=1)
        h2 = self._pack(h_orn, h_pn, h_ln, h_kc, h_mbon).unsqueeze(0)
        return y, h2


# ---------------------------------------------------------------------------
# Hybrid policy head — shared by all three SAC actor backbones
# ---------------------------------------------------------------------------


def _hybrid_sample(mu, log_std, cast_logit, action_low, action_high, cast_temperature):
    std = log_std.exp()
    normal = Normal(mu, std)
    x = normal.rsample()
    y = torch.tanh(x)

    cont_low, cont_high = action_low[:2], action_high[:2]
    cont_scale = (cont_high - cont_low) * 0.5
    cont_bias = (cont_high + cont_low) * 0.5
    cont_action = y * cont_scale + cont_bias

    cont_log_prob = normal.log_prob(x) - torch.log(cont_scale * (1.0 - y.pow(2)) + 1e-6)
    cont_log_prob = cont_log_prob.sum(dim=-1, keepdim=True)

    # Straight-through Bernoulli sampling for cast so Q-gradient reaches cast_logit.
    u = torch.clamp(torch.rand_like(cast_logit), 1e-6, 1.0 - 1e-6)
    logistic_noise = torch.log(u) - torch.log1p(-u)
    cast_soft = torch.sigmoid((cast_logit + logistic_noise) / cast_temperature)
    cast_hard = (cast_soft > 0.5).float()
    cast_action = cast_hard + cast_soft - cast_soft.detach()

    bern = Bernoulli(logits=cast_logit)
    disc_log_prob = bern.log_prob(cast_hard)

    action = torch.cat([cont_action, cast_action], dim=-1)
    log_prob = (cont_log_prob + disc_log_prob).squeeze(-1)
    cast_prob = torch.sigmoid(cast_logit)
    return action, log_prob, mu, cast_prob


def _hybrid_deterministic(mu, cast_logit, action_low, action_high):
    y = torch.tanh(mu)
    cont_low, cont_high = action_low[:2], action_high[:2]
    cont_scale = (cont_high - cont_low) * 0.5
    cont_bias = (cont_high + cont_low) * 0.5
    cont_action = y * cont_scale + cont_bias
    cast_action = (cast_logit > 0.0).float()
    return torch.cat([cont_action, cast_action], dim=-1)


class _HybridHead(nn.Module):
    """mu / log_std / cast_logit heads sitting on top of a (B, T, hidden) feature stream."""

    def __init__(self, hidden, cont_act_dim, log_std_min=-5.0, log_std_max=2.0):
        super().__init__()
        self.mu = nn.Linear(hidden, cont_act_dim)
        self.log_std = nn.Linear(hidden, cont_act_dim)
        self.cast_logit = nn.Linear(hidden, 1)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def forward(self, y):
        mu = self.mu(y)
        log_std = torch.clamp(self.log_std(y), self.log_std_min, self.log_std_max)
        cast_logit = self.cast_logit(y)
        return mu, log_std, cast_logit


# ---------------------------------------------------------------------------
# RSAC actors
# ---------------------------------------------------------------------------


class GRUActor(nn.Module):
    """Hybrid [v, omega, cast] policy with a GRU backbone."""

    def __init__(self, obs_dim, cont_act_dim=2, hidden=147, cast_temperature=0.5):
        super().__init__()
        self.rnn = nn.GRU(input_size=obs_dim, hidden_size=hidden, batch_first=True)
        self.head = _HybridHead(hidden, cont_act_dim)
        self.cast_temperature = float(max(cast_temperature, 1e-3))

    def forward(self, obs, h=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        y, h2 = self.rnn(obs, h)
        mu, log_std, cast_logit = self.head(y)
        return mu, log_std, cast_logit, h2

    def sample(self, obs, action_low, action_high, h=None):
        mu, log_std, cast_logit, h2 = self.forward(obs, h)
        action, log_prob, mu_out, cast_prob = _hybrid_sample(
            mu, log_std, cast_logit, action_low, action_high, self.cast_temperature
        )
        return action, log_prob, h2, mu_out, cast_prob

    def deterministic(self, obs, action_low, action_high, h=None):
        mu, _, cast_logit, h2 = self.forward(obs, h)
        return _hybrid_deterministic(mu, cast_logit, action_low, action_high), h2


class ConnectomeActor(nn.Module):
    """Hybrid policy with the 5-population connectome backbone."""

    def __init__(self, obs_dim, cont_act_dim=2, hidden=180, cast_temperature=0.5, connectome_steps=4):
        super().__init__()
        self.backbone = _ConnectomeBackbone(obs_dim, hidden, inner_steps=connectome_steps)
        self.head = _HybridHead(hidden, cont_act_dim)
        self.cast_temperature = float(max(cast_temperature, 1e-3))

    def forward(self, obs, h=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        y, h2 = self.backbone(obs, h)
        mu, log_std, cast_logit = self.head(y)
        return mu, log_std, cast_logit, h2

    def sample(self, obs, action_low, action_high, h=None):
        mu, log_std, cast_logit, h2 = self.forward(obs, h)
        action, log_prob, mu_out, cast_prob = _hybrid_sample(
            mu, log_std, cast_logit, action_low, action_high, self.cast_temperature
        )
        return action, log_prob, h2, mu_out, cast_prob

    def deterministic(self, obs, action_low, action_high, h=None):
        mu, _, cast_logit, h2 = self.forward(obs, h)
        return _hybrid_deterministic(mu, cast_logit, action_low, action_high), h2


class MLPActor(nn.Module):
    """Hybrid policy with a feed-forward MLP backbone (no recurrence).

    The `h` argument is accepted and returned (as None) only to keep the actor
    API uniform with GRUActor / ConnectomeActor.
    """

    def __init__(self, obs_dim, cont_act_dim=2, hidden=147, cast_temperature=0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = _HybridHead(hidden, cont_act_dim)
        self.cast_temperature = float(max(cast_temperature, 1e-3))

    def forward(self, obs, h=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        y = self.mlp(obs)
        mu, log_std, cast_logit = self.head(y)
        return mu, log_std, cast_logit, None

    def sample(self, obs, action_low, action_high, h=None):
        mu, log_std, cast_logit, _ = self.forward(obs, h)
        action, log_prob, mu_out, cast_prob = _hybrid_sample(
            mu, log_std, cast_logit, action_low, action_high, self.cast_temperature
        )
        return action, log_prob, None, mu_out, cast_prob

    def deterministic(self, obs, action_low, action_high, h=None):
        mu, _, cast_logit, _ = self.forward(obs, h)
        return _hybrid_deterministic(mu, cast_logit, action_low, action_high), None


# ---------------------------------------------------------------------------
# RSAC twin-critic component
# ---------------------------------------------------------------------------


class QCritic(nn.Module):
    """GRU-based Q(s, a). Used twice (q1, q2) for SAC twin-critic."""

    def __init__(self, obs_dim, act_dim, hidden=147):
        super().__init__()
        self.rnn = nn.GRU(input_size=obs_dim, hidden_size=hidden, batch_first=True)
        self.fc1 = nn.Linear(hidden + act_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.q = nn.Linear(hidden, 1)

    def forward(self, obs, act, h=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        if act.dim() == 2:
            act = act.unsqueeze(1)
        y, h2 = self.rnn(obs, h)
        z = torch.cat([y, act], dim=-1)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        q = self.q(z).squeeze(-1)
        return q, h2


# ---------------------------------------------------------------------------
# DRQN / DQN unified Q-net (recurrent toggle)
# ---------------------------------------------------------------------------


class QNet(nn.Module):
    """Q-net for the unified DRQN/DQN agent.

    - recurrent=True : GRU backbone (DRQN). API: forward(x, h) where x is
      (B, T, obs_dim) or (B, obs_dim). Returns (q, h2) with q shape (B, T, A).
    - recurrent=False: MLP backbone (DQN). API: forward(x, h=None). Returns
      (q, None) with q shape matching x (per-timestep if x is (B, T, D)).

    `forward_last` returns the Q-values of the final timestep only, for use in
    online action selection.
    """

    def __init__(self, obs_dim, n_actions, hidden=147, recurrent=True):
        super().__init__()
        self.recurrent = bool(recurrent)
        if self.recurrent:
            self.rnn = nn.GRU(input_size=obs_dim, hidden_size=hidden, batch_first=True)
            self.head = nn.Linear(hidden, n_actions)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
            )
            self.head = nn.Linear(hidden, n_actions)

    def forward(self, x, h=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if self.recurrent:
            y, h2 = self.rnn(x, h)
            return self.head(y), h2
        return self.head(self.mlp(x)), None

    def forward_last(self, x, h=None):
        q, h2 = self.forward(x, h)
        return q[:, -1], h2


# ---------------------------------------------------------------------------
# PPO feature extractor (sb3)
# ---------------------------------------------------------------------------

try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except ImportError:  # sb3 optional at import time
    BaseFeaturesExtractor = None


if BaseFeaturesExtractor is not None:

    class ConnectomeExtractor(BaseFeaturesExtractor):
        """Connectome feature extractor for sb3 RecurrentPPO.

        features_dim must be a positive multiple of 90 (default 180).
        Internally runs `inner_steps=4` connectome updates per env step,
        starting from zero state each call (sb3 manages the LSTM state
        separately on top of these features).
        """

        def __init__(self, observation_space, features_dim=180):
            super().__init__(observation_space, features_dim)
            obs_dim = int(observation_space.shape[0])
            self.cell = _ConnectomeCell(obs_dim, features_dim, inner_steps=4)
            self.readout = nn.Linear(self.cell.n_mbon, features_dim)

        def forward(self, observations):
            batch_size = observations.shape[0]
            device = observations.device
            dtype = observations.dtype
            h_orn, h_pn, h_ln, h_kc, h_mbon = self.cell.init_state(batch_size, device, dtype)
            x_t = self.cell.in_orn(observations)
            h_orn, h_pn, h_ln, h_kc, h_mbon = self.cell.step_inner(
                h_orn, h_pn, h_ln, h_kc, h_mbon, x_t
            )
            return self.readout(h_mbon)
