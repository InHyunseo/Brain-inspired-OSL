import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.distributions import Normal

class RQNet(nn.Module):
    """Recurrent Q-Network (GRU based)"""
    def __init__(self, obs_dim, act_dim, hidden=147):
        super().__init__()
        self.gru = nn.GRU(input_size=obs_dim, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, act_dim)

    def forward(self, x, h=None):
        # x: (B, T, obs_dim) or (B, obs_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        y, h2 = self.gru(x, h)          # y: (B, T, H)
        q = self.head(y)                # q: (B, T, A)
        return q, h2

    def forward_last(self, x, h=None):
        q, h2 = self.forward(x, h)
        return q[:, -1], h2             # (B, A), (1, B, H)

class DQN(nn.Module):
    """Simple MLP DQN"""
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.layer1 = nn.Linear(obs_dim, hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.layer3 = nn.Linear(hidden, act_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def _build_gru(input_size, hidden_size):
    return nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)


def _population_sizes(total_size, ratios=(3, 1, 4, 1)):
    """Split hidden size into ORN/PN/KC/LN using ratio 3:1:4:1."""
    if total_size < 4:
        raise ValueError("hidden must be >= 4 for Connectome backbone.")
    if len(ratios) != 4:
        raise ValueError("ratios must have 4 elements for ORN/PN/KC/LN.")
    if any(r <= 0 for r in ratios):
        raise ValueError("ratios must be positive.")

    # Keep every population non-empty, then distribute the remainder by ratio.
    sizes = [1, 1, 1, 1]
    rem_hidden = int(total_size - 4)
    ratio_sum = float(sum(ratios))
    raw = [rem_hidden * (float(r) / ratio_sum) for r in ratios]
    add = [int(v) for v in raw]
    for i in range(4):
        sizes[i] += add[i]

    leftover = rem_hidden - sum(add)
    frac_order = sorted(range(4), key=lambda i: raw[i] - add[i], reverse=True)
    for i in frac_order[:leftover]:
        sizes[i] += 1

    return sizes[0], sizes[1], sizes[2], sizes[3]


# Connectome2: ORN:PN:LN:KC:MBON = 24:7:4:54:1, base unit 90.
CONNECTOME2_BASE = 90
CONNECTOME2_RATIOS = (24, 7, 4, 54, 1)  # ORN, PN, LN, KC, MBON


def _population_sizes_v2(total_size):
    """Split hidden size into ORN/PN/LN/KC/MBON using ratio 24:7:4:54:1."""
    if total_size % CONNECTOME2_BASE != 0:
        raise ValueError(
            f"Connectome2 hidden_size must be a multiple of {CONNECTOME2_BASE}; got {total_size}."
        )
    k = total_size // CONNECTOME2_BASE
    if k < 2:
        raise ValueError(
            f"Connectome2 requires hidden_size >= {2 * CONNECTOME2_BASE} (MBON >= 2); got {total_size}."
        )

    n_orn = CONNECTOME2_RATIOS[0] * k
    n_pn = CONNECTOME2_RATIOS[1] * k
    n_ln = CONNECTOME2_RATIOS[2] * k
    n_kc = CONNECTOME2_RATIOS[3] * k
    n_mbon = CONNECTOME2_RATIOS[4] * k
    return n_orn, n_pn, n_ln, n_kc, n_mbon


class _ConnectomeBackbone(nn.Module):
    """
    Connectome-style recurrent backbone with four interacting populations:
    ORN, PN, KC, LN.
    Hidden state is packed as a single tensor so it fits RSAC actor API.
    """
    def __init__(self, obs_dim, hidden_size, inner_steps=4):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.inner_steps = int(max(1, inner_steps))

        n_orn, n_pn, n_kc, n_ln = _population_sizes(self.hidden_size)
        self.n_orn = n_orn
        self.n_pn = n_pn
        self.n_kc = n_kc
        self.n_ln = n_ln

        self.in_orn = nn.Linear(obs_dim, n_orn)

        # ORN <- PN/ORN/LN + input
        self.W_pto = nn.Linear(n_pn, n_orn, bias=False)
        self.W_oto = nn.Linear(n_orn, n_orn, bias=False)
        self.W_lto = nn.Linear(n_ln, n_orn, bias=False)

        # PN <- ORN/KC/LN/PN
        self.W_otp = nn.Linear(n_orn, n_pn, bias=False)
        self.W_ktp = nn.Linear(n_kc, n_pn, bias=False)
        self.W_ltp = nn.Linear(n_ln, n_pn, bias=False)
        self.W_ptp = nn.Linear(n_pn, n_pn, bias=False)

        # LN <- PN/ORN/LN
        self.W_ptl = nn.Linear(n_pn, n_ln, bias=False)
        self.W_otl = nn.Linear(n_orn, n_ln, bias=False)
        self.W_ltl = nn.Linear(n_ln, n_ln, bias=False)

        # KC <- PN/KC
        self.W_ptk = nn.Linear(n_pn, n_kc, bias=False)
        self.W_ktk = nn.Linear(n_kc, n_kc, bias=False)

        self.b_orn = nn.Parameter(torch.zeros(n_orn))
        self.b_pn = nn.Parameter(torch.zeros(n_pn))
        self.b_kc = nn.Parameter(torch.zeros(n_kc))
        self.b_ln = nn.Parameter(torch.zeros(n_ln))

        # Keep readout from KC as in the original template.
        self.readout = nn.Linear(n_kc, self.hidden_size)

    def _split_state(self, h_flat):
        i0 = self.n_orn
        i1 = i0 + self.n_pn
        i2 = i1 + self.n_kc
        h_orn = h_flat[:, :i0]
        h_pn = h_flat[:, i0:i1]
        h_kc = h_flat[:, i1:i2]
        h_ln = h_flat[:, i2:]
        return h_orn, h_pn, h_kc, h_ln

    def _pack_state(self, h_orn, h_pn, h_kc, h_ln):
        return torch.cat([h_orn, h_pn, h_kc, h_ln], dim=-1)

    def _init_state(self, batch_size, device, dtype):
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def _parse_hidden(self, h, batch_size, device, dtype):
        if h is None:
            return self._init_state(batch_size, device, dtype)
        if h.dim() == 3:
            if h.size(0) != 1:
                raise ValueError("Connectome hidden state expects first dim 1.")
            h_flat = h[0]
        elif h.dim() == 2:
            h_flat = h
        else:
            raise ValueError("Connectome hidden state must be rank-2 or rank-3 tensor.")
        if h_flat.size(0) != batch_size or h_flat.size(1) != self.hidden_size:
            raise ValueError(
                f"Connectome hidden state shape mismatch: got {tuple(h_flat.shape)}, "
                f"expected ({batch_size}, {self.hidden_size})."
            )
        return h_flat.to(device=device, dtype=dtype)

    def step(self, h_orn, h_pn, h_kc, h_ln, x_t):
        orn_in = self.W_pto(h_pn) + self.W_oto(h_orn) + self.W_lto(h_ln) + x_t
        orn_next = torch.tanh(orn_in + self.b_orn)

        pn_in = self.W_otp(orn_next) + self.W_ktp(h_kc) + self.W_ltp(h_ln) + self.W_ptp(h_pn)
        pn_next = torch.tanh(pn_in + self.b_pn)

        ln_in = self.W_ptl(pn_next) + self.W_otl(orn_next) + self.W_ltl(h_ln)
        ln_next = torch.tanh(ln_in + self.b_ln)

        kc_in = self.W_ptk(pn_next) + self.W_ktk(h_kc)
        kc_next = torch.tanh(kc_in + self.b_kc)

        return orn_next, pn_next, kc_next, ln_next

    def forward(self, obs, h=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        bsz, seq_len, _ = obs.shape
        device = obs.device
        dtype = obs.dtype

        h_flat = self._parse_hidden(h, bsz, device, dtype)
        h_orn, h_pn, h_kc, h_ln = self._split_state(h_flat)

        outputs = []
        for t in range(seq_len):
            x_t = self.in_orn(obs[:, t, :])
            for _ in range(self.inner_steps):
                h_orn, h_pn, h_kc, h_ln = self.step(h_orn, h_pn, h_kc, h_ln, x_t)
            outputs.append(self.readout(h_kc).unsqueeze(1))

        y = torch.cat(outputs, dim=1)
        h2 = self._pack_state(h_orn, h_pn, h_kc, h_ln).unsqueeze(0)
        return y, h2


class _Connectome2Backbone(nn.Module):
    """
    Connectome2 recurrent backbone with five interacting populations:
    ORN, PN, LN, KC, MBON.
    Hidden state is packed as a single tensor so it fits RSAC actor API.
    """
    def __init__(self, obs_dim, hidden_size, inner_steps=4):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.inner_steps = int(max(1, inner_steps))

        n_orn, n_pn, n_ln, n_kc, n_mbon = _population_sizes_v2(self.hidden_size)
        self.n_orn = n_orn
        self.n_pn = n_pn
        self.n_ln = n_ln
        self.n_kc = n_kc
        self.n_mbon = n_mbon

        self.in_orn = nn.Linear(obs_dim, n_orn)

        # ORN <- ORN, LN, input
        self.W_oto = nn.Linear(n_orn, n_orn, bias=False)
        self.W_lto = nn.Linear(n_ln, n_orn, bias=False)

        # PN <- ORN, LN, PN
        self.W_otp = nn.Linear(n_orn, n_pn, bias=False)
        self.W_ltp = nn.Linear(n_ln, n_pn, bias=False)
        self.W_ptp = nn.Linear(n_pn, n_pn, bias=False)

        # LN <- ORN, PN, LN
        self.W_otl = nn.Linear(n_orn, n_ln, bias=False)
        self.W_ptl = nn.Linear(n_pn, n_ln, bias=False)
        self.W_ltl = nn.Linear(n_ln, n_ln, bias=False)

        # KC <- KC, MBON, PN
        self.W_ktk = nn.Linear(n_kc, n_kc, bias=False)
        self.W_mtk = nn.Linear(n_mbon, n_kc, bias=False)
        self.W_ptk = nn.Linear(n_pn, n_kc, bias=False)

        # MBON <- KC
        self.W_ktm = nn.Linear(n_kc, n_mbon, bias=False)

        self.b_orn = nn.Parameter(torch.zeros(n_orn))
        self.b_pn = nn.Parameter(torch.zeros(n_pn))
        self.b_ln = nn.Parameter(torch.zeros(n_ln))
        self.b_kc = nn.Parameter(torch.zeros(n_kc))
        self.b_mbon = nn.Parameter(torch.zeros(n_mbon))

        self.readout = nn.Linear(n_mbon, self.hidden_size)

    def _split_state(self, h_flat):
        i0 = self.n_orn
        i1 = i0 + self.n_pn
        i2 = i1 + self.n_ln
        i3 = i2 + self.n_kc
        h_orn = h_flat[:, :i0]
        h_pn = h_flat[:, i0:i1]
        h_ln = h_flat[:, i1:i2]
        h_kc = h_flat[:, i2:i3]
        h_mbon = h_flat[:, i3:]
        return h_orn, h_pn, h_ln, h_kc, h_mbon

    def _pack_state(self, h_orn, h_pn, h_ln, h_kc, h_mbon):
        return torch.cat([h_orn, h_pn, h_ln, h_kc, h_mbon], dim=-1)

    def _init_state(self, batch_size, device, dtype):
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def _parse_hidden(self, h, batch_size, device, dtype):
        if h is None:
            return self._init_state(batch_size, device, dtype)
        if h.dim() == 3:
            if h.size(0) != 1:
                raise ValueError("Connectome2 hidden state expects first dim 1.")
            h_flat = h[0]
        elif h.dim() == 2:
            h_flat = h
        else:
            raise ValueError("Connectome2 hidden state must be rank-2 or rank-3 tensor.")
        if h_flat.size(0) != batch_size or h_flat.size(1) != self.hidden_size:
            raise ValueError(
                f"Connectome2 hidden state shape mismatch: got {tuple(h_flat.shape)}, "
                f"expected ({batch_size}, {self.hidden_size})."
            )
        return h_flat.to(device=device, dtype=dtype)

    def step(self, h_orn, h_pn, h_ln, h_kc, h_mbon, x_t):
        orn_next = torch.tanh(self.W_oto(h_orn) + self.W_lto(h_ln) + x_t + self.b_orn)
        pn_next = torch.tanh(self.W_otp(orn_next) + self.W_ltp(h_ln) + self.W_ptp(h_pn) + self.b_pn)
        ln_next = torch.tanh(self.W_otl(orn_next) + self.W_ptl(pn_next) + self.W_ltl(h_ln) + self.b_ln)
        kc_next = torch.tanh(self.W_ktk(h_kc) + self.W_mtk(h_mbon) + self.W_ptk(pn_next) + self.b_kc)
        mbon_next = torch.tanh(self.W_ktm(kc_next) + self.b_mbon)
        return orn_next, pn_next, ln_next, kc_next, mbon_next

    def forward(self, obs, h=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        bsz, seq_len, _ = obs.shape
        device = obs.device
        dtype = obs.dtype

        h_flat = self._parse_hidden(h, bsz, device, dtype)
        h_orn, h_pn, h_ln, h_kc, h_mbon = self._split_state(h_flat)

        outputs = []
        for t in range(seq_len):
            x_t = self.in_orn(obs[:, t, :])
            for _ in range(self.inner_steps):
                h_orn, h_pn, h_ln, h_kc, h_mbon = self.step(h_orn, h_pn, h_ln, h_kc, h_mbon, x_t)
            outputs.append(self.readout(h_mbon).unsqueeze(1))

        y = torch.cat(outputs, dim=1)
        h2 = self._pack_state(h_orn, h_pn, h_ln, h_kc, h_mbon).unsqueeze(0)
        return y, h2


class RecurrentGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, action_low, action_high, hidden=147, log_std_min=-5.0, log_std_max=2.0):
        super().__init__()
        self.rnn = _build_gru(obs_dim, hidden)
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale", (high - low) * 0.5)
        self.register_buffer("action_bias", (high + low) * 0.5)

    def forward(self, obs, h=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        y, h2 = self.rnn(obs, h)
        mu = self.mu(y)
        log_std = torch.clamp(self.log_std(y), self.log_std_min, self.log_std_max)
        return mu, log_std, h2

    def sample(self, obs, h=None):
        mu, log_std, h2 = self.forward(obs, h)
        std = log_std.exp()
        normal = Normal(mu, std)
        x = normal.rsample()
        y = torch.tanh(x)
        action = y * self.action_scale + self.action_bias

        # tanh-squashed Gaussian log-prob with change-of-variables correction
        log_prob = normal.log_prob(x) - torch.log(self.action_scale * (1.0 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        return action, log_prob, h2, mu

    def deterministic(self, obs, h=None):
        mu, _, h2 = self.forward(obs, h)
        y = torch.tanh(mu)
        action = y * self.action_scale + self.action_bias
        return action, h2


class RecurrentQCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=147):
        super().__init__()
        self.rnn = _build_gru(obs_dim, hidden)
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


class RecurrentHybridActor(nn.Module):
    """
    Hybrid policy for [v, omega, cast]:
    - continuous part: tanh-squashed Gaussian for [v, omega]
    - discrete part: Bernoulli for cast start decision
    """
    def __init__(
        self,
        obs_dim,
        cont_act_dim,
        hidden=147,
        log_std_min=-5.0,
        log_std_max=2.0,
        cast_temperature=0.5,
    ):
        super().__init__()
        self.rnn = _build_gru(obs_dim, hidden)
        self.mu = nn.Linear(hidden, cont_act_dim)
        self.log_std = nn.Linear(hidden, cont_act_dim)
        self.cast_logit = nn.Linear(hidden, 1)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.cast_temperature = float(max(cast_temperature, 1e-3))

    def forward(self, obs, h=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        y, h2 = self.rnn(obs, h)
        mu = self.mu(y)
        log_std = torch.clamp(self.log_std(y), self.log_std_min, self.log_std_max)
        cast_logit = self.cast_logit(y)
        return mu, log_std, cast_logit, h2

    def sample(self, obs, action_low, action_high, h=None):
        mu, log_std, cast_logit, h2 = self.forward(obs, h)
        std = log_std.exp()
        normal = Normal(mu, std)
        x = normal.rsample()
        y = torch.tanh(x)

        cont_low = action_low[:2]
        cont_high = action_high[:2]
        cont_scale = (cont_high - cont_low) * 0.5
        cont_bias = (cont_high + cont_low) * 0.5
        cont_action = y * cont_scale + cont_bias

        cont_log_prob = normal.log_prob(x) - torch.log(cont_scale * (1.0 - y.pow(2)) + 1e-6)
        cont_log_prob = cont_log_prob.sum(dim=-1, keepdim=True)

        # Straight-through Bernoulli sampling for cast so Q-gradient reaches cast_logit.
        # Forward uses hard {0,1}, backward uses relaxed sample gradient.
        u = torch.rand_like(cast_logit)
        u = torch.clamp(u, 1e-6, 1.0 - 1e-6)
        logistic_noise = torch.log(u) - torch.log1p(-u)
        cast_soft = torch.sigmoid((cast_logit + logistic_noise) / self.cast_temperature)
        cast_hard = (cast_soft > 0.5).float()
        cast_action = cast_hard + cast_soft - cast_soft.detach()

        bern = Bernoulli(logits=cast_logit)
        disc_log_prob = bern.log_prob(cast_hard)

        action = torch.cat([cont_action, cast_action], dim=-1)
        log_prob = (cont_log_prob + disc_log_prob).squeeze(-1)
        cast_prob = torch.sigmoid(cast_logit)
        return action, log_prob, h2, mu, cast_prob

    def deterministic(self, obs, action_low, action_high, h=None):
        mu, _, cast_logit, h2 = self.forward(obs, h)
        y = torch.tanh(mu)

        cont_low = action_low[:2]
        cont_high = action_high[:2]
        cont_scale = (cont_high - cont_low) * 0.5
        cont_bias = (cont_high + cont_low) * 0.5
        cont_action = y * cont_scale + cont_bias

        cast_action = (cast_logit > 0.0).float()
        action = torch.cat([cont_action, cast_action], dim=-1)
        return action, h2


class ConnectomeHybridActor(nn.Module):
    """
    Hybrid policy with connectome recurrent backbone for [v, omega, cast].
    The policy head API is identical to RecurrentHybridActor.
    """
    def __init__(
        self,
        obs_dim,
        cont_act_dim,
        hidden=256,
        log_std_min=-5.0,
        log_std_max=2.0,
        cast_temperature=0.5,
        connectome_steps=4,
    ):
        super().__init__()
        self.backbone = _ConnectomeBackbone(
            obs_dim=obs_dim,
            hidden_size=hidden,
            inner_steps=connectome_steps,
        )
        self.mu = nn.Linear(hidden, cont_act_dim)
        self.log_std = nn.Linear(hidden, cont_act_dim)
        self.cast_logit = nn.Linear(hidden, 1)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.cast_temperature = float(max(cast_temperature, 1e-3))

    def forward(self, obs, h=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        y, h2 = self.backbone(obs, h)
        mu = self.mu(y)
        log_std = torch.clamp(self.log_std(y), self.log_std_min, self.log_std_max)
        cast_logit = self.cast_logit(y)
        return mu, log_std, cast_logit, h2

    def sample(self, obs, action_low, action_high, h=None):
        mu, log_std, cast_logit, h2 = self.forward(obs, h)
        std = log_std.exp()
        normal = Normal(mu, std)
        x = normal.rsample()
        y = torch.tanh(x)

        cont_low = action_low[:2]
        cont_high = action_high[:2]
        cont_scale = (cont_high - cont_low) * 0.5
        cont_bias = (cont_high + cont_low) * 0.5
        cont_action = y * cont_scale + cont_bias

        cont_log_prob = normal.log_prob(x) - torch.log(cont_scale * (1.0 - y.pow(2)) + 1e-6)
        cont_log_prob = cont_log_prob.sum(dim=-1, keepdim=True)

        u = torch.rand_like(cast_logit)
        u = torch.clamp(u, 1e-6, 1.0 - 1e-6)
        logistic_noise = torch.log(u) - torch.log1p(-u)
        cast_soft = torch.sigmoid((cast_logit + logistic_noise) / self.cast_temperature)
        cast_hard = (cast_soft > 0.5).float()
        cast_action = cast_hard + cast_soft - cast_soft.detach()

        bern = Bernoulli(logits=cast_logit)
        disc_log_prob = bern.log_prob(cast_hard)

        action = torch.cat([cont_action, cast_action], dim=-1)
        log_prob = (cont_log_prob + disc_log_prob).squeeze(-1)
        cast_prob = torch.sigmoid(cast_logit)
        return action, log_prob, h2, mu, cast_prob

    def deterministic(self, obs, action_low, action_high, h=None):
        mu, _, cast_logit, h2 = self.forward(obs, h)
        y = torch.tanh(mu)

        cont_low = action_low[:2]
        cont_high = action_high[:2]
        cont_scale = (cont_high - cont_low) * 0.5
        cont_bias = (cont_high + cont_low) * 0.5
        cont_action = y * cont_scale + cont_bias

        cast_action = (cast_logit > 0.0).float()
        action = torch.cat([cont_action, cast_action], dim=-1)
        return action, h2


class Connectome2HybridActor(nn.Module):
    """
    Hybrid policy with Connectome2 recurrent backbone for [v, omega, cast].
    The policy head API is identical to RecurrentHybridActor.
    """
    def __init__(
        self,
        obs_dim,
        cont_act_dim,
        hidden=180,
        log_std_min=-5.0,
        log_std_max=2.0,
        cast_temperature=0.5,
        connectome_steps=4,
    ):
        super().__init__()
        self.backbone = _Connectome2Backbone(
            obs_dim=obs_dim,
            hidden_size=hidden,
            inner_steps=connectome_steps,
        )
        self.mu = nn.Linear(hidden, cont_act_dim)
        self.log_std = nn.Linear(hidden, cont_act_dim)
        self.cast_logit = nn.Linear(hidden, 1)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.cast_temperature = float(max(cast_temperature, 1e-3))

    def forward(self, obs, h=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        y, h2 = self.backbone(obs, h)
        mu = self.mu(y)
        log_std = torch.clamp(self.log_std(y), self.log_std_min, self.log_std_max)
        cast_logit = self.cast_logit(y)
        return mu, log_std, cast_logit, h2

    def sample(self, obs, action_low, action_high, h=None):
        mu, log_std, cast_logit, h2 = self.forward(obs, h)
        std = log_std.exp()
        normal = Normal(mu, std)
        x = normal.rsample()
        y = torch.tanh(x)

        cont_low = action_low[:2]
        cont_high = action_high[:2]
        cont_scale = (cont_high - cont_low) * 0.5
        cont_bias = (cont_high + cont_low) * 0.5
        cont_action = y * cont_scale + cont_bias

        cont_log_prob = normal.log_prob(x) - torch.log(cont_scale * (1.0 - y.pow(2)) + 1e-6)
        cont_log_prob = cont_log_prob.sum(dim=-1, keepdim=True)

        u = torch.rand_like(cast_logit)
        u = torch.clamp(u, 1e-6, 1.0 - 1e-6)
        logistic_noise = torch.log(u) - torch.log1p(-u)
        cast_soft = torch.sigmoid((cast_logit + logistic_noise) / self.cast_temperature)
        cast_hard = (cast_soft > 0.5).float()
        cast_action = cast_hard + cast_soft - cast_soft.detach()

        bern = Bernoulli(logits=cast_logit)
        disc_log_prob = bern.log_prob(cast_hard)

        action = torch.cat([cont_action, cast_action], dim=-1)
        log_prob = (cont_log_prob + disc_log_prob).squeeze(-1)
        cast_prob = torch.sigmoid(cast_logit)
        return action, log_prob, h2, mu, cast_prob

    def deterministic(self, obs, action_low, action_high, h=None):
        mu, _, cast_logit, h2 = self.forward(obs, h)
        y = torch.tanh(mu)

        cont_low = action_low[:2]
        cont_high = action_high[:2]
        cont_scale = (cont_high - cont_low) * 0.5
        cont_bias = (cont_high + cont_low) * 0.5
        cont_action = y * cont_scale + cont_bias

        cast_action = (cast_logit > 0.0).float()
        action = torch.cat([cont_action, cast_action], dim=-1)
        return action, h2
