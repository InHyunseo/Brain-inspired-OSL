"""sb3-contrib RecurrentActorCriticPolicy variant with CfC + AutoNCP.

Strategy: rather than overriding all of sb3's recurrent flow, we keep the
parent's `_process_sequence` static method *unchanged* and substitute the
LSTM module with an `nn.LSTM`-compatible wrapper backed by CfC. This is the
minimum-surface change that preserves sb3's correct handling of:
  - episode_starts masking inside the BPTT replay
  - value bootstrap on truncation
  - actor/value head structure

After training, weights are extracted into a standalone `NCPCore` instance
which `eval_dump.py` and the phase scripts consume — the analysis code does
NOT depend on sb3.
"""
from __future__ import annotations

import torch as th
from torch import nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP

from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy


class CfCAsLSTM(nn.Module):
    """Drop-in replacement for `nn.LSTM` backed by CfC + AutoNCP wiring.

    Matches the call signature sb3's `_process_sequence` expects:
        forward(x: (seq_len, batch, input_size), (h, c)) -> (y, (h, c))
    where:
        h, c shape: (num_layers=1, batch, units)    (we ignore c — keep zeros)
        y shape:    (seq_len, batch, output_size)

    Note: `self.hidden_size` must advertise the recurrent state width because
    RecurrentPPO uses it to allocate rollout states. The latent output width
    consumed by the policy/value heads remains `output_size`.
    """

    def __init__(self, input_size: int, units: int, output_size: int,
                 sparsity_level: float = 0.5, wiring_seed: int = 12345):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = units
        self.num_layers = 1
        self.units = units
        self.output_size = output_size

        self.wiring = AutoNCP(units=units, output_size=output_size,
                              sparsity_level=sparsity_level, seed=wiring_seed)
        self.cfc = CfC(input_size=input_size, units=self.wiring, batch_first=False)

    def forward(self, x: th.Tensor, state):
        # x:    (seq, batch, input_size)
        # state: (h, c), each (1, batch, units). c is unused (kept zero).
        h_in = state[0].squeeze(0)             # (batch, units)
        y, h_out = self.cfc(x, h_in)           # y: (seq, batch, output_size)
        new_state = (h_out.unsqueeze(0), state[1])
        return y, new_state


class RecurrentCfCPolicy(RecurrentActorCriticPolicy):
    """sb3 recurrent actor-critic with CfC backbone.

    Extra `policy_kwargs`:
        units (int): total AutoNCP units (motor + command + inter)
        output_size (int): motor neuron count (== latent dim seen by heads)
        sparsity_level (float)
        wiring_seed (int)
    """

    def __init__(self, observation_space, action_space, lr_schedule,
                 *args,
                 units: int = 32, output_size: int = 8,
                 sparsity_level: float = 0.5, wiring_seed: int = 12345,
                 **kwargs):
        # Stash CfC params before super; super uses lstm_hidden_size for
        # downstream head dims, so we set it to output_size.
        self._cfc_units = units
        self._cfc_output_size = output_size
        self._cfc_sparsity = sparsity_level
        self._cfc_seed = wiring_seed

        # Force sb3 settings compatible with shared-CfC design.
        kwargs.setdefault("shared_lstm", True)
        kwargs.setdefault("enable_critic_lstm", False)
        kwargs.setdefault("ortho_init", False)
        # Keep SB3 heads directly on the CfC motor outputs so they can be
        # exported into NCPCore without an extra MLP block.
        kwargs.setdefault("net_arch", [])
        kwargs["lstm_hidden_size"] = output_size      # mlp_extractor / head input dim
        kwargs["n_lstm_layers"] = 1

        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

        # Replace LSTM with CfC wrapper. Parent created `self.lstm_actor` as
        # nn.LSTM; we delete and substitute. `shared_lstm=True` means no
        # separate `lstm_critic`. The state buffer shape (used to allocate
        # initial hidden states) must match the CfC unit count, not output_size.
        del self.lstm_actor
        self.lstm_actor = CfCAsLSTM(
            input_size=self.features_dim,
            units=units,
            output_size=output_size,
            sparsity_level=sparsity_level,
            wiring_seed=wiring_seed,
        )
        # State shape: (n_layers=1, batch=1, units). RecurrentPPO uses this to
        # allocate (h, c) tuples in the rollout buffer.
        self.lstm_hidden_state_shape = (1, 1, units)

        # Rebuild optimizer to capture new CfC params (parent's init optimizer
        # was bound to the deleted LSTM parameters).
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    # -------- Analysis surface (used after training, by NCPCore extraction) --

    @property
    def wiring(self) -> AutoNCP:
        return self.lstm_actor.wiring

    def export_ncp_core_state(self) -> dict:
        """Build a state_dict compatible with `NCPCore` for analysis.

        NCPCore has:
            backbone (CfC)
            actor_head (Linear: output_size -> action_dim)
            critic_head (Linear: output_size -> 1)
            log_std (Parameter: action_dim)

        We copy from:
            self.lstm_actor.cfc         -> backbone
            self.action_net (Linear)    -> actor_head
            self.value_net (Linear)     -> critic_head
            self.log_std (Parameter)    -> log_std
        """
        sd = {}
        # backbone weights — prefix all CfC params with "backbone."
        for k, v in self.lstm_actor.cfc.state_dict().items():
            sd[f"backbone.{k}"] = v.detach().clone()
        # heads
        for k, v in self.action_net.state_dict().items():
            sd[f"actor_head.{k}"] = v.detach().clone()
        for k, v in self.value_net.state_dict().items():
            sd[f"critic_head.{k}"] = v.detach().clone()
        # log_std (Gaussian distribution param). Stored on the distribution.
        if hasattr(self, "log_std"):
            sd["log_std"] = self.log_std.detach().clone()
        return sd

    def export_ncp_core_config(self, obs_dim: int, action_dim: int) -> dict:
        return dict(
            obs_dim=obs_dim,
            action_dim=action_dim,
            units=self._cfc_units,
            output_size=self._cfc_output_size,
            sparsity_level=self._cfc_sparsity,
            wiring_seed=self._cfc_seed,
        )
