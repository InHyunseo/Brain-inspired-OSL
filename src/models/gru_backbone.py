"""GRU actor backbone — a non-connectome alternative for the OSL policy.

Mirrors the `Connectome` backbone interface so `Policy`/`SACPolicy` can swap
backbones with a single `backbone=` flag:

- `forward_step(actor_in, state, mask, patch=None) -> (latent, next_state)`
- `forward_sequence(seq, state0, mask_seq, patch=None) -> (latent_seq, state)`
- `initial_state(B, device)`, `state_size`, `latent_dim`, `group_indices`

Unlike the connectome (which consumes only the 2 sensor channels and re-injects
efference at the head), the GRU consumes the *full* observation, so its
`latent` is the GRU hidden state itself and the policy head should NOT re-concat
efference. `state_size == latent_dim == hidden`.

The hidden-state partition (`group_indices`) follows the osl_analysis GRU
convention: an even three-way split (`third_1/2/3`) plus `all`, since a plain
GRU has no biological cell-type structure to group by.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.models.connectome import _apply_patch


class GRUBackbone(nn.Module):
    """Single-layer GRU cell wrapped in the connectome backbone interface."""

    def __init__(self, input_size: int, hidden: int = 421):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden = int(hidden)
        self.cell = nn.GRUCell(input_size=self.input_size, hidden_size=self.hidden)
        self.state_size = self.hidden
        self.latent_dim = self.hidden

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.state_size, device=device)

    @property
    def group_indices(self) -> dict[str, list[int]]:
        h = self.hidden
        third = h // 3
        return {
            "all": list(range(h)),
            "third_1": list(range(0, third)),
            "third_2": list(range(third, 2 * third)),
            "third_3": list(range(2 * third, h)),
        }

    def forward_step(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        mask: torch.Tensor,
        patch: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Zero the carried state at episode boundaries (mask is (B, 1) of 0/1).
        h_prev = state * mask
        h_next = self.cell(obs, h_prev)
        h_next = _apply_patch(h_next, patch)
        return h_next, h_next

    def forward_sequence(
        self,
        obs_seq: torch.Tensor,
        state0: torch.Tensor,
        mask_seq: torch.Tensor,
        patch: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if obs_seq.ndim != 3:
            raise ValueError(f"Expected obs_seq shape [T, B, F], got {tuple(obs_seq.shape)}")
        state = state0
        outs = []
        for step in range(obs_seq.shape[0]):
            latent, state = self.forward_step(obs_seq[step], state, mask_seq[step], patch=patch)
            outs.append(latent)
        return torch.stack(outs, dim=0), state

    def describe(self) -> dict[str, Any]:
        return {
            "backbone": "gru",
            "input_size": self.input_size,
            "hidden": self.hidden,
            "state_size": self.state_size,
            "latent_dim": self.latent_dim,
        }
