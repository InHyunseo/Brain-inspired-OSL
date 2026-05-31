"""Larva connectome message-passing branch.

Loads a real 387-node connectivity matrix + cell-type metadata, augments it
with two sensor input nodes (left/right ORN fan-out) and `latent_dim` output
nodes (MBON fan-in), then runs `message_passing_steps` synchronous tanh
updates per environment step. Edge weights are learnable scalars.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def _apply_patch(state: torch.Tensor, patch: dict[str, Any] | None) -> torch.Tensor:
    """Overwrite hidden-state units at `patch['indices']` for ablation analysis.

    `patch = {"indices": <list/array/tensor>, "value": "zero"|"mean"|"flip"|float}`.
    Returns a new tensor; the input is left unmodified (autograd-safe). Mirrors
    osl_analysis `policy_adapter._apply_patch`.
    """
    if patch is None:
        return state
    indices = patch.get("indices")
    if indices is None:
        return state
    idx = torch.as_tensor(indices, dtype=torch.long, device=state.device)
    if idx.numel() == 0:
        return state
    out = state.clone()
    value = patch.get("value", "zero")
    if value == "zero":
        out[:, idx] = 0.0
    elif value == "mean":
        out[:, idx] = state[:, idx].mean()
    elif value == "flip":
        out[:, idx] = -state[:, idx]
    else:
        out[:, idx] = float(value)
    return out


def _activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "relu":
        return nn.ReLU()
    if key == "silu":
        return nn.SiLU()
    if key == "gelu":
        return nn.GELU()
    if key == "elu":
        return nn.ELU()
    return nn.Tanh()


@dataclass
class ConnectomeLayout:
    base_node_count: int
    left_sensor_index: int
    right_sensor_index: int
    output_indices: tuple[int, ...]
    left_orn_indices: list[int]
    right_orn_indices: list[int]
    mbon_indices: list[int]
    celltype_indices: dict[str, list[int]] | None = None

    @property
    def total_node_count(self) -> int:
        return self.base_node_count + 2 + len(self.output_indices)

    def analysis_group_indices(self) -> dict[str, list[int]]:
        """Cell-type partition of the augmented node activation `(total_node_count,)`.

        Keys mirror osl_analysis `neuron_groups.py` connectome convention so the
        analysis pipeline can group hidden units by biological cell type. Returns
        node *indices into the full augmented state vector* (base + 2 sensors +
        latent output nodes), not just base nodes.
        """
        groups: dict[str, list[int]] = {"all": list(range(self.total_node_count))}
        groups["left_orn"] = list(self.left_orn_indices)
        groups["right_orn"] = list(self.right_orn_indices)
        groups["orn"] = sorted(set(self.left_orn_indices) | set(self.right_orn_indices))
        groups["mbon"] = list(self.mbon_indices)
        groups["sensor_input"] = [self.left_sensor_index, self.right_sensor_index]
        groups["latent_output"] = list(self.output_indices)
        if self.celltype_indices:
            for name, idx in self.celltype_indices.items():
                groups[f"celltype_{name}"] = list(idx)
        return groups


def load_connectome_layout(
    weights_csv: str | Path,
    metadata_csv: str | Path,
    latent_dim: int,
) -> tuple[np.ndarray, ConnectomeLayout]:
    weights = np.loadtxt(weights_csv, delimiter=",")
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError(f"Expected square weights matrix in {weights_csv}, got shape {weights.shape}")
    if latent_dim <= 0:
        raise ValueError(f"latent_dim must be positive, got {latent_dim}")

    with Path(metadata_csv).open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != weights.shape[0]:
        raise ValueError(
            f"Metadata row count {len(rows)} does not match weights size {weights.shape[0]}"
        )

    truthy = {"1", "true", "True"}
    left_orn = [i for i, r in enumerate(rows) if r.get("is_left_orn", "0") in truthy]
    right_orn = [i for i, r in enumerate(rows) if r.get("is_right_orn", "0") in truthy]
    mbon = [i for i, r in enumerate(rows) if r.get("is_mbon", "0") in truthy]

    celltype_indices: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        ct = (r.get("celltype") or "").strip()
        if ct:
            celltype_indices.setdefault(ct, []).append(i)

    base = weights.shape[0]
    output_indices = tuple(base + 2 + i for i in range(latent_dim))
    layout = ConnectomeLayout(
        base_node_count=base,
        left_sensor_index=base,
        right_sensor_index=base + 1,
        output_indices=output_indices,
        left_orn_indices=left_orn,
        right_orn_indices=right_orn,
        mbon_indices=mbon,
        celltype_indices=celltype_indices or None,
    )
    return (weights > 0).astype(np.float32), layout


class Connectome(nn.Module):
    """Message-passing branch over the larva connectome.

    Input: sensor reads `(B, 2)` = `[c_left, c_right]`.
    Output: latent activation `(B, latent_dim)` from output nodes (MBON fan-in).

    State carries the full augmented node activation `(B, total_node_count)`
    across env steps; reset to zeros via `initial_state`. `mask` is `(B, 1)`
    with 1.0 to keep state, 0.0 to zero it (used at episode boundaries).
    """

    def __init__(
        self,
        weights_csv: str | Path,
        metadata_csv: str | Path,
        latent_dim: int = 32,
        message_passing_steps: int = 6,
        activation: str = "tanh",
        feature_dim: int = 8,
    ):
        super().__init__()
        connectivity, layout = load_connectome_layout(weights_csv, metadata_csv, latent_dim)
        self.layout = layout
        # Each node now carries a D-dim feature vector instead of a scalar, so a
        # single edge can route a richer message and the 6-hop recurrence has the
        # capacity to learn a real policy. `latent_dim` here names the number of
        # output *nodes* (kept for layout/back-compat); the actual latent fed to
        # the policy head is those nodes' D-dim states flattened -> see `out_dim`.
        self.feature_dim = int(feature_dim)
        self.num_output_nodes = int(latent_dim)
        self.message_passing_steps = int(message_passing_steps)
        self.state_activation = _activation(activation)
        n_total = layout.total_node_count
        self.n_total = n_total
        # state is stored flattened as (B, N*D); state_size keeps that convention
        # so runners/buffers (which mask with (B,1)) treat it as one hidden vector.
        self.state_size = n_total * self.feature_dim
        self.latent_dim = len(layout.output_indices) * self.feature_dim

        mask = np.zeros((n_total, n_total), dtype=np.float32)
        mask[: layout.base_node_count, : layout.base_node_count] = connectivity
        if layout.left_orn_indices:
            mask[layout.left_sensor_index, layout.left_orn_indices] = 1.0
        if layout.right_orn_indices:
            mask[layout.right_sensor_index, layout.right_orn_indices] = 1.0
        if layout.mbon_indices:
            for output_index in layout.output_indices:
                mask[layout.mbon_indices, output_index] = 1.0

        edge_sources, edge_targets = np.nonzero(mask)
        self.register_buffer("edge_sources", torch.as_tensor(edge_sources, dtype=torch.long))
        self.register_buffer("edge_targets", torch.as_tensor(edge_targets, dtype=torch.long))
        # Per-edge D-dim gate: a single synapse modulates each feature channel.
        self.edge_weight = nn.Parameter(torch.empty(len(edge_sources), self.feature_dim))
        # Per-node channel mixer so the D channels actually interact (depthwise
        # aggregation alone would keep channels independent). Shared across nodes.
        self.channel_mix = nn.Linear(self.feature_dim, self.feature_dim)
        self.norm = nn.LayerNorm(self.feature_dim)
        self.bias = nn.Parameter(torch.zeros(n_total, self.feature_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.edge_weight.numel() > 0:
            nn.init.xavier_uniform_(self.edge_weight)
        with torch.no_grad():
            self.bias.zero_()

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.state_size, device=device)

    def _inject_sensors(self, state: torch.Tensor, sensor_obs: torch.Tensor) -> torch.Tensor:
        # state: (B, N, D). Drive sensor nodes' channel 0 with the raw reading;
        # other channels are left for the network to populate via message passing.
        next_state = state.clone()
        next_state[:, self.layout.left_sensor_index, 0] = sensor_obs[:, 0]
        next_state[:, self.layout.right_sensor_index, 0] = sensor_obs[:, 1]
        return next_state

    def _aggregate(self, state: torch.Tensor) -> torch.Tensor:
        # state: (B, N, D) -> messages along edges, depthwise-gated, scattered to
        # targets. Returns aggregated (B, N, D).
        B, N, D = state.shape
        agg = torch.zeros(B, N, D, dtype=state.dtype, device=state.device)
        if self.edge_weight.numel() == 0:
            return agg
        messages = state[:, self.edge_sources] * self.edge_weight.unsqueeze(0)  # (B, E, D)
        agg.index_add_(1, self.edge_targets, messages)
        return agg

    @property
    def group_indices(self) -> dict[str, list[int]]:
        """Cell-type partition of the *flattened* (N*D) hidden state for analysis.

        The layout groups are node indices; with D-dim node features each node i
        occupies flat slots [i*D, i*D+D). We expand every group accordingly so
        ablation patches and per-group readouts address the right slots.
        """
        D = self.feature_dim
        node_groups = self.layout.analysis_group_indices()
        return {
            name: [i * D + c for i in nodes for c in range(D)]
            for name, nodes in node_groups.items()
        }

    def forward_step(
        self,
        sensor_obs: torch.Tensor,
        state: torch.Tensor,
        mask: torch.Tensor,
        patch: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # state arrives flattened (B, N*D); work in (B, N, D) then flatten back.
        B = state.shape[0]
        current = (state * mask).view(B, self.n_total, self.feature_dim)
        for _ in range(self.message_passing_steps):
            current = self._inject_sensors(current, sensor_obs)
            pre = self._aggregate(current) + self.bias              # (B, N, D)
            pre = self.channel_mix(pre)                             # mix channels
            update = self.state_activation(pre)
            # Residual + post-norm: the residual keeps the 6-hop gradient alive,
            # while LayerNorm over the *summed* state caps its magnitude so it
            # cannot accumulate across the 6 inner steps and across env steps
            # (without this the state grew linearly each step and saturated tanh).
            current = self.norm(current + update)
            current = self._inject_sensors(current, sensor_obs)
        current_flat = _apply_patch(current.reshape(B, -1), patch)
        current = current_flat.view(B, self.n_total, self.feature_dim)
        latent = current[:, list(self.layout.output_indices)].reshape(B, -1)
        return latent, current_flat

    def forward_sequence(
        self,
        sensor_obs_seq: torch.Tensor,
        state0: torch.Tensor,
        mask_seq: torch.Tensor,
        patch: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sensor_obs_seq.ndim != 3:
            raise ValueError(f"Expected sensor_obs_seq shape [T, B, 2], got {tuple(sensor_obs_seq.shape)}")
        state = state0
        outs = []
        for step in range(sensor_obs_seq.shape[0]):
            latent, state = self.forward_step(sensor_obs_seq[step], state, mask_seq[step], patch=patch)
            outs.append(latent)
        return torch.stack(outs, dim=0), state

    def describe(self) -> dict[str, Any]:
        return {
            "base_node_count": self.layout.base_node_count,
            "total_node_count": self.layout.total_node_count,
            "left_orn_count": len(self.layout.left_orn_indices),
            "right_orn_count": len(self.layout.right_orn_indices),
            "mbon_count": len(self.layout.mbon_indices),
            "feature_dim": self.feature_dim,
            "latent_dim": self.latent_dim,
            "active_edge_count": int(self.edge_sources.numel()),
            "message_passing_steps": self.message_passing_steps,
        }
