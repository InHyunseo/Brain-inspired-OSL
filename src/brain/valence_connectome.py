"""Perception-only connectome: a valence estimator, trained by supervised regression.

Biologically, the larva olfactory circuit (ORN -> KC -> MBON) does NOT issue motor
commands — it computes *valence*: "is this getting better?". So instead of an
end-to-end steering policy, we use the real connectome as a **valence estimator**
and ask the cleanest question the circuit can actually answer:

    Given only the bilateral sensor readings over time, can the connectome
    reconstruct dlog — the temporal change of log-concentration ("am I moving
    up-gradient?") — which is the signal larvae actually use for chemotaxis?

dlog is something the circuit can in principle compute (a temporal derivative of
its own inputs); absolute distance or global bearing are not, so dlog is the
biologically honest regression target.

This is a torch module (differentiable soft-spike LIF) so it trains with backprop
on dense supervision — sidestepping the sparse-reward RL failure entirely. The
connectome wiring (389-node larva graph) is the same fixed sparse structure as
the RL version; only per-edge synapse strengths and a small readout are learned.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.models.connectome import load_connectome_layout


class ValenceConnectome(nn.Module):
    """Soft-spiking (LIF) connectome that regresses a scalar valence from a
    sequence of bilateral sensor readings.

    Input  : input sequence (B, T, in_dim). Default in_dim=5 =
             [c_left, c_right, v, body_omega, head_omega] — bilateral sensors plus
             the efference copy of the current action. dlog is a function of the
             sensor *gradient* × the *movement* (a sensor×action interaction), so
             the action channels are required to make the target predictable.
    Output : valence sequence (B, T, 1) — prediction of dlog at each step.

    c_left/c_right are injected directly into the ORN sensor nodes; the full input
    (incl. efference) is also projected onto all nodes through a learned w_in, so
    the network can form the sensor×movement interaction over its 6-hop recurrence.
    """

    def __init__(
        self,
        weights_csv: str | Path = "assets/connectome/weights.csv",
        metadata_csv: str | Path = "assets/connectome/metadata.csv",
        in_dim: int = 5,
        n_output_nodes: int = 8,
        inner_steps: int = 4,
        leak: float = 0.9,
        v_threshold: float = 0.5,
        spike_temp: float = 0.25,
        reset_strength: float = 1.0,
        input_gain: float = 3.0,
    ):
        super().__init__()
        connectivity, layout = load_connectome_layout(weights_csv, metadata_csv, n_output_nodes)
        self.layout = layout
        self.N = layout.total_node_count
        self.inner_steps = int(inner_steps)
        self.leak = float(leak)
        self.v_threshold = float(v_threshold)
        self.spike_temp = float(spike_temp)
        self.reset_strength = float(reset_strength)
        self.input_gain = float(input_gain)
        self.in_dim = int(in_dim)

        n = self.N
        mask = np.zeros((n, n), dtype=np.float32)
        mask[: layout.base_node_count, : layout.base_node_count] = connectivity
        if layout.left_orn_indices:
            mask[layout.left_sensor_index, layout.left_orn_indices] = 1.0
        if layout.right_orn_indices:
            mask[layout.right_sensor_index, layout.right_orn_indices] = 1.0
        if layout.mbon_indices:
            for oi in layout.output_indices:
                mask[layout.mbon_indices, oi] = 1.0
        src, tgt = np.nonzero(mask)
        self.register_buffer("src", torch.as_tensor(src, dtype=torch.long))
        self.register_buffer("tgt", torch.as_tensor(tgt, dtype=torch.long))
        self.register_buffer("output_idx", torch.as_tensor(layout.output_indices, dtype=torch.long))
        self.left_sensor = layout.left_sensor_index
        self.right_sensor = layout.right_sensor_index

        scale = 1.0 / np.sqrt(max(1.0, len(src) / n))
        self.w_edge = nn.Parameter(torch.randn(len(src)) * scale)   # synapse strengths
        self.b_node = nn.Parameter(torch.zeros(n))
        # learned projection of the full input (sensors + efference) onto all nodes,
        # so the circuit can mix sensor gradient with self-movement (the interaction
        # that actually predicts dlog).
        self.w_in = nn.Parameter(torch.randn(self.in_dim, n) * 0.1)
        self.readout = nn.Linear(int(self.output_idx.numel()), 1)   # MBON rates -> valence

    def _soft_spike(self, v):
        return torch.sigmoid((v - self.v_threshold) / self.spike_temp)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        # input_seq: (B, T, in_dim) = [c_left, c_right, v, body_w, head_w].
        # Returns valence (B, T, 1).
        B, T, _ = input_seq.shape
        dev = input_seq.device
        v = torch.zeros(B, self.N, device=dev)
        s = torch.zeros(B, self.N, device=dev)
        outs = []
        for t in range(T):
            x = input_seq[:, t]                                         # (B, in_dim)
            drive = x @ self.w_in                                       # (B, N) efference proj
            drive[:, self.left_sensor] = drive[:, self.left_sensor] + x[:, 0] * self.input_gain
            drive[:, self.right_sensor] = drive[:, self.right_sensor] + x[:, 1] * self.input_gain
            for _ in range(self.inner_steps):
                msg = s[:, self.src] * self.w_edge.unsqueeze(0)          # (B, E)
                inp = torch.zeros(B, self.N, device=dev)
                inp.index_add_(1, self.tgt, msg)
                inp = inp + drive + self.b_node.unsqueeze(0)
                v = self.leak * v + inp
                s = self._soft_spike(v)
                v = v - self.reset_strength * s * self.v_threshold
            rate = s[:, self.output_idx]                                # (B, n_out)
            outs.append(self.readout(rate))                            # (B, 1)
        return torch.stack(outs, dim=1)                                # (B, T, 1)

    def describe(self) -> dict:
        return {"nodes": self.N, "edges": int(self.src.numel()),
                "params": sum(p.numel() for p in self.parameters()),
                "inner_steps": self.inner_steps,
                "output_nodes": int(self.output_idx.numel())}
