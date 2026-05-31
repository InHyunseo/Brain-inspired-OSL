"""Pseudo-spiking connectome policy, built directly from the two larva CSVs.

This is the *brain-style* policy for evolution-strategy (ES) training — no
backprop, no critic. Every layer is biologically motivated:

  - **structure**: the real 389-node larva connectivity matrix (`weights.csv`)
    with cell-type metadata (`metadata.csv`) designating ORN sensory-input nodes
    and MBON output nodes. Only real synapses are parameters (sparse).
  - **neuron**: a leaky integrate-and-fire (LIF) unit. Each node holds a membrane
    potential `v`; per env step it leaks, integrates incoming spikes weighted by
    synapse strength, and emits a *soft spike* `s = sigmoid((v - thr)/temp)`
    (a smooth surrogate for the 0/1 spike), then partially resets `v`. Spikes —
    not raw activations — propagate along edges, exactly like a real circuit.
  - **learning**: none here. ES perturbs the flat parameter vector and selects on
    fitness (env return). Spikes need not be differentiable.

Forward is pure numpy (fast, gradient-free). Parameters are exposed as one flat
vector via `get_params` / `set_params` so an ES loop can treat the whole brain
as a black box.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.models.connectome import load_connectome_layout


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class SpikingConnectome:
    """LIF message-passing policy over the larva connectome (numpy, ES-ready).

    Action is 3-D continuous in [-1, 1] (forward, body-omega, head-omega), read
    out from the MBON output nodes' spike rates through a small linear head.
    """

    def __init__(
        self,
        weights_csv: str | Path = "assets/connectome/weights.csv",
        metadata_csv: str | Path = "assets/connectome/metadata.csv",
        obs_dim: int = 6,
        action_dim: int = 3,
        n_output_nodes: int = 8,
        steps_per_env_step: int = 4,
        leak: float = 0.9,
        v_threshold: float = 0.5,
        spike_temp: float = 0.25,
        reset_strength: float = 1.0,
        input_gain: float = 3.0,
        seed: int = 0,
    ):
        connectivity, layout = load_connectome_layout(weights_csv, metadata_csv, n_output_nodes)
        self.layout = layout
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.N = layout.total_node_count
        self.steps = int(steps_per_env_step)
        self.leak = float(leak)
        self.v_threshold = float(v_threshold)
        self.spike_temp = float(spike_temp)
        self.reset_strength = float(reset_strength)
        self.input_gain = float(input_gain)

        # Build the directed edge list (synapses) once. Same wiring rules as the
        # torch Connectome: base graph + sensor->ORN + MBON->output.
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
        self.src, self.tgt = np.nonzero(mask)
        self.n_edges = int(self.src.size)
        self.output_idx = np.asarray(layout.output_indices, dtype=np.int64)
        self.left_sensor = layout.left_sensor_index
        self.right_sensor = layout.right_sensor_index
        self.left_orn = np.asarray(layout.left_orn_indices, dtype=np.int64)
        self.right_orn = np.asarray(layout.right_orn_indices, dtype=np.int64)

        # --- parameters (the ES search space) ---
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(max(1, self.n_edges / n))   # fan-in-aware init
        self.w_edge = (rng.standard_normal(self.n_edges).astype(np.float32) * scale)
        self.b_node = np.zeros(n, dtype=np.float32)
        # input projection: obs -> drive on ORN sensory nodes (beyond raw sensors)
        self.w_in = (rng.standard_normal((self.obs_dim, self.N)).astype(np.float32) * 0.1)
        # readout: output-node spike rates -> action (tanh-bounded)
        self.w_out = (rng.standard_normal((self.output_idx.size, self.action_dim)).astype(np.float32) * 0.3)
        self.b_out = np.zeros(self.action_dim, dtype=np.float32)

        self._shapes = [
            ("w_edge", self.w_edge.shape), ("b_node", self.b_node.shape),
            ("w_in", self.w_in.shape), ("w_out", self.w_out.shape), ("b_out", self.b_out.shape),
        ]
        self.n_params = sum(int(np.prod(s)) for _, s in self._shapes)
        self.reset_state()

    # ---- ES parameter interface (flat vector) ----
    def get_params(self) -> np.ndarray:
        return np.concatenate([self.w_edge.ravel(), self.b_node.ravel(),
                               self.w_in.ravel(), self.w_out.ravel(), self.b_out.ravel()])

    def set_params(self, flat: np.ndarray) -> None:
        i = 0
        for name, shape in self._shapes:
            k = int(np.prod(shape))
            setattr(self, name, flat[i:i + k].reshape(shape).astype(np.float32))
            i += k

    # ---- episode state ----
    def reset_state(self) -> None:
        self.v = np.zeros(self.N, dtype=np.float32)        # membrane potential
        self.s = np.zeros(self.N, dtype=np.float32)        # last spikes

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        # External drive: raw sensors onto their sensor nodes + a learned obs->node
        # projection (lets the network use dlog/efference, still injected as drive).
        drive = obs @ self.w_in                            # (N,)
        drive[self.left_sensor] += obs[0]
        drive[self.right_sensor] += obs[1]

        for _ in range(self.steps):
            # synaptic current: spikes from sources, weighted, scattered to targets
            msg = self.s[self.src] * self.w_edge
            inp = np.zeros(self.N, dtype=np.float32)
            np.add.at(inp, self.tgt, msg)
            inp += drive + self.b_node
            # LIF: leak + integrate
            self.v = self.leak * self.v + inp
            # soft spike (surrogate) + reset
            self.s = _sigmoid((self.v - self.v_threshold) / self.spike_temp)
            self.v = self.v - self.reset_strength * self.s * self.v_threshold

        rate = self.s[self.output_idx]                     # output spike rates
        action = np.tanh(rate @ self.w_out + self.b_out)
        return action.astype(np.float32)

    def describe(self) -> dict:
        return {
            "nodes": self.N, "edges": self.n_edges, "params": self.n_params,
            "steps_per_env_step": self.steps, "output_nodes": int(self.output_idx.size),
        }
