"""2D-OSL connectome/GRU policy analysis pipeline.

Ports the precise multi-phase analysis from the 3D `osl_analysis` ROS2 package
to the 2D Colab notebooks (SAC/PPO × connectome/GRU). All analysis logic is
simulation-dimension agnostic: it consumes per-episode trace ``.npz`` files
produced by :mod:`Analysis.osl2d.eval_dump`.

Phases:
    phase1_label       — behavior label distribution + transition matrix + cast PSD
    phase2a_latent_viz — PCA→UMAP hidden-state separation (silhouette / CH)
    phase2b_probe      — episode-split linear probe + confusion + timeline
    phase2c_neuron     — per-neuron contributions → neuron_groups.json
    phase3a_jacobian   — Jacobian eigenmode (slow / oscillatory) analysis
    phase3b_fixedpoint — fixed-point search + stability classification
    phase4_ablation    — neuron-group ablation causal test (live env)
"""
