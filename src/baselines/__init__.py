"""Non-learning, purely computational baselines for 2D OSL.

These controllers use no neural network and no training — they map the same
bilateral-sensor observation the RL policies see to an action via fixed
chemotaxis rules. They serve as a fair sensor-only comparison group and an
upper-reference for "does the task get solved at all" in clean fields.
"""
from src.baselines.chemotaxis import (
    BilateralChemotaxis,
    ChemotaxisConfig,
    run_episode,
)

__all__ = ["BilateralChemotaxis", "ChemotaxisConfig", "run_episode"]
