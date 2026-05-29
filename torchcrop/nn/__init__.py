"""ML components for hybrid crop modeling."""

from torchcrop.nn.hybrid import (
    HybridManager,
    ResidualHead,
    ResidualSpec,
    default_slots,
)
from torchcrop.nn.learned_stress import LearnedStressFactor
from torchcrop.nn.param_net import ParameterNet
from torchcrop.nn.residual import NeuralResidual

__all__ = [
    "HybridManager",
    "LearnedStressFactor",
    "NeuralResidual",
    "ParameterNet",
    "ResidualHead",
    "ResidualSpec",
    "default_slots",
]
