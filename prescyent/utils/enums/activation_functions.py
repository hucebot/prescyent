from enum import Enum


class ActivationFunctions(str, Enum):
    """Map to a given activation function"""

    RELU = "relu"
    "torch.nn.ReLU"
    GELU = "gelu"
    "torch.nn.GELU"
    SIGMOID = "sigmoid"
    "torch.nn.Sigmoid"
