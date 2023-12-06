from enum import Enum


class ActivationFunctions(str, Enum):
    RELU = "relu"
    GELU = "gelu"
    SIGMOID = "sigmoid"
