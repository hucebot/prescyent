"""Config elements for MLP Pytorch Lightning module usage"""
from pydantic import field_validator

from prescyent.predictor.lightning.configs.module_config import ModuleConfig
from prescyent.utils.enums import ActivationFunctions


class Config(ModuleConfig):
    """Pydantic Basemodel for MLP Module configuration"""

    hidden_size: int = 64
    num_layers: int = 2
    activation: ActivationFunctions = ActivationFunctions.RELU

    @field_validator("num_layers")
    @classmethod
    def name_sup_or_equal_one(cls, v):
        if v < 1:
            raise ValueError("num_layers must be >= 1")
        return v
