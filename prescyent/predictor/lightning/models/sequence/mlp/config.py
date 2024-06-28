"""Config elements for MLP Pytorch Lightning module usage"""
from typing import Optional
from pydantic import Field

from prescyent.predictor.lightning.configs.module_config import ModuleConfig
from prescyent.utils.enums import ActivationFunctions


class Config(ModuleConfig):
    """Pydantic Basemodel for MLP Module configuration"""

    hidden_size: int = 64
    """Size of the hidden FC Layers in the MLP"""
    num_layers: int = Field(2, gt=0)
    """Number of FC layers in the MLP"""
    activation: Optional[ActivationFunctions] = ActivationFunctions.RELU
    """Activation function used between layers"""
