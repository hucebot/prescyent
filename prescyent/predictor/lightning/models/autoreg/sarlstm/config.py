"""Config elements for SARLSTM Lightning module usage"""
from pydantic import validator

from prescyent.predictor.lightning.configs.module_config import ModuleConfig


class Config(ModuleConfig):
    """Pydantic Basemodel for SARLSTM configuration"""

    feature_size: int
    hidden_size: int = 10
    num_layers: int = 2

    @validator("num_layers")
    def name_sup_or_equal_one(cls, v):
        if v < 1:
            raise ValueError("num_layers must be >= 1")
        return v
