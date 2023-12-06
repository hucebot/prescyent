"""Config elements for Seq2Seq Pytorch Lightning module usage"""
from pydantic import field_validator

from prescyent.predictor.lightning.configs.module_config import ModuleConfig


class Config(ModuleConfig):
    """Pydantic Basemodel for Seq2Seq Module configuration"""

    hidden_size: int = 128
    num_layers: int = 2

    @field_validator("num_layers")
    @classmethod
    def name_sup_or_equal_one(cls, v):
        if v < 1:
            raise ValueError("num_layers must be >= 1")
        return v
