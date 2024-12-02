"""Config elements for SARLSTM Lightning module usage"""
from pydantic import field_validator

from prescyent.predictor.lightning.configs.module_config import ModuleConfig


class SARLSTMConfig(ModuleConfig):
    """Pydantic Basemodel for SARLSTM configuration"""

    hidden_size: int = 128
    """Hidden size of the LSTMCells"""
    num_layers: int = 2
    """Number of LSTMCell"""

    @field_validator("num_layers")
    @classmethod
    def name_sup_or_equal_one(cls, v: int):
        if v < 1:
            raise ValueError("num_layers must be >= 1")
        return v
