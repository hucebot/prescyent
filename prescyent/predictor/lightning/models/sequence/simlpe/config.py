"""Config elements for Linear Pytorch Lightning module usage"""
from pydantic import BaseModel, field_validator

from prescyent.predictor.lightning.configs.module_config import ModuleConfig


DEFAULT_HIDDEN = 64


class Config(ModuleConfig):
    """Pydantic Basemodel for MLP Module configuration"""

    input_size: int
    output_size: int
    num_points: int
    num_dims: int
    hidden_size: int = DEFAULT_HIDDEN
    num_layers: int = 48
    dct: bool = True
    temporal_fc_in: bool = False
    temporal_fc_out: bool = False
    spatial_fc_only: bool = False

    @field_validator("num_layers")
    @classmethod
    def name_sup_or_equal_one(cls, v):
        if v < 1:
            raise ValueError("num_layers must be >= 1")
        return v
