"""Config elements for Linear Pytorch Lightning module usage"""

from typing import Optional
from pydantic import BaseModel, field_validator

from prescyent.predictor.lightning.configs.module_config import ModuleConfig
from prescyent.utils.enums import Normalizations


class Config(ModuleConfig):
    """Pydantic Basemodel for MLP Module configuration"""

    num_layers: int = 48
    dct: bool = True
    spatial_fc_only: bool = False
    temporal_fc_in: bool = False
    temporal_fc_out: bool = False
    simple_blocks_norm: Optional[Normalizations] = Normalizations.SPATIAL

    @field_validator("num_layers")
    @classmethod
    def name_sup_or_equal_one(cls, v):
        if v < 1:
            raise ValueError("num_layers must be >= 1")
        return v
