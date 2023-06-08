"""Config elements for Linear Pytorch Lightning module usage"""
from typing import List
from pydantic import BaseModel

from prescyent.predictor.lightning.configs.module_config import ModuleConfig

DEFAULT_HIDDEN = 64


class MotionFCInConfig(BaseModel):
    # in_features =    hidden_size  # hidden_dim
    # out_features = feature_size
    with_norm = False
    activation = "relu"
    init_w_trunc_normal = False


class MotionFCOutConfig(BaseModel):
    # in_features = feature_size
    # out_features =  hidden_size  # hidden_dim
    with_norm = False
    activation = "relu"
    init_w_trunc_normal = True


class Config(ModuleConfig):
    """Pydantic Basemodel for MLP Module configuration"""

    input_size: int
    output_size: int
    feature_size: int
    hidden_size: int = DEFAULT_HIDDEN
    num_layers: int = 48
    dct: bool = True
    temporal_fc_in: bool = False
    temporal_fc_out: bool = False
    with_normalization: bool = True
    spatial_fc_only: bool = False
    norm_axis: str = "spatial"
    motion_fc_in: MotionFCInConfig = MotionFCInConfig()
    motion_fc_out: MotionFCOutConfig = MotionFCOutConfig()
