"""Config elements for Linear Pytorch Lightning module usage"""
from typing import List
from prescyent.predictor.lightning.module_config import ModuleConfig

DEFAULT_HIDDEN = 64

class MotionMlpConfig(ModuleConfig):
    hidden_dim = DEFAULT_HIDDEN
    # seq_len =  motion.h36m_input_length_dct  # history_size
    num_layers = 48
    with_normalization = True
    spatial_fc_only = False
    norm_axis = 'spatial'

class MotionFCInConfig(ModuleConfig):
    # in_features =    hidden_size  # hidden_dim
    # out_features = feature_size
    with_norm = False
    activation = 'relu'
    init_w_trunc_normal = False

class MotionFCOutConfig(ModuleConfig):
    # in_features = feature_size
    # out_features =  hidden_size  # hidden_dim
    with_norm = False
    activation = 'relu'
    init_w_trunc_normal = True

class Config(ModuleConfig):
    """Pydantic Basemodel for MLP Module configuration"""
    input_size: int
    output_size: int
    input_size_dct: int
    feature_size: int
    hidden_size: int = DEFAULT_HIDDEN
    pre_dct = True
    post_dct = True
    temporal_fc_in = False
    temporal_fc_out = False
    motion_mlp: MotionMlpConfig = MotionMlpConfig()
    motion_fc_in: MotionFCInConfig = MotionFCInConfig()
    motion_fc_out: MotionFCOutConfig = MotionFCOutConfig()
