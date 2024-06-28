from enum import Enum


class Normalizations(str, Enum):
    """Map to a given normalization layer and method"""

    BATCH = "batch_normalization"
    """torch.nn.BatchNorm2d"""
    ALL = "all"
    """torch.nn.LayerNorm on all dims"""
    SPATIAL = "spatial_normalization"
    """torch.nn.LayerNorm on spatial dims"""
    TEMPORAL = "temporal_normalization"
    """torch.nn.LayerNorm on temporal dim"""
