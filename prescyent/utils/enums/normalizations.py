from enum import Enum


class Normalizations(str, Enum):
    BATCH = "batch_normalization"
    ALL = "all"
    SPATIAL = "spatial_normalization"
    TEMPORAL = "temporal_normalization"
