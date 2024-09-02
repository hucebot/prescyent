from enum import Enum


class TrajectoryDimensions(str, Enum):
    """Dimensions ids of batched trajectories"""

    BATCH = [0]
    TEMPORAL = [1]
    POINT = [2]
    FEATURE = [3]
    SPATIAL = [2, 3]
    ALL = [0, 1, 2, 3]
