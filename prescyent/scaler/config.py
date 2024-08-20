"""Config elements for Scaling usage"""
from typing import Literal

from prescyent.base_config import BaseConfig
from prescyent.utils.enums import Scalers, TrajectoryDimensions


class ScalerConfig(BaseConfig):
    """Pydantic Basemodel for Scaling configuration"""

    scale_rotations: bool = False
    do_feature_wise_scaling: bool = False
    scaler: Scalers = Scalers.STANDARDIZATION
    scaling_axis: Literal[
        TrajectoryDimensions.FEATURE,
        TrajectoryDimensions.POINT,
        TrajectoryDimensions.SPATIAL,
        TrajectoryDimensions.TEMPORAL,
    ] = TrajectoryDimensions.SPATIAL
    # TODO: feature wise scaler, and checks
