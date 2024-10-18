"""Config elements for Scaling usage"""
from typing import Literal, Optional

from prescyent.base_config import BaseConfig
from prescyent.utils.enums import Scalers, TrajectoryDimensions


class ScalerConfig(BaseConfig):
    """Pydantic Basemodel for Scaling configuration"""

    scaler: Optional[Scalers] = Scalers.STANDARDIZATION
    """scaling method to use. If None we will not use scaling"""
    scaling_axis: Literal[
        TrajectoryDimensions.FEATURE,
        TrajectoryDimensions.POINT,
        TrajectoryDimensions.SPATIAL,
        TrajectoryDimensions.TEMPORAL,
        None,
    ] = TrajectoryDimensions.SPATIAL
    """dimensions on which the scaling will be applied. If None we will not use scaling"""
    do_feature_wise_scaling: bool = False
    """if True, will train a scaler for each feature, else we scale over all feature with one scaler, Defaults to False"""
    scale_rotations: bool = False
    """if False and do feature do_feature_wise_scaling, rotations will not be scaled, Defaults to False"""
