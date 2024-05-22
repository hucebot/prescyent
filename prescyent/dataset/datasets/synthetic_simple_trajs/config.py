"""Config elements for SST dataset usage"""
from typing import List

from prescyent.dataset.config import MotionDatasetConfig
from prescyent.dataset.features import Feature
from .metadata import FEATURES, POINT_LABELS


class DatasetConfig(MotionDatasetConfig):
    """Pydantic Basemodel for SSTDataset configuration"""

    history_size: int = 50
    future_size: int = 50
    subsampling_step: int = 1  # subsampling -> 50 Hz to 50Hz
    num_traj: int = 1000
    ratio_train: float = 0.8
    ratio_test: float = 0.1
    ratio_val: float = 0.1
    # pos parameters
    min_x: float = 1.0
    max_x: float = 2.0
    min_y: float = -1.0
    max_y: float = 1.0
    min_z: float = -1.0
    max_z: float = 1.0
    starting_pose: List[float] = [0, 0, 0, 0, 0, 0, 1]  # CoordinateXYZ + RotationQuat
    # Controller parameters
    dt: float = 0.02
    gain_lin: float = 1.0
    gain_ang: float = 1.0
    clamp_lin: float = 0.2
    clamp_ang: float = 0.5
    in_features: List[Feature] = FEATURES
    out_features: List[Feature] = FEATURES
    in_points: List[int] = list(range(len(POINT_LABELS)))
    out_points: List[int] = list(range(len(POINT_LABELS)))
