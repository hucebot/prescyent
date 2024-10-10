"""Config elements for SST dataset usage"""
from typing import List

from pydantic import model_validator, ValidationError

from prescyent.dataset.config import MotionDatasetConfig
from prescyent.dataset.features import Features
from .metadata import DEFAULT_FEATURES, POINT_LABELS


class DatasetConfig(MotionDatasetConfig):
    """Pydantic Basemodel for SSTDataset configuration"""

    num_traj: int = 1000
    """Number of trajectories to generate"""
    ratio_train: float = 0.8
    """ratio of trajectories placed in Trajectories.train"""
    ratio_test: float = 0.1
    """ratio of trajectories placed in Trajectories.test"""
    ratio_val: float = 0.1
    """ratio of trajectories placed in Trajectories.val"""
    # pos parameters
    min_x: float = 1.0
    """min value for x"""
    max_x: float = 2.0
    """max value for x"""
    min_y: float = -1.0
    """min value for y"""
    max_y: float = 1.0
    """max value for y"""
    min_z: float = -1.0
    """min value for z"""
    max_z: float = 1.0
    """max value for z"""
    starting_pose: List[float] = [0, 0, 0, 0, 0, 0]
    """position used as the starting point of all generated trajectories,
    the features are [CoordinateXYZ(range(3)), RotationEuler(range(3, 6))]"""
    # Controller parameters
    dt: float = 0.02
    """frequency of the controller"""
    gain_lin: float = 1.0
    """linear gain for the \"controller\""""
    gain_ang: float = 1.0
    """angular gain for the \"controller\""""
    clamp_lin: float = 0.2
    """max value for the linear speed"""
    clamp_ang: float = 0.5
    """max value for the angular speed"""
    # Override default values with the dataset's
    frequency: int = 50
    history_size: int = 50
    future_size: int = 50
    in_features: Features = DEFAULT_FEATURES
    out_features: Features = DEFAULT_FEATURES
    in_points: List[int] = list(range(len(POINT_LABELS)))
    out_points: List[int] = list(range(len(POINT_LABELS)))

    @model_validator(mode="after")
    def check_context_keys(self):
        """check that requested keys exists in the dataset"""
        if self.context_keys:
            raise ValidationError("This dataset cannot handle context keys")
        return self
