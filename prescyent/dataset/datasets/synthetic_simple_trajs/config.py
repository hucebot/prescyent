"""Config elements for SST dataset usage"""
from typing import List

from pydantic import field_validator

from prescyent.dataset.config import TrajectoriesDatasetConfig
from prescyent.dataset.features import Features
from .metadata import DEFAULT_FEATURES, POINT_LABELS


class SSTDatasetConfig(TrajectoriesDatasetConfig):
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
    """The frequency in Hz of the dataset, If different from original data we'll use linear upsampling or downsampling of the data"""
    history_size: int = 50
    """Number of timesteps as input"""
    future_size: int = 50
    """Number of timesteps predicted as output"""
    in_features: Features = DEFAULT_FEATURES
    """List of features used as input, if None, use default from the dataset"""
    out_features: Features = DEFAULT_FEATURES
    """List of features used as output, if None, use default from the dataset"""
    in_points: List[int] = list(range(len(POINT_LABELS)))
    """Ids of the points used as input."""
    out_points: List[int] = list(range(len(POINT_LABELS)))
    """Ids of the points used as output."""

    @field_validator("context_keys")
    def check_context_keys(cls, value):
        """check that requested keys exists in the dataset"""
        if value:
            raise ValueError("This dataset cannot handle context keys")
        return value
