"""Config elements for H36M dataset usage"""
import os
from typing import List, Optional

from pydantic import model_validator, ValidationError

from prescyent.dataset.config import DEFAULT_DATA_PATH, MotionDatasetConfig
from prescyent.dataset.features import Features
from .metadata import DEFAULT_ACTIONS, DEFAULT_FEATURES, DEFAULT_USED_JOINTS


class DatasetConfig(MotionDatasetConfig):
    """Pydantic Basemodel for Dataset configuration"""

    hdf5_path: str = os.path.join(DEFAULT_DATA_PATH, "h36m.hdf5")
    """Path to the hdf5 data file"""
    actions: List[str] = DEFAULT_ACTIONS
    """List of the H36M Actions to consider"""
    subjects_train: List[str] = ["S1", "S6", "S7", "S8", "S9"]
    """Subject from which's trajectories are placed in Trajectories.train"""
    subjects_test: List[str] = ["S5"]
    """Subject from which's trajectories are placed in Trajectories.test"""
    subjects_val: List[str] = ["S11"]
    """Subject from which's trajectories are placed in Trajectories.val"""
    frequency: int = 25
    """The frequency in Hz of the dataset
    If different from original data we'll use linear upsampling or downsampling of the data
    Default is downsampling 50 Hz to 25Hz"""
    history_size: int = 25
    """number of timesteps as input, default to 1sec at 25Hz"""
    future_size: int = 25
    """number of predicted timesteps, default to 1sec at 25Hz"""
    # Override default values with the dataset's
    in_features: Features = DEFAULT_FEATURES
    out_features: Features = DEFAULT_FEATURES
    in_points: List[int] = DEFAULT_USED_JOINTS
    out_points: List[int] = DEFAULT_USED_JOINTS

    @model_validator(mode="after")
    def check_context_keys(self):
        """check that requested keys exists in the dataset"""
        if self.context_keys:
            raise ValidationError("This dataset cannot handle context keys")
        return self
