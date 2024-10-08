"""Config elements for TeleopIcub dataset usage"""
import os
from typing import List, Optional

from pydantic import model_validator, ValidationError

from prescyent.dataset.config import DEFAULT_DATA_PATH, MotionDatasetConfig
from prescyent.dataset.features import Features
from .metadata import DEFAULT_FEATURES, POINT_LABELS, CONTEXT_KEYS


class DatasetConfig(MotionDatasetConfig):
    """Pydantic Basemodel for TeleopIcubDataset configuration"""

    hdf5_path: str = os.path.join(
        DEFAULT_DATA_PATH, "AndyData-lab-prescientTeleopICub.hdf5"
    )
    """Path to the hdf5 data file"""
    subsets: Optional[List[str]] = None
    """Pattern used to find the list of files using a rglob method"""
    shuffle_data_files: bool = True
    """If True the list of files is shuffled"""
    ratio_train: float = 0.7
    """ratio of trajectories placed in Trajectories.train"""
    ratio_test: float = 0.2
    """ratio of trajectories placed in Trajectories.test"""
    ratio_val: float = 0.1
    """ratio of trajectories placed in Trajectories.val"""
    # Override default values with the dataset's
    frequency: int = 10
    """The frequency in Hz of the dataset
    If different from original data we'll use linear upsampling or downsampling of the data
    Default is downsampling 100Hz to 10Hz"""
    history_size: int = 10
    """number of timesteps as input, default to 1s at 10Hz"""
    future_size: int = 10
    """number of predicted timesteps, default to 1s at 10Hz"""
    in_features: Features = DEFAULT_FEATURES
    out_features: Features = DEFAULT_FEATURES
    in_points: List[int] = list(range(len(POINT_LABELS)))
    out_points: List[int] = list(range(len(POINT_LABELS)))

    @model_validator(mode="after")
    def check_context_keys(self):
        """check that requested keys exists in the dataset"""
        if self.context_keys:
            for key in self.context_keys:
                if key not in CONTEXT_KEYS:
                    raise ValidationError(f"{key} is not valid in context_keys")
        return self
