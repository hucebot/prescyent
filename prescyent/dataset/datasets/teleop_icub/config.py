"""Config elements for TeleopIcub dataset usage"""
import os
from typing import List

from prescyent.dataset.config import DEFAULT_DATA_PATH, MotionDatasetConfig
from prescyent.dataset.features import Feature
from .metadata import FEATURES, POINT_LABELS


class DatasetConfig(MotionDatasetConfig):
    """Pydantic Basemodel for TeleopIcubDataset configuration"""

    url: str = "https://zenodo.org/record/5913573/files/AndyData-lab-prescientTeleopICub.zip?download=1"
    """Url used to download the dataset"""
    data_path: str = os.path.join(DEFAULT_DATA_PATH, "AndyData-lab-prescientTeleopICub")
    """Directory where the data files is"""
    glob_dir: str = "datasetMultipleTasks/*/p*.csv"
    """Pattern used to find the list of files using a rglob method"""
    shuffle_data_files: bool = True
    """If True the list of files is shuffled"""
    subsampling_step: int = 10
    """Ratio used to downsample the frames of the trajectories
    Default is subsampling -> 100 Hz to 10Hz"""
    used_joints: List[int] = list(range(len(POINT_LABELS)))
    """Ids of the joints loaded. Default is all joints"""
    ratio_train: float = 0.8
    """ratio of trajectories placed in Trajectories.train"""
    ratio_test: float = 0.15
    """ratio of trajectories placed in Trajectories.test"""
    ratio_val: float = 0.05
    """ratio of trajectories placed in Trajectories.val"""
    # Override default values with the dataset's
    history_size: int = 10
    """number of timesteps as input, default to 1s at 10Hz"""
    future_size: int = 10
    """number of predicted timesteps, default to 1s at 10Hz"""
    in_features: List[Feature] = FEATURES
    out_features: List[Feature] = FEATURES
    in_points: List[int] = list(range(len(used_joints)))
    out_points: List[int] = list(range(len(used_joints)))
