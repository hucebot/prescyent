"""Config elements for H36M dataset usage"""
import os
from typing import List, Optional

from prescyent.dataset.config import DEFAULT_DATA_PATH, MotionDatasetConfig
from prescyent.dataset.features import Feature
from .metadata import FEATURES, POINT_LABELS


class DatasetConfig(MotionDatasetConfig):
    """Pydantic Basemodel for Dataset configuration"""

    url: Optional[str] = None
    """Url used to download the dataset"""
    data_path: str = os.path.join(DEFAULT_DATA_PATH, "h36m")
    """Directory where the data files is"""
    subsampling_step: int = 2
    """Ratio used to downsample the frames of the trajectories
    Default is subsampling -> 50 Hz to 25Hz"""
    used_joints: List[int] = [
        2,
        3,
        4,
        5,
        7,
        8,
        9,
        10,
        12,
        13,
        14,
        15,
        17,
        18,
        19,
        21,
        22,
        25,
        26,
        27,
        29,
        30,
    ]  # indexes of the joints, default is taken from benchmarks like siMLPe's
    """Ids of the joints loaded. Default is all joints"""
    actions: List[str] = [
        "directions",
        "discussion",
        "eating",
        "greeting",
        "phoning",
        "posing",
        "purchases",
        "sitting",
        "sittingdown",
        "smoking",
        "takingphoto",
        "waiting",
        "walking",
        "walkingdog",
        "walkingtogether",
    ]
    """List of the H36M Actions to consider"""
    subjects_train: List[str] = ["S1", "S6", "S7", "S8", "S9"]
    """Subject from which's trajectories are placed in Trajectories.train"""
    subjects_test: List[str] = ["S5"]
    """Subject from which's trajectories are placed in Trajectories.test"""
    subjects_val: List[str] = ["S11"]
    """Subject from which's trajectories are placed in Trajectories.val"""
    history_size: int = 25
    """number of timesteps as input, default to 1sec at 25Hz"""
    future_size: int = 25
    """number of predicted timesteps, default to 1sec at 25Hz"""
    # Override default values with the dataset's
    in_features: List[Feature] = FEATURES
    out_features: List[Feature] = FEATURES
    in_points: List[int] = list(range(len(used_joints)))
    out_points: List[int] = list(range(len(used_joints)))
