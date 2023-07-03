"""Config elements for TeleopIcub dataset usage"""
import os
from typing import List, Union
from prescyent.dataset.config import MotionDatasetConfig, DEFAULT_DATA_PATH


class DatasetConfig(MotionDatasetConfig):
    """Pydantic Basemodel for TeleopIcubDataset configuration"""

    url: Union[str, None] = None
    data_path: str = os.path.join(DEFAULT_DATA_PATH,"h36m")
    subsampling_step: int = 2  # subsampling -> 50 Hz to 25Hz
    dimensions: Union[List[int], None] = None  # num features in the data
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
    ]  # indexes of the joints
    # type of actions to load
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
    subjects_train: List[str] = ["S1", "S6", "S7", "S8", "S9"]
    subjects_test: List[str] = ["S5"]
    subjects_val: List[str] = ["S11"]
    history_size: int = 50  # number of timesteps as input
    future_size: int = 10  # number of predicted timesteps
