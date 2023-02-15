"""
    Config elements for Sine dataset usage
    Inspired by: https://github.com/pytorch/examples/tree/main/time_sequence_prediction
"""
from typing import List, Union
from prescyent.dataset.motion.config import MotionDatasetConfig


class SineDatasetConfig(MotionDatasetConfig):
    """Pydantic Basemodel for TeleopIcubDataset configuration"""
    dimensions: Union[List[int], None] = [2]         # dimension in the data
    # number of trajectories
    size: int = 100
    # period of the sine waves
    period : int = 20
    # length of a trajectory
    length : int = 1000  
    # splits
    ratio_train: float = .8
    ratio_test: float = .15
    ratio_val: float = .05
