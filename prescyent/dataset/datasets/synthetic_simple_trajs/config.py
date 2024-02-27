"""Config elements for SST dataset usage"""
from typing import List

import torch

from prescyent.dataset.config import  MotionDatasetConfig


class DatasetConfig(MotionDatasetConfig):
    """Pydantic Basemodel for SSTDataset configuration"""
    subsampling_step: int = 1  # subsampling -> 10 Hz to 10Hz
    num_traj: int = 1000
    ratio_train: float = .8
    ratio_test: float = .1
    ratio_val: float = .1
    # pos parameters
    min_x: float = 1.
    max_x: float = 2.
    min_y: float = -1.
    max_y: float = 1.
    min_z: float = -1.
    max_z: float = 1.
    starting_pose: List[float] = [0,0,0,0,0,0,1] #CoordinateXYZ + RotationQuat
    #Controller parameters
    dt: float = .1
    gain_lin: float = 1.
    gain_ang: float = 1.
    clamp_lin: float = .2
    clamp_ang: float = .5
