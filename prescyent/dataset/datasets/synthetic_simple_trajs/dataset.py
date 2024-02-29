"""Class and methods for the SST Dataset
https://zenodo.org/record/5913573#.Y75xK_7MIaw
"""
import random
from pathlib import Path
from typing import Union, Dict, List

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from prescyent.utils.logger import logger, DATASET
from prescyent.utils.dataset_manipulation import (
    split_array_with_ratios,
    update_parent_ids,
)
from prescyent.dataset.datasets.synthetic_simple_trajs.config import DatasetConfig
from prescyent.dataset.datasets.synthetic_simple_trajs.metadata import *
from prescyent.dataset.features import CoordinateXYZ, RotationQuat
from prescyent.dataset.trajectories.trajectories import Trajectories
from prescyent.dataset.trajectories.trajectory import Trajectory
from prescyent.dataset.dataset import MotionDataset
from prescyent.utils.logger import logger, DATASET


def clamp_vect_norm(vect, limit):
    norm = np.linalg.norm(vect, ord=2)
    axis = vect / (norm + 1e-6)
    return axis * min(norm, limit)


class Dataset(MotionDataset):
    """Simple dataset with generated trajectories from a starting and ending pose"""

    DATASET_NAME = "SST"

    def __init__(self, config: Union[Dict, DatasetConfig, str, Path] = None) -> None:
        logger.getChild(DATASET).info(
            f"Initializing {self.DATASET_NAME} Dataset",
        )
        self._init_from_config(config, DatasetConfig)
        self.trajectories = self.generate_trajectories()
        super().__init__(self.DATASET_NAME)

    def generate_trajectories(self):
        """create a list of Trajectories from config variables"""
        train_trajectories = [
            self.generate_traj(i)
            for i in range(int(self.config.num_traj * self.config.ratio_train))
        ]
        logger.getChild(DATASET).info(
            f"Generated {len(train_trajectories)} train trajectories",
        )
        test_trajectories = [
            self.generate_traj(i)
            for i in range(int(self.config.num_traj * self.config.ratio_test))
        ]
        logger.getChild(DATASET).info(
            f"Generated {len(test_trajectories)} test trajectories",
        )
        val_trajectories = [
            self.generate_traj(i)
            for i in range(int(self.config.num_traj * self.config.ratio_val))
        ]
        logger.getChild(DATASET).info(
            f"Generated {len(val_trajectories)} val trajectories",
        )
        return Trajectories(train_trajectories, test_trajectories, val_trajectories)

    def generate_traj(self, id: int) -> Trajectory:
        """generate smooth linear traj from a starting point and random target point
         all variables are taken from dataset config

        Args:
            id (int): id used for trajectory name

        Returns:
            Trajectory: new smooth simple traj between two pose
        """
        starting_pose = np.array(self.config.starting_pose)
        target_pose = self.get_random_target()
        trajectory = torch.FloatTensor(starting_pose).unsqueeze(0)
        curr_pose = starting_pose
        while not np.allclose(curr_pose, target_pose, rtol=0.001):
            curr_pose = self.controller_goto(curr_pose, target_pose)
            trajectory = torch.cat(
                (trajectory, torch.FloatTensor(curr_pose).unsqueeze(0)), dim=0
            )
        trajectory = trajectory.unsqueeze(1)
        trajectory = trajectory[:: self.config.subsampling_step]
        return Trajectory(
            trajectory,
            BASE_FREQUENCY / self.config.subsampling_step,
            FEATURES,
            file_path=f"synthetic_traj_{id}",
            title=f"synthetic_traj_{id}",
        )

    def get_random_target(self) -> List[float]:
        x = random.uniform(self.config.min_x, self.config.max_x)
        y = random.uniform(self.config.min_y, self.config.max_y)
        z = random.uniform(self.config.min_z, self.config.max_z)
        random_quat = list(R.random().as_quat(canonical=True))  # x, y, z, w
        return np.array([x, y, z] + random_quat)

    # Code from Quentin, used to loop over next pos to reach target
    def controller_goto(self, curr_pose, target_pose):
        # Compute rotation matrix from quaternions
        curr_pos, curr_quat = np.array(curr_pose[:3]), np.array(curr_pose[3:])
        target_pos, target_quat = np.array(target_pose[:3]), np.array(target_pose[3:])
        curr_mat = R.from_quat(curr_quat).as_matrix()
        target_mat = R.from_quat(target_quat).as_matrix()
        # Target pose in current commanded hand pose
        rel_mat = target_mat @ curr_mat.transpose()
        rel_pos = target_pos - curr_pos
        # Compute saturated P velocity control
        vel_lin = clamp_vect_norm(self.config.gain_lin * rel_pos, self.config.clamp_lin)
        vel_ang = clamp_vect_norm(
            self.config.gain_ang * R.from_matrix(rel_mat).as_rotvec(),
            self.config.clamp_ang,
        )
        # Integrate velocity command into pose command
        curr_pos = curr_pos + self.config.dt * vel_lin
        curr_mat = curr_mat @ R.from_rotvec(self.config.dt * vel_ang).as_matrix()
        # Return updated pose command
        return np.concatenate(
            [curr_pos, R.from_matrix(curr_mat).as_quat(canonical=True)]
        )
