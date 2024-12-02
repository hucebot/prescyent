"""Class and methods to generate simple linear trajectories with a rotation
"""
from pathlib import Path
import tempfile
from typing import Union, Dict, List

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from prescyent.dataset.hdf5_utils import write_metadata
from prescyent.dataset.dataset import TrajectoriesDataset
from prescyent.dataset.trajectories.trajectories import Trajectories
from prescyent.utils.logger import logger, DATASET
from prescyent.utils.interpolate import update_tensor_frequency

from . import metadata
from .config import SSTDatasetConfig


SEQ = "zyx"


def clamp_vect_norm(vect, limit):
    norm = np.linalg.norm(vect, ord=2)
    axis = vect / (norm + 1e-6)
    return axis * min(norm, limit)


class SSTDataset(TrajectoriesDataset):
    """Simple dataset with generated trajectories from a starting and ending pose"""

    DATASET_NAME = "SST"

    def __init__(
        self,
        config: Union[Dict, SSTDatasetConfig, str, Path] = None,
    ) -> None:
        logger.getChild(DATASET).info(
            f"Initializing {self.DATASET_NAME} Dataset",
        )
        self._init_from_config(config, SSTDatasetConfig)
        super().__init__(name=self.DATASET_NAME)

    def prepare_data(self):
        """create a list of Trajectories from config variables"""
        if hasattr(self, "_trajectories"):
            return
        self.tmp_hdf5 = tempfile.NamedTemporaryFile(suffix=".hdf5")
        frequency = 1 / self.config.dt
        tmp_hdf5_data = h5py.File(self.tmp_hdf5.name, "w")
        write_metadata(
            tmp_hdf5_data,
            frequency=frequency,
            point_parents=metadata.POINT_PARENTS,
            point_names=metadata.POINT_LABELS,
            features=metadata.DEFAULT_FEATURES,
        )
        np.random.seed(self.config.seed)
        for i in tqdm(
            range(int(self.config.num_traj * self.config.ratio_train)),
            desc="Generating train_trajectories",
            colour="blue",
        ):
            tensor = self.generate_traj()
            context = {}
            tensor, context = update_tensor_frequency(
                tensor,
                frequency,
                self.config.frequency,
                metadata.DEFAULT_FEATURES,
                context,
            )
            tmp_hdf5_data.create_dataset(f"train/synthetic_traj_{i}/traj", data=tensor)
        for i in tqdm(
            range(int(self.config.num_traj * self.config.ratio_test)),
            desc="Generating test_trajectories",
            colour="blue",
        ):
            tensor = self.generate_traj()
            tensor, context = update_tensor_frequency(
                tensor,
                frequency,
                self.config.frequency,
                metadata.DEFAULT_FEATURES,
                context,
            )
            tmp_hdf5_data.create_dataset(f"test/synthetic_traj_{i}/traj", data=tensor)
        for i in tqdm(
            range(int(self.config.num_traj * self.config.ratio_val)),
            desc="Generating val_trajectories",
            colour="blue",
        ):
            tensor = self.generate_traj()
            tensor, context = update_tensor_frequency(
                tensor,
                frequency,
                self.config.frequency,
                metadata.DEFAULT_FEATURES,
                context,
            )
            tmp_hdf5_data.create_dataset(f"val/synthetic_traj_{i}/traj", data=tensor)
        self.trajectories = Trajectories.__init_from_hdf5__(self.tmp_hdf5.name)
        tmp_hdf5_data.close()
        np.random.seed()

    def generate_traj(self) -> torch.Tensor:
        """generate smooth linear traj from a starting point and random target point
         all variables are taken from dataset config

        Returns:
            torch.Tensor: new smooth simple traj between two pose
        """
        starting_pose = np.array(self.config.starting_pose)
        target_pose = self.get_random_target()
        tensor = torch.FloatTensor(starting_pose).unsqueeze(0)
        curr_pose = starting_pose
        while not np.allclose(curr_pose, target_pose, rtol=0.001):
            curr_pose = self.controller_goto(curr_pose, target_pose)
            tensor = torch.cat(
                (tensor, torch.FloatTensor(curr_pose).unsqueeze(0)), dim=0
            )
        tensor = tensor.unsqueeze(1)
        return tensor

    def get_random_target(self) -> List[float]:
        """return a random position, given the config's attribute

        Returns:
            List[float]: [x, y, z, euler_z, euler_y, euler_x]
        """
        x = np.random.uniform(self.config.min_x, self.config.max_x)
        y = np.random.uniform(self.config.min_y, self.config.max_y)
        z = np.random.uniform(self.config.min_z, self.config.max_z)
        random_euler = list(R.random().as_euler(SEQ))  # zyx
        return np.array([x, y, z] + random_euler)

    # Code adapted from Quentin's simple controller, used to loop over next pos to reach target
    def controller_goto(self, curr_pose, target_pose) -> np.ndarray:
        """generate next pose given current and target

        Args:
            curr_pose (np.ndarray): current position and rotation
            target_pose (np.ndarray): target position and rotation

        Returns:
            np.ndarray: next pose toward target
        """
        # Compute rotation matrix from eulers
        curr_pos, curr_euler = np.array(curr_pose[:3]), np.array(curr_pose[3:])
        target_pos, target_euler = np.array(target_pose[:3]), np.array(target_pose[3:])
        curr_mat = R.from_euler(SEQ, curr_euler).as_matrix()
        target_mat = R.from_euler(SEQ, target_euler).as_matrix()
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
        return np.concatenate([curr_pos, R.from_matrix(curr_mat).as_euler(SEQ)])
