"""Class and methods for the Human 3.6M Dataset"""
from pathlib import Path
from typing import List, Union, Dict

import numpy as np
import torch

from prescyent.utils.logger import logger, DATASET
from prescyent.dataset.dataset import MotionDataset
from prescyent.dataset.trajectories.trajectories import Trajectories
from prescyent.dataset.trajectories.trajectory import Trajectory
from prescyent.utils.dataset_manipulation import (
    expmap2rotmat_torch,
    rotmat2xyz_torch,
    update_parent_ids,
)
from prescyent.dataset.features import CoordinateXYZ, RotationRep6D, convert_to_rep6d
import prescyent.dataset.datasets.human36m.metadata as metadata
from prescyent.dataset.datasets.human36m.config import DatasetConfig


class Dataset(MotionDataset):
    """Class for data loading et preparation before the MotionDataset sampling"""

    DATASET_NAME = "H36M"

    def __init__(self, config: Union[Dict, DatasetConfig] = None) -> None:
        self._init_from_config(config, DatasetConfig)
        if not Path(self.config.data_path).exists():
            logger.getChild(DATASET).warning(
                "Dataset files not found at path %s",
                self.config.data_path,
            )
            self._get_from_web()
        self.trajectories = self._load_files()
        super().__init__(self.DATASET_NAME)

    # load a set of trajectory, keeping them separate
    def _load_files(self) -> Trajectories:
        """read txt files and create trajectories"""
        logger.getChild(DATASET).info("Reading files from %s", self.config.data_path)
        train_files = self._get_filenames_for_subject(self.config.subjects_train)
        val_files = self._get_filenames_for_subject(self.config.subjects_val)
        test_files = self._get_filenames_for_subject(self.config.subjects_test)
        # each files gives an expmap and has to be converted into xyz
        train = self.pathfiles_to_trajectories(train_files)
        logger.getChild(DATASET).info(
            "Found %d trajectories in the train set", len(train)
        )
        test = self.pathfiles_to_trajectories(test_files)
        logger.getChild(DATASET).info(
            "Found %d trajectories in the test set", len(test)
        )
        val = self.pathfiles_to_trajectories(val_files)
        logger.getChild(DATASET).info("Found %d trajectories in the val set", len(val))
        return Trajectories(train, test, val)

    def _get_filenames_for_subject(self, subject_names: List[str]) -> List[Path]:
        filenames = []
        for subject_name in subject_names:
            for action in self.config.actions:
                filenames += list(
                    (Path(self.config.data_path) / subject_name).rglob(
                        f"{action}_*.txt"
                    )
                )
        filenames.sort()
        return filenames

    def _get_from_web(self) -> None:
        raise NotImplementedError(
            "This dataset must be downloaded manually, "
            "please follow the instructions in the README"
        )

    def pathfiles_to_trajectories(
        self,
        files: List,
        delimiter: str = ",",
    ) -> List[Trajectory]:
        """util method to turn a list of pathfiles to a list of their data
        :rtype: List[Trajectory]
        """
        used_joints = self.config.used_joints
        subsampling_step = self.config.subsampling_step
        trajectory_arrray = list()
        if used_joints is None:
            used_joints = list(range(len(metadata.POINT_LABELS)))
        for file_path in files:
            with file_path.open() as file:
                expmap = file.readlines()
            pose_info = []
            for line in expmap:
                line = line.strip().split(delimiter)
                if len(line) > 0:
                    pose_info.append(np.array([float(x) for x in line]))
            # get expmap from file
            pose_info = np.array(pose_info)
            S = pose_info.shape[0]
            pose_info = pose_info.reshape(-1, 33, 3)
            pose_info[:, :2] = 0
            pose_info = pose_info[:, 1:, :].reshape(-1, 3)
            # get rotmatrices from expmap
            rotmatrices = expmap2rotmat_torch(
                torch.from_numpy(pose_info).float()
            ).reshape(S, 32, 3, 3)
            # get xyz and world relative rotmatrice from joint rotmatrices and bone/parent infos
            xyz_info, world_rotmatrices = rotmat2xyz_torch(
                rotmatrices.clone().detach(), metadata._get_metadata
            )
            xyz_info = (
                xyz_info[::subsampling_step, used_joints, :] / 1000
            )  # mm to meter conversion
            rotmatrices = world_rotmatrices[::subsampling_step, used_joints, :]
            S, P = xyz_info.shape[0], xyz_info.shape[1]
            # as we permuted x and y in "fkl_torch", we permute x and y axis in rotmatrices
            permute_x_y = torch.FloatTensor(
                [
                    [0.0000000, 1.0000000, 0.0000000],
                    [1.0000000, 0.0000000, 0.0000000],
                    [0.0000000, 0.0000000, 1.0000000],
                ]
            )
            rotmatrices = (
                permute_x_y @ rotmatrices.reshape(-1, 3, 3) @ permute_x_y.T
            ).reshape(S, P, 9)
            # We use REP6D as default representation for our rotations
            rot6d_info = convert_to_rep6d(rotmatrices)
            position_traj_tensor = torch.cat((xyz_info, rot6d_info), dim=-1)
            freq = (
                metadata.BASE_FREQUENCY // subsampling_step
                if subsampling_step
                else metadata.BASE_FREQUENCY
            )
            title = f"{Path(file_path).parts[-2]}_{Path(file_path).stem}"
            trajectory = Trajectory(
                tensor=position_traj_tensor,
                frequency=freq,
                tensor_features=[
                    CoordinateXYZ(list(range(3))),
                    RotationRep6D(list(range(3, 9))),
                ],
                file_path=file_path,
                title=title,
                point_parents=update_parent_ids(used_joints, metadata.POINT_PARENTS),
                point_names=[metadata.POINT_LABELS[i] for i in used_joints],
            )
            trajectory_arrray.append(trajectory)
        return trajectory_arrray
