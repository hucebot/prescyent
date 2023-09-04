"""Class and methods for the Human 3.6M Dataset"""
from pathlib import Path
from typing import List, Union, Dict

import numpy as np
import torch

from prescyent.dataset.trajectories import Trajectory
from prescyent.utils.logger import logger, DATASET
from prescyent.dataset.dataset import MotionDataset, Trajectories
from prescyent.utils.dataset_manipulation import expmap2rotmat_torch, rotmat2xyz_torch
from prescyent.dataset.human36m.config import DatasetConfig

# 32-long list with indices into angles
ROTATION_IDS = [
    [5, 6, 4],
    [8, 9, 7],
    [11, 12, 10],
    [14, 15, 13],
    [17, 18, 16],
    [],
    [20, 21, 19],
    [23, 24, 22],
    [26, 27, 25],
    [29, 30, 28],
    [],
    [32, 33, 31],
    [35, 36, 34],
    [38, 39, 37],
    [41, 42, 40],
    [],
    [44, 45, 43],
    [47, 48, 46],
    [50, 51, 49],
    [53, 54, 52],
    [56, 57, 55],
    [],
    [59, 60, 58],
    [],
    [62, 63, 61],
    [65, 66, 64],
    [68, 69, 67],
    [71, 72, 70],
    [74, 75, 73],
    [],
    [77, 78, 76],
    [],
]
# 32-long list with indices into expmap angles
EXPMAP_IDS = np.split(np.arange(4, 100) - 1, 32)

POINT_LABELS = [
    "crotch_0",
    "right_hip_1",
    "right knee_2",
    "right_foot_3",
    "right_foot_4",
    "right_foot_5",
    "left_hip_6",
    "left_knee_7",
    "left_foot_8",
    "left_foot_9",
    "left_foot_10",
    "crotch_11",
    "torso_12",
    "neck_13",
    "head_14",
    "top_head_15",
    "neck_16",
    "left_shoulder_17",
    "left_elbow_18",
    "left_wrist_19",
    "left_wrist_20",
    "left_hand_21",
    "left_hand_22",
    "left_hand_23",
    "neck_24",
    "right_shoulder_25",
    "right_elbow_26",
    "right_wrist_27",
    "right_wrist_28",
    "right_hand_29",
    "right_hand_30",
    "right_hand_31",
    # "right_arm_32",
]

FILE_LABELS = []
for point in POINT_LABELS:
    FILE_LABELS.append(point + "_x")
    FILE_LABELS.append(point + "_y")
    FILE_LABELS.append(point + "_z")


class Dataset(MotionDataset):
    """Class for data loading et preparation before the MotionDataset sampling"""

    DATASET_NAME = "H36M"

    def __init__(self, config: Union[Dict, DatasetConfig] = None):
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
    def _load_files(self):
        """read txt files and create trajectories"""
        logger.getChild(DATASET).info("Reading files from %s", self.config.data_path)
        train_files = self._get_filenames_for_subject(self.config.subjects_train)
        val_files = self._get_filenames_for_subject(self.config.subjects_val)
        test_files = self._get_filenames_for_subject(self.config.subjects_test)
        # each files gives an expmap and has to be converted into xyz
        train = self.pathfiles_to_trajectories(
            train_files,
            used_joints=self.config.used_joints,
            subsampling_step=self.config.subsampling_step,
        )
        logger.getChild(DATASET).info("Found %d trajectories in the train set", len(train))
        test = self.pathfiles_to_trajectories(
            test_files,
            used_joints=self.config.used_joints,
            subsampling_step=self.config.subsampling_step,
        )
        logger.getChild(DATASET).info("Found %d trajectories in the test set", len(test))
        val = self.pathfiles_to_trajectories(
            val_files,
            used_joints=self.config.used_joints,
            subsampling_step=self.config.subsampling_step,
        )
        logger.getChild(DATASET).info("Found %d trajectories in the val set", len(val))
        return Trajectories(train, test, val)

    def _get_filenames_for_subject(self, subject_names: List[str]):
        filenames = []
        for subject_name in subject_names:
            for action in self.config.actions:
                filenames += list(
                    (Path(self.config.data_path) / subject_name).rglob(
                        f"{action}_*.txt"
                    )
                )
        return filenames

    def _get_from_web(self):
        raise NotImplementedError(
            "This dataset must be downloaded manually, "
            "please follow the instructions in the README"
        )

    def pathfiles_to_trajectories(
        self,
        files: List,
        delimiter: str = ",",
        subsampling_step: int = 0,
        used_joints: List[int] = None,
    ) -> list:
        """util method to turn a list of pathfiles to a list of their data
        :rtype: list
        """
        trajectory_arrray = list()
        for file_path in files:
            with file_path.open() as file:
                expmap = file.readlines()
            pose_info = []
            for line in expmap:
                line = line.strip().split(delimiter)
                if len(line) > 0:
                    pose_info.append(np.array([float(x) for x in line]))
            pose_info = np.array(pose_info)
            T = pose_info.shape[0]
            pose_info = pose_info.reshape(-1, 33, 3)
            pose_info[:, :2] = 0
            pose_info = pose_info[:, 1:, :].reshape(-1, 3)
            pose_info = expmap2rotmat_torch(
                torch.from_numpy(pose_info).float()
            ).reshape(T, 32, 3, 3)
            xyz_info = rotmat2xyz_torch(pose_info)
            xyz_info = (
                xyz_info[::subsampling_step, used_joints, :] / 1000
            )  # meter conversion
            trajectory = Trajectory(
                xyz_info, file_path, [POINT_LABELS[i] for i in used_joints]
            )
            trajectory_arrray.append(trajectory)
        return trajectory_arrray
