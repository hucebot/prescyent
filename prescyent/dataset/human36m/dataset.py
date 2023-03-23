"""Class and methods for the TeleopIcub Dataset
https://zenodo.org/record/5913573#.Y75xK_7MIaw
"""
from pathlib import Path
from typing import Callable, List, Union, Dict

import numpy as np
import torch

from prescyent.dataset.trajectories import Trajectory
from prescyent.utils.logger import logger, DATASET
from prescyent.dataset.dataset import MotionDataset, Trajectories
from prescyent.utils.dataset_manipulation import expmap2rotmat_torch, rotmat2xyz_torch, split_array_with_ratios
from prescyent.dataset.human36m.config import DatasetConfig

# 32-long list with indices into angles
ROTATION_IDS = [[5, 6, 4],
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
            []]
# 32-long list with indices into expmap angles
EXPMAP_IDS = np.split(np.arange(4,100)-1,32)

FILE_LABELS = {}
FILE_LABELS['torso'] = list(range(6))
FILE_LABELS['torso'].extend(list(range(36,51)))
FILE_LABELS['right_arm'] = list(range(75,99))
FILE_LABELS['left_arm'] = list(range(51,75))
FILE_LABELS['right_leg'] = list(range(6,21))
FILE_LABELS['left_leg'] = list(range(21,36))

POINT_LABELS = ["torso_0", "torso_1",
                "right_leg_2", "right_leg_3", "right_leg_4", "right_leg_5", "right_leg_6",
                "left_leg_7", "left_leg_8", "left_leg_9", "left_leg_10", "left_leg_11",
                "torso_12", "torso_13", "torso_14", "torso_15", "torso_16",
                "left_arm_17", "left_arm_18", "left_arm_19", "left_arm_20", "left_arm_21",
                "left_arm_22", "left_arm_23", "left_arm_24",
                "right_arm_25", "right_arm_26", "right_arm_27", "right_arm_28", "right_arm_29",
                "right_arm_30", "right_arm_31", "right_arm_32",
                ]

class Dataset(MotionDataset):
    """Class for data loading et preparation before the MotionDataset sampling
    """
    def __init__(self, config: Union[Dict, DatasetConfig] = None,
                 scaler: Callable = None):
        self._init_from_config(config, DatasetConfig)
        if not Path(self.config.data_path).exists():
            self._get_from_web()
        self.trajectories = self._load_files()
        super().__init__(scaler)


    # load a set of trajectory, keeping them separate
    def _load_files(self):
        """read txt files and create trajectories"""
        logger.info("Reading files from %s", self.config.data_path,
                    group=DATASET)
        train_files = self._get_filenames_for_subject(self.config.subjects_train)
        val_files = self._get_filenames_for_subject(self.config.subjects_val)
        test_files = self._get_filenames_for_subject(self.config.subjects_test)
        # each files gives an expmap and has to be converted into xyz
        train = pathfiles_to_trajectories(train_files, used_joints=self.config.used_joints,
                                        subsampling_step=self.config.subsampling_step)
        logger.info("Found %d trajectories in the train set", len(train), group=DATASET)
        test = pathfiles_to_trajectories(test_files, used_joints=self.config.used_joints,
                                        subsampling_step=self.config.subsampling_step)
        logger.info("Found %d trajectories in the test set", len(test), group=DATASET)
        val = pathfiles_to_trajectories(val_files, used_joints=self.config.used_joints,
                                        subsampling_step=self.config.subsampling_step)
        logger.info("Found %d trajectories in the val set", len(val), group=DATASET)
        return Trajectories(train, test, val)

    def _get_filenames_for_subject(self, subject_names:List[str]):
        filenames = []
        for subject_name in subject_names:
            for action in self.config.actions :
                filenames += list((Path(self.config.data_path) / subject_name)
                                .rglob(f"{action}*.txt"))
        return filenames

    def _get_from_web(self):
        raise NotImplementedError("This dataset must be downloaded manually, "
                                  "please follow the instructions in the README")


def pathfiles_to_trajectories(files: List,
                              delimiter: str = ',',
                              subsampling_step: int = 0,
                              used_joints: List[int] = None) -> list:
    """util method to turn a list of pathfiles to a list of their data
    :rtype: list
    """
    trajectory_arrray = list()
    for file in files:
        expmap = file.open().readlines()
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
        pose_info = expmap2rotmat_torch(torch.from_numpy(pose_info).float()).reshape(T, 32, 3, 3)
        xyz_info = rotmat2xyz_torch(pose_info)
        xyz_info = xyz_info[::subsampling_step, used_joints, :]
        trajectory = Trajectory(xyz_info, file, [POINT_LABELS[i] for i in used_joints])
        trajectory_arrray.append(trajectory)
    return trajectory_arrray
