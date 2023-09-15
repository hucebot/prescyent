"""Class and methods for the TeleopIcub Dataset
https://zenodo.org/record/5913573#.Y75xK_7MIaw
"""
from pathlib import Path
from typing import List, Union, Dict

import numpy as np
import torch

from prescyent.dataset.trajectories import Trajectory
from prescyent.utils.logger import logger, DATASET
from prescyent.dataset.dataset import MotionDataset, Trajectories
from prescyent.utils.dataset_manipulation import split_array_with_ratios
from prescyent.dataset.teleop_icub.config import DatasetConfig

BASE_FREQUENCY = 100
FILE_LABELS = [
    "waist_z",
    "right_hand_x",
    "right_hand_y",
    "right_hand_z",
    "left_hand_x",
    "left_hand_y",
    "left_hand_z",
]
POINT_LABELS = ["waist", "right_hand", "left_hand"]


class Dataset(MotionDataset):
    """TODO: present the dataset here
    Architecture

    Dataset is not splitted into test / train / val
    It as to be at initialisation, through the parameters
    """

    DATASET_NAME = "TeleopIcub"

    def __init__(self, config: Union[Dict, DatasetConfig, str, Path] = None):
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
        logger.getChild(DATASET).debug(
            "Searching Dataset files from path %s", self.config.data_path
        )
        files = list(Path(self.config.data_path).rglob(self.config.glob_dir))
        files.sort()
        if len(files) == 0:
            logger.getChild(DATASET).error(
                "No files matching '%s' rule for this path %s",
                self.config.glob_dir,
                self.config.data_path,
            )
            raise FileNotFoundError(self.config.data_path)
        train_files, test_files, val_files = split_array_with_ratios(
            files,
            self.config.ratio_train,
            self.config.ratio_test,
            self.config.ratio_val,
            shuffle=self.config.shuffle,
        )
        train = pathfiles_to_trajectories(
            train_files,
            subsampling_step=self.config.subsampling_step,
            used_joints=self.config.used_joints,
        )
        test = pathfiles_to_trajectories(
            test_files,
            subsampling_step=self.config.subsampling_step,
            used_joints=self.config.used_joints,
        )
        val = pathfiles_to_trajectories(
            val_files,
            subsampling_step=self.config.subsampling_step,
            used_joints=self.config.used_joints,
        )
        return Trajectories(train, test, val)

    def _get_from_web(self):
        self._download_files(self.config.url, self.config.data_path + ".zip")
        self._unzip(self.config.data_path + ".zip")


def pathfiles_to_trajectories(
    files: List,
    delimiter: str = ",",
    start: int = None,
    end: int = None,
    subsampling_step: int = 0,
    used_joints: List[int] = None,
) -> list:
    """util method to turn a list of pathfiles to a list of their data

    :param files: list of files
    :type files: List
    :param delimiter: delimiter to split the data on, defaults to ','
    :type delimiter: str, optional
    :param subsampling_step: the step used for final list, allows to skip data, defaults to 0
    :type subsampling_step: int, optional
    :param used_joints: _description_, defaults to 0
    :type used_joints: List[int], optional, defaults to None
    :raises FileNotFoundError: _description_
    :return: the data of the dataset, grouped per file
    :rtype: list
    """
    if start is None:
        start = 0
    trajectory_arrray = list()
    for file in files:
        if not file.exists():
            logger.getChild(DATASET).error("file does not exist: %s", file)
            raise FileNotFoundError(file)
        file_sequence = np.loadtxt(file, delimiter=delimiter)
        if end is None:
            end = len(file_sequence)
        file_sequence = file_sequence[start:end:subsampling_step]
        # add waist x and waist y
        tensor = torch.FloatTensor(
            np.array(
                [
                    np.concatenate((np.zeros(2), file_timestep))
                    for file_timestep in file_sequence
                ]
            )
        )
        # reshape (seq_len, 9) => (seq_len, 3, 3)
        seq_len = tensor.shape[0]
        tensor = tensor.reshape(seq_len, 3, 3)
        # keep only asked joints
        if used_joints is None:
            used_joints = list(range(len(POINT_LABELS)))
        tensor = tensor[:, used_joints]
        freq = (
            BASE_FREQUENCY // subsampling_step if subsampling_step else BASE_FREQUENCY
        )
        title = f"{Path(file).parts[-3]}_{Path(file).parts[-2]}_{Path(file).stem}"
        trajectory_arrray.append(
            Trajectory(
                tensor, freq, file, title, [POINT_LABELS[i] for i in used_joints]
            )
        )
    return trajectory_arrray
