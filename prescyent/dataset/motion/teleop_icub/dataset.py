"""Class and methods for the TeleopIcub Dataset
https://zenodo.org/record/5913573#.Y75xK_7MIaw
"""
from typing import Callable, List, Union, Dict

import numpy as np
from pathlib import Path
import torch

from prescyent.dataset.motion.episodes import Episode
from prescyent.utils.logger import logger, DATASET
from prescyent.dataset.motion.dataset import MotionDataset, Episodes
from prescyent.utils.dataset_manipulation import split_array_with_ratios
from prescyent.dataset.motion.teleop_icub.config import TeleopIcubDatasetConfig


TELEOP_DIMENSIONS_NAMES = [
    "waist_z", "right_hand_x", "right_hand_y", "right_hand_z",
    "left_hand_x", "left_hand_y", "left_hand_z"
]


class TeleopIcubDataset(MotionDataset):
    """TODO: present the dataset here
    Architecture

    Dataset is not splitted into test / train / val
    It as to be at initialisation, througt the parameters
    """
    def __init__(self, config: Union[Dict, TeleopIcubDatasetConfig] = None,
                 scaler: Callable = None):
        if not config:
            config = TeleopIcubDatasetConfig()
        self._init_from_config(config)
        if not Path(self.config.data_path).exists():
            self._get_from_web()
        self.episodes = self._load_files()
        super().__init__(scaler)

    def _init_from_config(self, config):
        if isinstance(config, dict):
            config = TeleopIcubDatasetConfig(**config)
        self.config = config
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.batch_size = config.batch_size

    # load a set of trajectory, keeping them separate
    def _load_files(self):
        files = list(Path(self.config.data_path).rglob(self.config.glob_dir))
        if len(files) == 0:
            logger.error("No files matching '%s' rule for this path %s",
                         self.config.glob_dir, self.config.data_path,
                         group=DATASET)
            raise FileNotFoundError(self.config.data_path)
        train_files, test_files, val_files = split_array_with_ratios(files,
                                                                     self.config.ratio_train,
                                                                     self.config.ratio_test,
                                                                     self.config.ratio_val,
                                                                     shuffle=self.config.shuffle)
        train_episodes = pathfiles_to_episodes(train_files,
                                               subsampling_step=self.config.subsampling_step,
                                               dimensions=self.config.dimensions)
        test_episodes = pathfiles_to_episodes(test_files,
                                              subsampling_step=self.config.subsampling_step,
                                              dimensions=self.config.dimensions)
        val_episodes = pathfiles_to_episodes(val_files,
                                             subsampling_step=self.config.subsampling_step,
                                             dimensions=self.config.dimensions)
        self.feature_size = train_episodes[0].shape[1]
        return Episodes(train_episodes, test_episodes, val_episodes)

    def _get_from_web(self):
        self._download_files(self.config.url,
                             self.config.data_path + ".zip")
        self._unzip(self.config.data_path + ".zip")


def pathfiles_to_episodes(files: List,
                          delimiter: str = ',',
                          start: int = None,
                          end: int = None,
                          subsampling_step: int = 0,
                          dimensions: List[int] = None) -> list:
    """util method to turn a list of pathfiles to a list of their data

    :param files: list of files
    :type files: List
    :param delimiter: delimiter to split the data on, defaults to ','
    :type delimiter: str, optional
    :param subsampling_step: the step used for final list, allows to skip data, defaults to 0
    :type subsampling_step: int, optional
    :param dimensions: _description_, defaults to 0
    :type dimensions: List[int], optional, defaults to None
    :raises FileNotFoundError: _description_
    :return: the data of the dataset, grouped per file
    :rtype: list
    """
    if start is None:
        start = 0
    episode_arrray = list()
    for file in files:
        if not file.exists():
            logger.error("file does not exist: %s", file,
                         group=DATASET)
            raise FileNotFoundError(file)
        file_array = np.loadtxt(file, delimiter=delimiter)
        if end is None:
            end = len(file_array)
        if dimensions is not None:
            tensor = torch.FloatTensor(file_array[start:end:subsampling_step, dimensions])
        else:
            tensor = torch.FloatTensor(file_array[start:end:subsampling_step])
            dimensions = list(range(len(TELEOP_DIMENSIONS_NAMES)))
        episode_arrray.append(Episode(tensor, file,
                                      [TELEOP_DIMENSIONS_NAMES[i] for i in dimensions]))
    return episode_arrray
