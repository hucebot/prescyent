"""Class and methods for the TeleopIcub Dataset
https://zenodo.org/record/5913573#.Y75xK_7MIaw
"""
from typing import Union, Dict

from pathlib import Path

from prescyent.utils.logger import logger, DATASET
from prescyent.dataset.motion.dataset import MotionDataset, Episodes
from prescyent.utils.dataset_manipulation import split_array_with_ratios, pathfiles_to_array
from prescyent.dataset.motion.teleop_icub.config import TeleopIcubDatasetConfig


class TeleopIcubDataset(MotionDataset):
    """TODO: present the dataset here
    Architecture

    Dataset is not splitted into test / train / val
    It as to be at initialisation, througt the parameters
    """
    def __init__(self, config: Union[Dict, TeleopIcubDatasetConfig]=None, scaler=None):
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
        self.num_workers = config.num_workers

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
        train_data = pathfiles_to_array(train_files,
                                        subsampling_step=self.config.subsampling_step,
                                        dimensions=self.config.dimensions)
        test_data = pathfiles_to_array(test_files,
                                       subsampling_step=self.config.subsampling_step,
                                       dimensions=self.config.dimensions)
        val_data = pathfiles_to_array(val_files,
                                      subsampling_step=self.config.subsampling_step,
                                      dimensions=self.config.dimensions)
        self.feature_size = train_data[0].shape[1]
        return Episodes(train_data, test_data, val_data)

    def _get_from_web(self):
        self._download_files(self.config.url,
                             self.config.data_path + ".zip")
        self._unzip(self.config.data_path + ".zip")
