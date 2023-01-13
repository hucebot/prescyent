"""Class and methods for the TeleopIcub Dataset
https://zenodo.org/record/5913573#.Y75xK_7MIaw
"""
from typing import Union, Dict

from pathlib import Path

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
        self.input_length = config.input_length
        self.output_length = config.output_length
        self.batch_size = config.batch_size

    # load a set of trajectory, keeping them separate
    def _load_files(self):
        files = list(Path(self.config.data_path).rglob(self.config.glob_dir))
        if len(files) == 0:
            print("ERROR: no files matching '%s' rule for this path %s" %
                  (self.config.glob_dir, self.config.data_path))
            raise FileNotFoundError(self.config.data_path)
        train_files, test_files, val_files = split_array_with_ratios(files,
                                                                     self.config.ratio_train,
                                                                     self.config.ratio_test,
                                                                     self.config.ratio_val,
                                                                     shuffle=self.config.shuffle)
        train_data = pathfiles_to_array(train_files,
                                        skip_data=self.config.skip_data,
                                        column=self.config.column)
        test_data = pathfiles_to_array(test_files,
                                       skip_data=self.config.skip_data,
                                       column=self.config.column)
        val_data = pathfiles_to_array(val_files,
                                      skip_data=self.config.skip_data,
                                      column=self.config.column)
        return Episodes(train_data, test_data, val_data)

    def _get_from_web(self):
        self._download_files(self.config.url,
                             self.config.data_path + ".zip")
        self._unzip(self.config.data_path + ".zip")
