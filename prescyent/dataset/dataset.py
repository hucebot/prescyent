"""Standard class for motion datasets"""
import zipfile
from pathlib import Path
from typing import Dict, Union, Type

import requests
import torch
import json
from torch.utils.data import Dataset, DataLoader

from prescyent.dataset.config import LearningTypes, MotionDatasetConfig
from prescyent.dataset.trajectories import Trajectories
from prescyent.dataset.datasamples import MotionDataSamples
from prescyent.utils.logger import logger, DATASET

class MotionDataset(Dataset):
    """Base classe for all motion datasets"""
    config: MotionDatasetConfig
    config_class: Type[MotionDatasetConfig]
    batch_size: int
    history_size: int
    future_size: int
    trajectories: Trajectories
    train_datasample: MotionDataSamples
    test_datasample: MotionDataSamples
    val_datasample: MotionDataSamples

    def __init__(self) -> None:
        logger.debug("Tensor pairs will be generated for a %s learning type",
                        self.config.learning_type,
                        group=DATASET)
        self.train_datasample = MotionDataSamples(self.trajectories.train,
                                                  history_size=self.history_size,
                                                  future_size=self.future_size,
                                                  sampling_type=self.config.learning_type)
        logger.info("Train dataset has a size of %d", len(self.train_datasample))
        self.test_datasample = MotionDataSamples(self.trajectories.test,
                                                  history_size=self.history_size,
                                                  future_size=self.future_size,
                                                  sampling_type=self.config.learning_type)
        logger.info("Test dataset has a size of %d", len(self.test_datasample))
        self.val_datasample = MotionDataSamples(self.trajectories.val,
                                                  history_size=self.history_size,
                                                  future_size=self.future_size,
                                                  sampling_type=self.config.learning_type)
        logger.info("Val dataset has a size of %d", len(self.val_datasample))

    @property
    def train_dataloader(self):
        return DataLoader(self.train_datasample, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.config.num_workers,
                          pin_memory=self.config.pin_memory,
                          drop_last=self.config.drop_last,
                          persistent_workers=self.config.persistent_workers)

    @property
    def test_dataloader(self):
        return DataLoader(self.test_datasample, batch_size=self.batch_size,
                          shuffle=False, num_workers=1,
                          pin_memory=self.config.pin_memory,
                          drop_last=False, persistent_workers=False)

    @property
    def val_dataloader(self):
        return DataLoader(self.val_datasample, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.config.num_workers,
                          pin_memory=self.config.pin_memory,
                          drop_last=False,
                          persistent_workers=self.config.persistent_workers)

    @property
    def num_points(self):
        return self.trajectories.train[0].shape[1]

    @property
    def num_dims(self):
        return self.trajectories.train[0].shape[2]

    @property
    def feature_size(self):
        return self.num_dims * self.num_points

    def _init_from_config(self,
                          config: Union[Dict, None, str, Path, MotionDatasetConfig],
                          config_class: Type[MotionDatasetConfig]):
        self.config_class = config_class
        if isinstance(config, dict):  # use pydantic for dict verification
            config = config_class(**config)
        elif config is None:  # use default config if none
            logger.info("Using default config because none was provided", group=DATASET)
            config = config_class()
        elif isinstance(config, str):  # load from a string
            config = Path(config)
        config = self._load_config(config)
        assert isinstance(config, config_class)   # check our config type
        self.config = config
        self.history_size = config.history_size
        self.future_size = config.future_size
        self.batch_size = config.batch_size

    def _load_config(self, config):
        if isinstance(config, Path):  # load from a Path
            logger.info("Loading config from %s", config, group=DATASET)
            with open(config, encoding="utf-8") as conf_file:
                return self.config_class(**json.load(conf_file))
        return config

    def save_config(self, save_path: Path):
        # check if parent folder exist, or create it
        if isinstance(save_path, str):
            save_path = Path(save_path)
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving config to %s", save_path, group=DATASET)
        with save_path.open('w', encoding="utf-8") as conf_file:
            logger.debug(self.config.dict(), group=DATASET)
            json.dump(self.config.dict(), conf_file, indent=4, sort_keys=True)

    def _download_files(self, url, path):
        """get the dataset files from an url"""
        logger.info("Downloading data from %s", url,
                    group=DATASET)
        data = requests.get(url, timeout=10)
        path = Path(path)
        if path.is_dir():
            path = path / "downloaded_data.zip"
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving data to %s", path,
                    group=DATASET)
        with path.open("wb") as pfile:
            pfile.write(data.content)

    def _unzip(self, zip_path: str):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(zip_path.replace(".zip", ""))
        logger.info("Archive unziped at %s", zip_path.replace(".zip", ""),
                    group=DATASET)
