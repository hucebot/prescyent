"""Standard class for motion datasets"""
import zipfile
from pathlib import Path
from typing import Callable, Dict, Union, Type

import requests
import torch
import json
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from prescyent.dataset.config import LearningTypes, MotionDatasetConfig
from prescyent.dataset.trajectories import Trajectories
from prescyent.dataset.datasamples import MotionDataSamples
from prescyent.utils.logger import logger, DATASET

class MotionDataset(Dataset):
    """Base classe for all motion datasets"""
    config: MotionDatasetConfig
    scaler: StandardScaler
    batch_size: int
    history_size: int
    future_size: int
    trajectories: Trajectories
    train_datasample: MotionDataSamples
    test_datasample: MotionDataSamples
    val_datasample: MotionDataSamples

    def __init__(self, scaler: Callable) -> None:
        self.scaler = self._train_scaler(scaler)
        self.trajectories.scale_function = self.scale
        self.train_datasample = self._make_datasample(self.trajectories.train_scaled)
        logger.info("Generated %d tensor pairs for training", len(self.train_datasample),
                    group=DATASET)
        self.test_datasample = self._make_datasample(self.trajectories.test_scaled)
        logger.info("Generated %d tensor pairs for testing", len(self.test_datasample),
                    group=DATASET)
        self.val_datasample = self._make_datasample(self.trajectories.val_scaled)
        logger.info("Generated %d tensor pairs for validation", len(self.val_datasample),
                    group=DATASET)

    @property
    def train_dataloader(self):
        return DataLoader(self.train_datasample, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.config.num_workers,
                          pin_memory=self.config.pin_memory,
                          persistent_workers=self.config.persistent_workers)

    @property
    def test_dataloader(self):
        return DataLoader(self.test_datasample, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.config.num_workers,
                          pin_memory=self.config.pin_memory,
                          persistent_workers=self.config.persistent_workers)

    @property
    def val_dataloader(self):
        return DataLoader(self.val_datasample, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.config.num_workers,
                          pin_memory=self.config.pin_memory,
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

    def __getitem__(self, index):
        return self.val_datasample[index]

    def __len__(self):
        return len(self.val_datasample)

    def _init_from_config(self,
                          config: Union[Dict, None, str, Path, MotionDatasetConfig],
                          config_class: Type[MotionDatasetConfig]):
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
                return json.load(conf_file)
        return config

    def save_config(self, save_path: Path):
        logger.info("Saving config to %s", save_path, group=DATASET)
        with open(save_path, 'w', encoding="utf-8") as conf_file:
            logger.debug(self.config.dict(), group=DATASET)
            json.dump(self.config.dict(), conf_file, indent=4, sort_keys=True)

    def scale(self, l_array):
        T = l_array.shape
        l_array = l_array.reshape(l_array.shape[0], -1)
        l_array = torch.FloatTensor(self.scaler.transform(l_array))
        return l_array.reshape(T)

    def unscale(self, l_array):
        T = l_array.shape
        l_array = l_array.reshape(l_array.shape[0], -1)
        l_array = torch.FloatTensor(self.scaler.inverse_transform(l_array))
        return l_array.reshape(T)

    # scale all the trajectories (same scaling for all the data)
    def _train_scaler(self, other_scaler):
        # first, get all the data in a single tensor
        # scale according to all the data
        if other_scaler is None:
            logger.debug("Training scaler", group=DATASET)
            train_all = torch.zeros((1, self.num_points * self.num_dims))
            for trajectory in self.trajectories.train:
                train_all = torch.cat((train_all, trajectory.tensor.reshape(
                        trajectory.tensor.shape[0], -1)))
            scaler = StandardScaler()
            scaler.fit(train_all)
        else:
            scaler = other_scaler
        return scaler

    def _make_datasample(self, scaled_trajectory):
        logger.debug("Sampling trajectories into sample/truth tensor pairs", group=DATASET)
        sample = torch.FloatTensor([])   # shape(num_sample, seq_len, num_point, num_dim)
        truth = torch.FloatTensor([])
        logger.debug("Tensor pairs will be generated for a %s learning type",
                        self.config.learning_type,
                        group=DATASET)
        if self.config.learning_type == LearningTypes.SEQ2SEQ:
            tensor_pair_function = self._make_seq2seq_pairs
        elif self.config.learning_type == LearningTypes.AUTOREG:
            tensor_pair_function = self._make_autoreg_pairs
        else:
            raise NotImplementedError(f"Learning type {self.config.learning_type}"
                                        " is not implemented yet")
        for trajectory in scaled_trajectory:
            sample_trajectory, truth_trajectory = tensor_pair_function(trajectory)
            sample = torch.cat([sample, sample_trajectory], dim=0)
            truth = torch.cat([truth, truth_trajectory], dim=0)
        return MotionDataSamples(sample, truth)

    # This could use padding to get recognition from the first time-steps
    def _make_seq2seq_pairs(self, trajectory):
        if len(trajectory) < self.history_size + self.future_size:
            raise ValueError("Check that the intended history size and future size are compatible"
                             f" with the dataset. A trajectory of size {len(trajectory)} can't be"
                             f" split in samples of sizes {self.history_size}"
                             f" and {self.future_size}")
        sample = [trajectory[i:i + self.history_size]
                  for i in range(len(trajectory) - self.history_size - self.future_size + 1)]
        truth = [trajectory[i + self.history_size:i + self.history_size + self.future_size]
                 for i in range(len(trajectory) - self.history_size - self.future_size + 1)]
        # -- use the stack function to convert the list of 1D tensors
        # into a 2D tensor where each element of the list is now a row
        sample = torch.stack(sample)
        truth = torch.stack(truth)
        return sample, truth

    # This could use padding to get recognition from the first time-steps
    def _make_autoreg_pairs(self, trajectory):
        if len(trajectory) < self.history_size + 1:
            raise ValueError("Check that the intended history size and future size are compatible"
                             f" with the dataset. A trajectory of size {len(trajectory)} can't be"
                             f" split in samples of sizes {self.history_size} + 1")
        sample = [trajectory[i:i + self.history_size]
                  for i in range(len(trajectory) - self.history_size)]
        truth = [trajectory[i + 1:i + self.history_size + 1]
                 for i in range(len(trajectory) - self.history_size)]
        # -- use the stack function to convert the list of 1D tensors
        # into a 2D tensor where each element of the list is now a row
        sample = torch.stack(sample)
        truth = torch.stack(truth)
        return sample, truth

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
