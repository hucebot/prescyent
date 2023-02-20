"""Standard class for motion datasets"""
import zipfile
from pathlib import Path

import requests
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from prescyent.dataset.config import LearningTypes, MotionDatasetConfig
from prescyent.dataset.trajectories import Trajectories
from prescyent.dataset.datasamples import MotionDataSamples


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

    def __init__(self, scaler) -> None:
        self.scaler = self._train_scaler(scaler)
        self.trajectories.scale_function = self.scale
        self.train_datasample = self._make_datasample(self.trajectories.train_scaled)
        self.test_datasample = self._make_datasample(self.trajectories.test_scaled)
        self.val_datasample = self._make_datasample(self.trajectories.val_scaled)

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

    def __getitem__(self, index):
        return self.val_datasample[index]

    def __len__(self):
        return len(self.val_datasample)

    def scale(self, l_array):
        return torch.FloatTensor(self.scaler.transform(l_array))

    def unscale(self, l_array):
        return torch.FloatTensor(self.scaler.inverse_transform(l_array))

    # scale all the trajectories (same scaling for all the data)
    def _train_scaler(self, other_scaler):
        # first, get all the data in a single tensor
        # scale according to all the data
        if other_scaler is None:
            train_all = torch.zeros((1, self.trajectories.train[0].shape[1]))
            for trajectory in self.trajectories.train:
                train_all = torch.cat((train_all, trajectory.tensor))    # useful for normalization
            scaler = StandardScaler()
            scaler.fit(train_all)
        else:
            scaler = other_scaler
        return scaler

    def _make_datasample(self, scaled_trajectory):
        sample = torch.FloatTensor([])   # shape(num_sample, seq_len, features)
        truth = torch.FloatTensor([])
        for trajectory in scaled_trajectory:
            if self.config.learning_type == LearningTypes.SEQ2SEQ:
                sample_trajectory, truth_trajectory = self._make_seq2seq_pairs(trajectory)
            elif self.config.learning_type == LearningTypes.AUTOREG:
                sample_trajectory, truth_trajectory = self._make_autoreg_pairs(trajectory)
            else:
                raise NotImplementedError(f"Learning type {self.config.learning_type}"
                                          " is not implemented yet")
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
        data = requests.get(url, timeout=10)
        path = Path(path)
        if path.is_dir():
            path = path / "downloaded_data.zip"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as pfile:
            pfile.write(data.content)

    def _unzip(self, zip_path: str):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(zip_path.replace(".zip", ""))
