"""Standard class for motion datasets"""
import zipfile
from pathlib import Path

import requests
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from prescyent.dataset.motion.config import MotionDatasetConfig
from prescyent.dataset.motion.trajectories import Trajectories
from prescyent.dataset.motion.datasamples import MotionDataSamples


class MotionDataset(Dataset):
    """Base classe for all motion datasets"""
    config: MotionDatasetConfig
    scaler: StandardScaler
    batch_size: int
    input_size: int
    output_size: int
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
            sample_trajectory, truth_trajectory = self._make_sample_truth_pairs(trajectory)
            sample = torch.cat([sample, sample_trajectory], dim=0)
            truth = torch.cat([truth, truth_trajectory], dim=0)
        return MotionDataSamples(sample, truth)

    # This could use padding to get recognition from the first time-steps
    def _make_sample_truth_pairs(self, trajectory):
        sample = [trajectory[i:i + self.input_size]
                  for i in range(len(trajectory) - self.input_size - self.output_size)]
        truth = [trajectory[i + self.input_size:i + self.input_size + self.output_size]
                 for i in range(len(trajectory) - self.input_size - self.output_size)]
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
