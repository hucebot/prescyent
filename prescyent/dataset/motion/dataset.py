"""Standard class for motion datasets"""
import requests
import zipfile
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from prescyent.dataset.motion.episodes import Episodes
from prescyent.dataset.motion.datasamples import MotionDataSamples


def _scale_episodes(episode_array, scaler):
    res_array = list()
    for episode in episode_array:
        scaled_tensor = torch.FloatTensor(scaler.transform(episode))
        res_array.append(scaled_tensor)
    return res_array


class MotionDataset(Dataset):
    scaler: StandardScaler
    batch_size: int
    input_size: int
    output_size: int
    episodes: Episodes
    episodes_scaled: Episodes
    train_datasample: MotionDataSamples
    test_datasample: MotionDataSamples
    val_datasample: MotionDataSamples

    def __init__(self, scaler) -> None:
        self.episodes_scaled, self.scaler = self._scale_episodes(scaler)
        self.train_datasample = self._make_datasample(self.episodes_scaled.train)
        self.test_datasample = self._make_datasample(self.episodes_scaled.test)
        self.val_datasample = self._make_datasample(self.episodes_scaled.val)

    @property
    def train_dataloader(self):
        return DataLoader(self.train_datasample, batch_size=self.batch_size, shuffle=True)

    @property
    def test_dataloader(self):
        return DataLoader(self.test_datasample, batch_size=self.batch_size, shuffle=True)

    @property
    def val_dataloader(self):
        return DataLoader(self.val_datasample, batch_size=self.batch_size, shuffle=True)

    def __getitem__(self, index):
        return self.episodes_scaled[index]

    def __len__(self):
        return len(self.episodes_scaled)

    def scale(self, l_array):
        return self.scaler.transform(l_array)

    def unscale(self, l_array):
        return self.scaler.inverse_transform(l_array)

    # scale all the episodes (same scaling for all the data)
    def _scale_episodes(self, other_scaler):
        # first, get all the data in a single tensor
        # scale according to all the data
        if other_scaler is None:
            train_all = np.empty((1, self.episodes.train[0].shape[1]))
            for episode in self.episodes.train:
                train_all = np.concatenate((train_all, episode))    # useful for normalization
            scaler = StandardScaler()
            scaler.fit(train_all)
        else:
            scaler = other_scaler

        # scale each episode of each subset
        train_data = [torch.FloatTensor(scaler.transform(episode)) for episode in self.episodes.train]
        test_data = [torch.FloatTensor(scaler.transform(episode)) for episode in self.episodes.test]
        val_data = [torch.FloatTensor(scaler.transform(episode)) for episode in self.episodes.val]
        return Episodes(train_data, test_data, val_data), scaler

    def _make_datasample(self, scaled_episode):
        x = torch.FloatTensor([])   # shape(num_sample, seq_len, features)
        y = torch.FloatTensor([])
        for ep in scaled_episode:
            x_ep, y_ep = self._make_x_y_pairs(ep)
            x = torch.cat([x, x_ep], dim=0)
            y = torch.cat([y, y_ep], dim=0)
        return MotionDataSamples(x, y)

    # This could use padding to get recognition from the first time-steps
    def _make_x_y_pairs(self, ep):
        x = [ep[i:i + self.input_size]
             for i in range(len(ep) - self.input_size - self.output_size)]
        y = [ep[i + self.input_size:i + self.input_size + self.output_size]
             for i in range(len(ep) - self.input_size - self.output_size)]
        # -- use the stack function to convert the list of 1D tensors
        # into a 2D tensor where each element of the list is now a row
        x = torch.stack(x)
        y = torch.stack(y)
        return x, y

    def _download_files(self, url, path):
        """get the dataset files from an url"""
        data = requests.get(url)
        p = Path(path)
        if p.is_dir():
            p = p / "downloaded_data.zip"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as pfile:
            pfile.write(data.content)

    def _unzip(self, zip_path: str):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(zip_path.replace(".zip", ""))
