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


class MotionDataset(Dataset):
    scaler: StandardScaler
    batch_size: int
    input_length: int
    output_length: int
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
        return DataLoader(self.train_datasample, batch_size=self.batch_size)

    @property
    def test_dataloader(self):
        return DataLoader(self.test_datasample, batch_size=self.batch_size)

    @property
    def val_dataloader(self):
        return DataLoader(self.val_datasample, batch_size=self.batch_size)


    def __getitem__(self, index):
        return self.episodes_scaled[index]

    def __len__(self):
        return len(self.episodes_scaled)


    def scale(self, l_array):
        return self.scaler.transform(l_array.reshape(-1, 1)).flatten()

    def unscale(self, l_array):
        return self.scaler.inverse_transform(l_array.reshape(-1, 1)).flatten()

    # scale all the episodes (same scaling for all the data)
    def _scale_episodes(self, other_scaler):
        # first, get all the data in a single tensor
        # scale according to all the data
        if other_scaler is None:
            train_all = np.array([])
            for episode in self.episodes.train:
                train_all = np.concatenate((train_all, episode))    # useful for normalization
            scaler = StandardScaler()
            scaler.fit(train_all.reshape(-1, 1))
        else:
            scaler = other_scaler

        # scale each episode
        # episodes_scaled = []
        # for l_array in self.episodes:
        #     scaled = scaler.transform(l_array.reshape(-1, 1))
        #     episodes_scaled += [torch.FloatTensor(scaled).view(-1)]
        train_data = [torch.FloatTensor(scaler.transform(l_array.reshape(-1, 1))).view(-1)
                      for l_array in self.episodes.train]
        test_data = [torch.FloatTensor(scaler.transform(l_array.reshape(-1, 1))).view(-1)
                     for l_array in self.episodes.test]
        val_data = [torch.FloatTensor(scaler.transform(l_array.reshape(-1, 1))).view(-1)
                    for l_array in self.episodes.val]
        return Episodes(train_data, test_data, val_data), scaler

    def _make_datasample(self, scaled_episode):
        x = torch.FloatTensor([])
        y = torch.FloatTensor([])
        for ep in scaled_episode:
            x_ep, y_ep = self._make_x_y_pairs(ep)
            x = torch.cat([x, x_ep], dim=0)
            y = torch.cat([y, y_ep], dim=0)
        return MotionDataSamples(x, y)

    # This could use padding to get recognition from the first time-steps
    def _make_x_y_pairs(self, ep):
        x = [ep[i:i + self.input_length]
             for i in range(len(ep) - self.input_length - self.output_length)]
        y = [ep[i + self.input_length:i + self.input_length + self.output_length]
             for i in range(len(ep) - self.input_length - self.output_length)]
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
