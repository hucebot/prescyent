import re
from typing import List

import h5py
import numpy as np
import torch

from prescyent.dataset.trajectories.trajectory import Trajectory
from prescyent.dataset.features.features import Features


class TrajectoryHDF5(Trajectory):
    """Child class of Trajectory were tensor and context are read from hdf5 file when called"""

    def __init__(
        self,
        frequency: float,
        tensor_features: Features,
        file_path: str,
        title: str,
        point_parents: List[int],
        point_names: List[str],
    ) -> None:
        # remove any trailing /traj in traj's name, as we want the group's name
        self.title = re.sub(r"\/traj$", "", title)
        self.file_path = file_path
        self.frequency = frequency
        self.tensor_features = tensor_features
        self.point_parents = point_parents
        self.point_names = point_names
        h_file = h5py.File(self.file_path, "r")
        self.context_keys = [key for key in h_file[self.title].keys() if key != "traj"]

    @property
    def traj_group(self):
        h_file = h5py.File(self.file_path, "r")
        return h_file[self.title]

    @property
    def w_traj_group(self):
        h_file = h5py.File(self.file_path, "r+")
        return h_file[self.title]

    @property
    def tensor(self):
        with h5py.File(self.file_path, "r") as h_file:
            traj_group = h_file[self.title]
            return torch.FloatTensor(np.array(traj_group["traj"]))

    @tensor.setter
    def tensor(self, value):
        # Create new dataset with the value, as shapes may vary
        with h5py.File(self.file_path, "r+") as h_file:
            w_traj_group = h_file[self.title]
            del w_traj_group["traj"]
            w_traj_group.create_dataset("traj", data=value)

    @property
    def context(self):
        with h5py.File(self.file_path, "r") as h_file:
            traj_group = h_file[self.title]
            if self.context_keys:
                return {
                    key: torch.FloatTensor(np.array(traj_group[f"{key}"]))
                    for key in self.context_keys
                }
            return {}

    @context.setter
    def context(self, new_context):
        with h5py.File(self.file_path, "r+") as h_file:
            w_traj_group = h_file[self.title]
            if new_context is None:
                self.context_keys = None
                return
            for key, value in new_context.keys():
                if key not in self.context_keys:
                    w_traj_group.create_dataset(key, data=value)
                    self.context_keys.append(key)
                else:
                    del w_traj_group[
                        key
                    ]  # Create new dataset with the values, as shapes may vary
                    w_traj_group.create_dataset(key, data=value)
