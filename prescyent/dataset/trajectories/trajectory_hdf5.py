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
        h_file: h5py.File,
        title: str,
        point_parents: List[int],
        point_names: List[str],
    ) -> None:
        # remove any trailing /traj in traj's name, as we want the group's name
        self.title = re.sub(r"\/traj$", "", title)
        self.file_path = h_file.filename
        self.frequency = frequency
        self.tensor_features = tensor_features
        self.point_parents = point_parents
        self.point_names = point_names
        self.h_file = h_file
        self.context_keys = [
            key for key in self.h_file[self.title].keys() if key != "traj"
        ]

    @property
    def traj_group(self):
        return self.h_file[self.title]

    @property
    def tensor(self):
        return torch.from_numpy(np.array(self.traj_group["traj"]))

    @tensor.setter
    def tensor(self, value):
        # Create new dataset with the value, as shapes may vary
        del self.traj_group["traj"]
        self.traj_group.create_dataset("traj", data=value)

    @property
    def context(self):
        if self.context_keys:
            return {
                key: torch.from_numpy(np.array(self.traj_group[f"{key}"]))
                for key in self.context_keys
            }
        return None

    @context.setter
    def context(self, new_context):
        if new_context is None:
            self.context_keys = None
            return
        for key, value in new_context.keys():
            if key not in self.context_keys:
                self.traj_group.create_dataset(key, data=value)
                self.context_keys.append(key)
            else:
                del self.traj_group[
                    key
                ]  # Create new dataset with the values, as shapes may vary
                self.traj_group.create_dataset(key, data=value)
