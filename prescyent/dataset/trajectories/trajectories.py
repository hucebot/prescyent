"""Module for trajectories classes"""
import math
from typing import Dict, Iterable

import h5py
import numpy as np

from prescyent.dataset.hdf5_utils import load_features, get_dataset_keys
from prescyent.dataset.features import Features
from prescyent.dataset.trajectories.trajectory import Trajectory
from prescyent.dataset.trajectories.trajectory_hdf5 import TrajectoryHDF5


class Trajectories:
    """Trajectories are collections of n Trajectory, organized into train, val, test"""

    train: Iterable[Trajectory]
    test: Iterable[Trajectory]
    val: Iterable[Trajectory]

    def __init__(
        self,
        train: Iterable[Trajectory],
        test: Iterable[Trajectory],
        val: Iterable[Trajectory],
        # h_file: h5py.File = None,
    ) -> None:
        self.train = train
        self.test = test
        self.val = val
        # self.h_file = h_file
        self._check_trajectories_consistancy()

    @staticmethod
    def __init_from_hdf5__(file_path: str) -> None:
        """Init Trajectories from a hdf5 file following prescyent's format"""
        h_file = h5py.File(file_path, "r")
        frequency = h_file.attrs["frequency"]
        point_names = list(h_file.attrs["point_names"])
        point_parents = list(h_file.attrs["point_parents"])
        tensor_features = load_features(h_file)
        all_keys = get_dataset_keys(h_file)
        traj_keys = [key for key in all_keys if key[-5:] == "/traj"]
        train, test, val = [], [], []
        if "train" in h_file.keys():
            train = [
                TrajectoryHDF5(
                    frequency=frequency,
                    tensor_features=tensor_features,
                    file_path=h_file.filename,
                    title=traj_name,
                    point_parents=point_parents,
                    point_names=point_names,
                )
                for traj_name in traj_keys
                if traj_name[: len("train")] == "train"
            ]
        if "test" in h_file.keys():
            test = [
                TrajectoryHDF5(
                    frequency=frequency,
                    tensor_features=tensor_features,
                    file_path=h_file.filename,
                    title=traj_name,
                    point_parents=point_parents,
                    point_names=point_names,
                )
                for traj_name in traj_keys
                if traj_name[: len("test")] == "test"
            ]
        if "val" in h_file.keys():
            val = [
                TrajectoryHDF5(
                    frequency=frequency,
                    tensor_features=tensor_features,
                    file_path=h_file.filename,
                    title=traj_name,
                    point_parents=point_parents,
                    point_names=point_names,
                )
                for traj_name in traj_keys
                if traj_name[: len("val")] == "val"
            ]
        h_file.close()
        return Trajectories(train=train, test=test, val=val)

    def _check_trajectories_consistancy(self):
        _frequency = {t.frequency for t in self.train + self.test + self.val}
        try:
            assert len(_frequency) == 1
        except AssertionError as err:
            raise AssertionError(
                "We expect all Trajectory to have the same frequency"
            ) from err
        _num_points = {t.shape[1] for t in self.train + self.test + self.val}
        try:
            assert len(_num_points) == 1
        except AssertionError as err:
            raise AssertionError(
                "We expect all Trajectory to have the same num_points"
            ) from err

        _num_dims = {t.shape[2] for t in self.train + self.test + self.val}
        try:
            assert len(_num_dims) == 1
        except AssertionError as err:
            raise AssertionError(
                "We expect all Trajectory to have the same num_dims"
            ) from err

        _tensor_features = {
            t.tensor_features for t in self.train + self.test + self.val
        }
        try:
            assert len(_tensor_features) == 1
        except AssertionError as err:
            raise AssertionError(
                "We expect all Trajectory to have the same tensor_features"
            ) from err
        if all([t.context for t in self.train + self.test + self.val]):
            _context_size_sum = {
                sum([math.prod(c_tensor.shape[1:]) for c_tensor in t.context.values()])
                for t in self.train + self.test + self.val
            }
            try:
                assert len(_context_size_sum) == 1
            except AssertionError as err:
                raise AssertionError(
                    "When calling .context_size_sum on Trajectories level, we expect all Trajectory to have the same context_size_sum"
                ) from err

    def _all_len(self) -> int:
        return len(self.train) + len(self.test) + len(self.val)

    def __len__(self) -> int:
        return self._all_len()

    def __getitem__(self, index) -> Trajectory:
        return (self.train + self.test + self.val)[index]

    @property
    def frequency(self) -> int:
        for t in self.train + self.test + self.val:
            return t.frequency

    @property
    def num_points(self) -> int:
        for t in self.train + self.test + self.val:
            return t.shape[1]

    @property
    def num_dims(self) -> int:
        for t in self.train + self.test + self.val:
            return t.shape[2]

    @property
    def tensor_features(self) -> Features:
        for t in self.train + self.test + self.val:
            return t.tensor_features

    @property
    def context_sizes(self) -> Dict[str, int]:
        self.context_size_sum  # Just check that we not raise an assertion error
        if all([t.context for t in self.train + self.test + self.val]):
            for t in self.train + self.test + self.val:
                return {
                    c_key: c_tensor.shape[1:] for c_key, c_tensor in t.context.items()
                }
        return None

    @property
    def context_size_sum(self) -> int:
        if all([t.context for t in self.train + self.test + self.val]):
            for t in self.train + self.test + self.val:
                return sum(
                    [math.prod(c_tensor.shape[1:]) for c_tensor in t.context.values()]
                )
        return None
