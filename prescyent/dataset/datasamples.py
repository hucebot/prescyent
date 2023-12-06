"""Data pair of sample and truth for motion data in ML"""
from typing import List

import torch

from prescyent.utils.enums import LearningTypes
from prescyent.dataset.trajectories import Trajectory
from prescyent.dataset.config import MotionDatasetConfig


class MotionDataSamples:
    """Class storing x,y pairs for ML trainings on motion data"""

    trajectories: List[Trajectory]

    def __init__(
        self, trajectories: List[Trajectory], config: MotionDatasetConfig
    ) -> None:
        self.trajectories = trajectories
        self.config = config
        if (
            config.learning_type != LearningTypes.SEQ2SEQ
            and config.learning_type != LearningTypes.AUTOREG
            and config.learning_type != LearningTypes.SEQ2ONE
        ):
            raise NotImplementedError(
                f"We don't handle {config.learning_type} sampling for now."
            )
        self.sample_ids = self._map_to_flatten_trajs()

    def _map_to_flatten_trajs(self):
        _map = []
        if self.config.learning_type in [LearningTypes.SEQ2SEQ, LearningTypes.SEQ2ONE]:
            invalid_frames_per_traj = self.config.history_size + self.config.future_size
        if self.config.learning_type == LearningTypes.AUTOREG:
            invalid_frames_per_traj = self.config.history_size + 1
        for t, trajectory in enumerate(self.trajectories):
            if len(trajectory) < invalid_frames_per_traj:
                raise ValueError(
                    "Check that history size and future size are compatible with the"
                    f" dataset. Trajectory of size {len(trajectory)} can't be split "
                    f"in samples of sizes {invalid_frames_per_traj}"
                )
            _map += [
                (t, i) for i in range(len(trajectory) - invalid_frames_per_traj + 1)
            ]
        return _map

    def _get_item_seq2seq(self, index: int):
        traj_id, tensor_id = self.sample_ids[index]
        trajectory = self.trajectories[traj_id]
        sample = trajectory[tensor_id : tensor_id + self.config.history_size]
        truth = trajectory[
            tensor_id
            + self.config.history_size : tensor_id
            + self.config.history_size
            + self.config.future_size
        ]
        return sample, truth

    def _get_item_autoreg(self, index: int):
        traj_id, tensor_id = self.sample_ids[index]
        trajectory = self.trajectories[traj_id]
        sample = trajectory[tensor_id : tensor_id + self.config.history_size]
        truth = trajectory[tensor_id + 1 : tensor_id + self.config.history_size + 1]
        return sample, truth

    def _get_item_seq2one(self, index: int):
        traj_id, tensor_id = self.sample_ids[index]
        trajectory = self.trajectories[traj_id]
        sample = trajectory[tensor_id : tensor_id + self.config.history_size]
        truth = trajectory[
            tensor_id + self.config.history_size + self.config.future_size - 1
        ]
        truth = torch.unsqueeze(truth, 0)
        return sample, truth

    def __getitem__(self, index: int):
        if self.config.learning_type == LearningTypes.SEQ2SEQ:
            _in, _out = self._get_item_seq2seq(index)
        elif self.config.learning_type == LearningTypes.AUTOREG:
            _in, _out = self._get_item_autoreg(index)
        elif self.config.learning_type == LearningTypes.SEQ2ONE:
            _in, _out = self._get_item_seq2one(index)
        else:
            raise NotImplementedError(
                f"We don't handle {self.config.learning_type} sampling for now."
            )
        _in = _in[:, self.config.in_points]
        _in = _in[:, :, self.config.in_dims]
        _out = _out[:, self.config.out_points]
        _out = _out[:, :, self.config.out_dims]
        return _in, _out

    def __len__(self):
        return len(self.sample_ids)
