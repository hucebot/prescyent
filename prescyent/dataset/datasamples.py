"""Data pair of sample and truth for motion data in ML"""
from typing import List

import torch

from prescyent.utils.enums import LearningTypes
from prescyent.dataset.trajectories import Trajectory


class MotionDataSamples:
    """Class storing x,y pairs for ML trainings on motion data"""

    trajectories: List[Trajectory]

    def __init__(
        self,
        trajectories: List[Trajectory],
        history_size: int,
        future_size: int,
        sampling_type: LearningTypes,
        in_dims: List[int] = None,
        out_dims: List[int] = None,
    ) -> None:
        self.trajectories = trajectories
        self.history_size = history_size
        self.future_size = future_size
        if in_dims is None:
            in_dims = list(range(trajectories[0].shape[2]))
        if out_dims is None:
            out_dims = list(range(trajectories[0].shape[2]))
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.sampling_type = sampling_type
        if (
            self.sampling_type != LearningTypes.SEQ2SEQ
            and self.sampling_type != LearningTypes.AUTOREG
            and self.sampling_type != LearningTypes.SEQ2ONE
        ):
            raise NotImplementedError(
                f"We don't handle {self.sampling_type} sampling for now."
            )
        self.sample_ids = self._map_to_flatten_trajs()

    def _map_to_flatten_trajs(self):
        _map = []
        if self.sampling_type in [LearningTypes.SEQ2SEQ, LearningTypes.SEQ2ONE]:
            invalid_frames_per_traj = self.history_size + self.future_size
        if self.sampling_type == LearningTypes.AUTOREG:
            invalid_frames_per_traj = self.history_size + 1
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
        sample = trajectory[tensor_id : tensor_id + self.history_size]
        truth = trajectory[
            tensor_id
            + self.history_size : tensor_id
            + self.history_size
            + self.future_size
        ]
        return sample, truth

    def _get_item_autoreg(self, index: int):
        traj_id, tensor_id = self.sample_ids[index]
        trajectory = self.trajectories[traj_id]
        sample = trajectory[tensor_id : tensor_id + self.history_size]
        truth = trajectory[tensor_id + 1 : tensor_id + self.history_size + 1]
        return sample, truth

    def _get_item_seq2one(self, index: int):
        traj_id, tensor_id = self.sample_ids[index]
        trajectory = self.trajectories[traj_id]
        sample = trajectory[tensor_id : tensor_id + self.history_size]
        truth = trajectory[tensor_id + self.history_size + self.future_size - 1]
        truth = torch.unsqueeze(truth, 0)
        return sample, truth

    def __getitem__(self, index: int):
        if self.sampling_type == LearningTypes.SEQ2SEQ:
            _in, _out = self._get_item_seq2seq(index)
        elif self.sampling_type == LearningTypes.AUTOREG:
            _in, _out = self._get_item_autoreg(index)
        elif self.sampling_type == LearningTypes.SEQ2ONE:
            _in, _out = self._get_item_seq2one(index)
        else:
            raise NotImplementedError(
                f"We don't handle {self.sampling_type} sampling for now."
            )
        return _in[:, :, self.in_dims], _out[:, :, self.out_dims]

    def __len__(self):
        return len(self.sample_ids)
