"""generation of samples, context and truth for trajectory data usage in ML"""
from typing import Dict, List, Tuple

import numpy as np
import torch

from prescyent.dataset.config import TrajectoriesDatasetConfig
from prescyent.dataset.trajectories.trajectory import Trajectory
from prescyent.dataset.features import convert_tensor_features_to
from prescyent.utils.enums import LearningTypes


class TrajectoryDataSamples:
    """Class generating (sample, context, truth) from a list of Trajectory for ML trainings on traj data"""

    trajectories: List[Trajectory]
    """List of the trajectories that are sampled"""
    config: TrajectoriesDatasetConfig
    """The dataset configuration used to create the TrajectoryDataSamples"""
    sample_ids: List[Tuple[int, int]]
    """The list of ids used to iterate over the trajectories"""

    def __init__(
        self, trajectories: List[Trajectory], config: TrajectoriesDatasetConfig
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
        if self.config.reverse_pair_ratio:
            np.random.seed(seed=self.config.seed)

    def _map_to_flatten_trajs(self):
        """creates a map to frames from all trajectories if they are valid as a starting frame for the sampling"""
        _map = []
        if self.config.learning_type in [LearningTypes.SEQ2SEQ, LearningTypes.SEQ2ONE]:
            invalid_frames_per_traj = (
                self.config.history_size + self.config.future_size - 1
            )
        if self.config.learning_type == LearningTypes.AUTOREG:
            invalid_frames_per_traj = self.config.history_size
        if self.config.loop_over_traj:
            invalid_frames_per_traj = 0
        for t, trajectory in enumerate(self.trajectories):
            if len(trajectory) < invalid_frames_per_traj:
                raise ValueError(
                    "Check that history size and future size are compatible with the"
                    f" dataset. Trajectory of size {len(trajectory)} can't be split "
                    f"in samples of sizes {invalid_frames_per_traj}"
                )
            _map += [(t, i) for i in range(len(trajectory) - invalid_frames_per_traj)]
        return _map

    def _get_seq_with_size(
        self, index: int, history_size: int, future_size: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """returns tensor frames and context according to starting id, history_size and future_size

        Args:
            index (int): strating frame id
            history_size (int): number of frames as history
            future_size (int): number of frames as future

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: frames tensor and context
        """
        size = history_size + future_size
        traj_id, tensor_id = self.sample_ids[index]
        trajectory = self.trajectories[traj_id]
        context = {}
        if tensor_id + size > len(trajectory.tensor) and self.config.loop_over_traj:
            seq = torch.cat(
                (
                    trajectory.tensor[tensor_id:],
                    trajectory.tensor[: size - len(trajectory.tensor[tensor_id:])],
                ),
                0,
            )
            if trajectory.context and self.config.context_keys:
                context = {
                    c_name: torch.cat(
                        (
                            c_tensor[tensor_id:],
                            c_tensor[: size - len(c_tensor[tensor_id:])],
                        ),
                        0,
                    )[:history_size]
                    for c_name, c_tensor in trajectory.context.items()
                    if c_name in self.config.context_keys
                }
        else:
            seq = trajectory.tensor[tensor_id : tensor_id + size]
            if trajectory.context and self.config.context_keys:
                context = {
                    c_name: c_tensor[tensor_id : tensor_id + history_size]
                    for c_name, c_tensor in trajectory.context.items()
                    if c_name in self.config.context_keys
                }
        if (
            self.config.reverse_pair_ratio
            and np.random.uniform(0, 1) <= self.config.reverse_pair_ratio
        ):
            seq = torch.flip(seq, (0,))
        return seq, context

    def _get_item_seq2seq(
        self, index: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """get seq2seq tensors

        Args:
            index (int): the sample id

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]: the input of the model, additional context, the truth to compare the model output with
        """
        seq, context = self._get_seq_with_size(
            index, self.config.history_size, self.config.future_size
        )
        sample = seq[: self.config.history_size]
        truth = seq[self.config.history_size :]
        return sample, context, truth

    def _get_item_autoreg(
        self, index: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """get autoregressive tensors

        Args:
            index (int): the sample id

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]: the input of the model, additional context, the truth to compare the model output with
        """
        seq, context = self._get_seq_with_size(index, self.config.history_size, 1)
        sample = seq[:-1]
        truth = seq[1:]
        return sample, context, truth

    def _get_item_seq2one(
        self, index: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """get seq2one tensors

        Args:
            index (int): the sample id

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]: the input of the model, additional context, the truth to compare the model output with
        """
        seq, context = self._get_seq_with_size(
            index, self.config.history_size, self.config.future_size
        )
        sample = seq[: self.config.history_size]
        truth = torch.unsqueeze(seq[-1], 0)
        return sample, context, truth

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """iterate over generated samples

        Args:
            index (int): the sample id

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]: the input of the model, additional context, the truth to compare the model output with
        """
        if self.config.learning_type == LearningTypes.SEQ2SEQ:
            _in, _in_context, _out = self._get_item_seq2seq(index)
        elif self.config.learning_type == LearningTypes.AUTOREG:
            _in, _in_context, _out = self._get_item_autoreg(index)
        elif self.config.learning_type == LearningTypes.SEQ2ONE:
            _in, _in_context, _out = self._get_item_seq2one(index)
        else:
            raise NotImplementedError(
                f"We don't handle {self.config.learning_type} sampling for now."
            )
        tensor_feats = self.trajectories[self.sample_ids[index][0]].tensor_features
        _in = _in[:, self.config.in_points]
        _in = convert_tensor_features_to(_in, tensor_feats, (self.config.in_features))
        _out = _out[:, self.config.out_points]
        _out = convert_tensor_features_to(
            _out, tensor_feats, (self.config.out_features)
        )
        return _in, _in_context, _out

    def __len__(self):
        return len(self.sample_ids)
