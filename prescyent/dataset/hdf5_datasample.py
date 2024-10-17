"""Data pair of sample and truth for motion data in ML"""
import tempfile
from typing import Dict, List, Tuple, Union

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

from prescyent.dataset.config import TrajectoriesDatasetConfig
from prescyent.dataset.trajectories.trajectory import Trajectory
from prescyent.dataset.features import convert_tensor_features_to
from prescyent.utils.enums import LearningTypes
from prescyent.utils.logger import logger, DATASET


class HDF5TrajectoryDataSamples:
    """Class storing x,y pairs for ML trainings on trajectory data"""

    config: TrajectoriesDatasetConfig
    """The dataset configuration used to create the TrajectoryDataSamples"""
    tmp_hdf5: tempfile._TemporaryFileWrapper
    """generated hdf5 temporary file"""
    size: int
    """len of the class"""

    def __init__(
        self, trajectories: List[Trajectory], config: TrajectoriesDatasetConfig
    ) -> None:
        self.config = config
        if (
            config.learning_type != LearningTypes.SEQ2SEQ
            and config.learning_type != LearningTypes.AUTOREG
            and config.learning_type != LearningTypes.SEQ2ONE
        ):
            raise NotImplementedError(
                f"We don't handle {config.learning_type} sampling for now."
            )
        logger.getChild(DATASET).info(
            f"Generating new datapair from {len(trajectories)} trajectories on disk"
        )
        self.create_data_pairs_to_hdf5(trajectories)
        if self.config.reverse_pair_ratio:
            np.random.seed(seed=self.config.seed)

    def create_data_pairs_to_hdf5(self, trajectories: List[Trajectory]):
        """generate samples, context and truth into a new tmp hdf5 file from given list of Trajectory

        Args:
            trajectories (List[Trajectory]): trajectories used to generate the samples
        """
        if self.config.loop_over_traj:
            raise ValueError(
                "We cannot use 'loop_over_traj' along with 'save_samples_on_disk', please switch one off"
            )
        if self.config.learning_type in [LearningTypes.SEQ2SEQ, LearningTypes.SEQ2ONE]:
            frames_per_pair = self.config.history_size + self.config.future_size
        if self.config.learning_type == LearningTypes.AUTOREG:
            frames_per_pair = self.config.history_size + 1
        context = {key: [] for key in self.config.context_keys}
        valid_frames_per_traj = [
            len(traj.tensor) - frames_per_pair + 1 for traj in trajectories
        ]
        if any([True if t <= 0 else False for t in valid_frames_per_traj]):
            raise ValueError(
                "Check that history size and future size are compatible with the"
                f" dataset. Trajectory of size bellow {frames_per_pair} can't be split "
                f"in samples of sizes {frames_per_pair}"
            )
        num_pairs = sum(valid_frames_per_traj)
        self.tmp_hdf5 = tempfile.NamedTemporaryFile(suffix=".hdf5")
        tmp_hdf5_data = h5py.File(self.tmp_hdf5.name, "w")
        if not trajectories:
            samples = tmp_hdf5_data.create_dataset("samples", data=[])
            truths = tmp_hdf5_data.create_dataset("truths", data=[])
            context = {
                key: tmp_hdf5_data.create_dataset(key, data=[])
                for key in self.config.context_keys
            }
            tmp_hdf5_data.close()
            self.size = 0
            return
        samples = tmp_hdf5_data.create_dataset(
            "samples",
            shape=(
                num_pairs,
                self.config.history_size,
                self.config.num_in_points,
                self.config.num_in_dims,
            ),
            dtype=trajectories[0].tensor.numpy().dtype,
        )
        context = {
            key: tmp_hdf5_data.create_dataset(
                key,
                shape=(
                    num_pairs,
                    self.config.history_size,
                    *trajectories[0].context[key].shape[1:],
                ),
                dtype=trajectories[0].context[key].numpy().dtype,
            )
            for key in self.config.context_keys
        }
        if self.config.learning_type == LearningTypes.SEQ2SEQ:
            out_size = self.config.future_size
        elif self.config.learning_type == LearningTypes.SEQ2ONE:
            out_size = 1
        elif self.config.learning_type == LearningTypes.AUTOREG:
            out_size = self.config.history_size
        truths = tmp_hdf5_data.create_dataset(
            "truths",
            shape=(
                num_pairs,
                out_size,
                self.config.num_out_points,
                self.config.num_out_dims,
            ),
            dtype=trajectories[0].tensor.numpy().dtype,
        )
        i = 0
        for traj in tqdm(
            trajectories,
            desc="Iterating over trajectories to create data pairs",
            colour="blue",
        ):
            in_tensor = traj.tensor[:, self.config.in_points]
            in_tensor = convert_tensor_features_to(
                in_tensor, traj.tensor_features, self.config.in_features
            )
            out_tensor = traj.tensor[:, self.config.out_points]
            out_tensor = convert_tensor_features_to(
                out_tensor, traj.tensor_features, self.config.out_features
            )
            for j in tqdm(
                range(len(traj) - frames_per_pair + 1),
                desc="Iterating over frames",
                colour="green",
            ):
                samples[i] = in_tensor[j : j + self.config.history_size]
                for c_key in self.config.context_keys:
                    context[c_key][i] = traj.context[c_key][
                        j : j + self.config.history_size
                    ]
                if self.config.learning_type == LearningTypes.SEQ2SEQ:
                    start_id = j + self.config.history_size
                    end_id = start_id + out_size
                elif self.config.learning_type == LearningTypes.SEQ2ONE:
                    start_id = (
                        j + self.config.history_size + self.config.future_size - 1
                    )
                    end_id = start_id + out_size
                elif self.config.learning_type == LearningTypes.AUTOREG:
                    start_id = j + 1
                    end_id = start_id + out_size
                truths[i] = out_tensor[start_id:end_id]
                i += 1
        self.size = len(samples)
        tmp_hdf5_data.close()

    def __getitem__(
        self, index: Union[int, List[int]]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """iterate over generated samples

        Args:
            index (int): the sample id

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]: the input of the model, additional context, the truth to compare the model output with
        """
        if isinstance(index, int):
            index = [index]
        index = sorted(index)
        tmp_hdf5_data = h5py.File(self.tmp_hdf5.name, "r")
        samples = tmp_hdf5_data["samples"]
        _in = torch.FloatTensor(np.array(samples[index]))
        _in_context = {
            key: torch.FloatTensor(np.array(tmp_hdf5_data[key][index]))
            for key in self.config.context_keys
        }
        _out = torch.FloatTensor(np.array(tmp_hdf5_data["truths"][index]))
        tmp_hdf5_data.close()
        if (
            self.config.reverse_pair_ratio
            and np.random.uniform(0, 1) <= self.config.reverse_pair_ratio
        ):
            _in = torch.flip(_in, (0,))
            _in_context = {
                key: torch.flip(_in_context[key], (0,))
                for key in self.config.context_keys
            }
            _out = torch.flip(_out, (0,))
        return _in, _in_context, _out

    def __len__(self):
        return self.size
