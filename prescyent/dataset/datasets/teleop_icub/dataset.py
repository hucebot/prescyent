"""Class and methods for the TeleopIcub Dataset
https://zenodo.org/record/5913573#.Y75xK_7MIaw
"""

import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from prescyent.dataset.hdf5_utils import write_metadata
from prescyent.dataset.dataset import TrajectoriesDataset
from prescyent.dataset.trajectories import Trajectories
from prescyent.utils.dataset_manipulation import split_array_with_ratios
from prescyent.utils.interpolate import update_tensor_frequency
from prescyent.utils.logger import logger, DATASET

from . import metadata
from .config import TeleopIcubDatasetConfig


class TeleopIcubDataset(TrajectoriesDataset):
    """
    https://zenodo.org/record/5913573#.Y75xK_7MIaw
    Dataset is not splitted into test / train / val
    It as to be at initialisation, through the parameters
    """

    DATASET_NAME = "TeleopIcub"

    def __init__(self, config) -> None:
        self._init_from_config(config, TeleopIcubDatasetConfig)
        super().__init__(name=self.DATASET_NAME)

    def prepare_data(self):
        """get trajectories from files or web"""
        if hasattr(self, "_trajectories"):
            return
        if not Path(self.config.hdf5_path).exists():
            raise FileNotFoundError(
                "Dataset file not found at %s" % self.config.hdf5_path
            )
        self.tmp_hdf5 = tempfile.NamedTemporaryFile(suffix=".hdf5")
        hdf5_data = h5py.File(self.config.hdf5_path, "r")
        tmp_hdf5_data = h5py.File(self.tmp_hdf5.name, "w")
        trajectory_names = self.get_trajnames_from_hdf5(hdf5_data)
        self.copy_attributes_from_hdf5(hdf5_data, tmp_hdf5_data)
        if self.config.subsets:
            trajectory_names = [
                key
                for key in trajectory_names
                if any([subset in key for subset in self.config.subsets])
            ]
        # create subset train, test, val from ratios:
        train_trajs, test_trajs, val_trajs = split_array_with_ratios(
            trajectory_names,
            self.config.ratio_train,
            self.config.ratio_test,
            self.config.ratio_val,
            shuffle=self.config.shuffle_data_files,
        )
        for key, trajs in {
            "train/": train_trajs,
            "test/": test_trajs,
            "val/": val_trajs,
        }.items():
            for traj_name in tqdm(
                trajs, colour="blue", desc="Writing used trajectories in temp file hdf5"
            ):
                tensor = torch.FloatTensor(np.array(hdf5_data[traj_name]))
                context = {
                    key: torch.FloatTensor(np.array(hdf5_data[traj_name[:-4] + key]))
                    for key in self.config.context_keys
                }
                # update frequency
                tensor, context = update_tensor_frequency(
                    tensor,
                    metadata.BASE_FREQUENCY,
                    self.config.frequency,
                    metadata.DEFAULT_FEATURES,
                    context,
                )
                tmp_hdf5_data.create_dataset(key + traj_name, data=tensor)
                if self.config.context_keys:
                    for context_name, context_tensor in context.items():
                        tmp_hdf5_data.create_dataset(
                            key + traj_name[:-4] + context_name,
                            data=context_tensor,
                        )
        tmp_hdf5_data.attrs["frequency"] = self.config.frequency
        self.trajectories = Trajectories.__init_from_hdf5__(self.tmp_hdf5.name)
        tmp_hdf5_data.close()
        hdf5_data.close()

    @staticmethod
    def create_hdf5(
        hdf5_path: str,
        data_dir: str,
        glob_pattern: str,
        remove_orignal_files: bool = False,
        compression="gzip",
    ):
        """script to generate the hdf5 file from the original dataset's files

        Args:
            hdf5_path (str): path where to save the hdf5 file
            data_dir (str): dir of the original data
            glob_pattern (str): pattern used to retreive the list of files
            remove_orignal_files (bool, optional): if true, we delete the original files when the function is done. Defaults to False.
            compression (str, optional): compression used writing the hdf5 file. Defaults to "gzip".
        """
        files = list(Path(data_dir).rglob(glob_pattern))
        logger.getChild(DATASET).info(f"Found {len(files)} files")
        Path(hdf5_path).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(hdf5_path, "w") as hdf5_f:
            write_metadata(
                hdf5_f,
                metadata.BASE_FREQUENCY,
                metadata.POINT_PARENTS,
                metadata.POINT_LABELS,
                metadata.DEFAULT_FEATURES,
            )
        for f_path in tqdm(
            files, colour="blue", desc="Iterating through dataset files"
        ):
            file_sequence = np.loadtxt(f_path, delimiter=",")
            subgroups = list(f_path.relative_to(data_dir).parts)
            traj_groups = subgroups[:-1]
            file_type = subgroups[-1]
            if file_type[0] == "c":  # center of mass
                traj_groups = traj_groups + [subgroups[-1][1:-4]]
                tensor = file_sequence
                key = "center_of_mass"
            elif file_type[0] == "p":  # positions
                traj_groups = subgroups[:-1] + [subgroups[-1][1:-4]]
                tensor = torch.FloatTensor(
                    np.array(
                        [
                            np.concatenate((np.zeros(2), file_timestep))
                            for file_timestep in file_sequence
                        ]
                    )
                )
                seq_len = tensor.shape[0]
                tensor = tensor.reshape(seq_len, 3, 3)
                key = "traj"
                # reshape (seq_len, 9) => (seq_len, 3, 3)
            else:  # icub dofs
                traj_groups = subgroups[:-1] + [subgroups[-1][:-4]]
                tensor = file_sequence
                key = "icub_dof"
            with h5py.File(hdf5_path, "a") as f:
                try:
                    group = f.create_group("/".join(traj_groups))
                except ValueError:  # Group already exist
                    group = f["/".join(traj_groups)]
                group.create_dataset(key, data=tensor, compression=compression)
        logger.getChild(DATASET).info(f"Created new HDF5 at {hdf5_path}")
        if remove_orignal_files:
            logger.getChild(DATASET).info(f"Removing all files in {data_dir}")
            shutil.rmtree(data_dir)
