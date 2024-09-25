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

from prescyent.dataset.hdf5_utils import get_dataset_keys, write_metadata
from prescyent.dataset.dataset import MotionDataset
from prescyent.dataset.trajectories import Trajectories
from prescyent.utils.logger import logger, DATASET
from prescyent.utils.dataset_manipulation import split_array_with_ratios

from . import metadata
from .config import DatasetConfig


class Dataset(MotionDataset):
    """
    https://zenodo.org/record/5913573#.Y75xK_7MIaw
    Dataset is not splitted into test / train / val
    It as to be at initialisation, through the parameters
    """

    DATASET_NAME = "TeleopIcub"

    def __init__(self, config=DatasetConfig(), load_data_at_init: bool = True) -> None:
        self._init_from_config(config, DatasetConfig)
        super().__init__(name=self.DATASET_NAME, load_data_at_init=load_data_at_init)

    def prepare_data(self):
        """get trajectories from files or web"""
        # Download and prepare hdf5 ??
        if not Path(self.config.hdf5_path).exists():
            logger.getChild(DATASET).warning(
                "Dataset files not found at path %s",
                self.config.hdf5_path,
            )
            self._get_from_web()
        # Create new temp hdf5 with features from config
        self.tmp_hdf5 = tempfile.NamedTemporaryFile(suffix=".hdf5")
        hdf5_data = h5py.File(self.config.hdf5_path, "r")
        tmp_hdf5_data = h5py.File(self.tmp_hdf5.name, "w")
        # Copy root attributes
        for attr in hdf5_data.attrs.keys():
            tmp_hdf5_data.attrs[attr] = hdf5_data.attrs[attr]
        all_keys = get_dataset_keys(hdf5_data)
        # Features
        all_feature_names = [key for key in all_keys if key[:16] == "tensor_features/"]
        for feat_name in all_feature_names:
            old_feat = hdf5_data[feat_name]
            feat = tmp_hdf5_data.create_dataset(feat_name, data=old_feat)
            for attr_name in old_feat.attrs.keys():
                feat.attrs[attr_name] = old_feat.attrs[attr_name]
        # Select only trajs from subset:
        all_trajectory_names = [key for key in all_keys if key[-5:] == "/traj"]
        if self.config.subsets:
            all_trajectory_names = [
                key
                for key in all_trajectory_names
                if any([subset in key for subset in self.config.subsets])
            ]
        # create subset train, test, val from ratios:
        train_trajs, test_trajs, val_trajs = split_array_with_ratios(
            all_trajectory_names,
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
            for traj_name in trajs:
                tmp_hdf5_data.create_dataset(key + traj_name, data=hdf5_data[traj_name])
                if self.config.context_keys:
                    for context_name in self.config.context_keys:
                        tmp_hdf5_data.create_dataset(
                            key + traj_name[:-4] + context_name,
                            data=hdf5_data[traj_name[:-4] + context_name],
                        )
        self.trajectories = Trajectories.__init_from_hdf5__(self.tmp_hdf5.name)

    def _get_from_web(self) -> None:
        zip_path = Path(self.config.hdf5_path).with_suffix(".zip")
        self._download_files(self.config.url, zip_path)
        data_dir = self._unzip(zip_path)
        data_dir = data_dir / "AndyData-lab-prescientTeleopICub"
        Dataset.create_teleop_icub_hdf5(self.config.hdf5_path, data_dir, "*.csv", True)

    @staticmethod
    def create_teleop_icub_hdf5(
        hdf5_path: str, data_dir: str, subsets: str, remove_csv: bool = False
    ):
        files = list(Path(data_dir).rglob(subsets))
        logger.getChild(DATASET).info(f"Found {len(files)} files")
        with h5py.File(hdf5_path, "w") as hdf5_f:
            write_metadata(hdf5_f, metadata)

        for f_path in tqdm(files):
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
                group.create_dataset(key, data=tensor)
        logger.getChild(DATASET).info(f"Created new HDF5 at {hdf5_path}")
        if remove_csv:
            logger.getChild(DATASET).info(f"Removing all files in {data_dir}")
            shutil.rmtree(data_dir)
