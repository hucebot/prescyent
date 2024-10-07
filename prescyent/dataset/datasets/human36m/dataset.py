"""Class and methods for the Human 3.6M Dataset"""
from pathlib import Path
import shutil
import tempfile
from typing import List, Literal, Union, Dict

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

from prescyent.dataset.hdf5_utils import write_metadata
from prescyent.utils.interpolate import update_tensor_frequency
from prescyent.utils.logger import logger, DATASET
from prescyent.dataset.dataset import MotionDataset
from prescyent.dataset.trajectories.trajectories import Trajectories
from prescyent.dataset.trajectories.trajectory import Trajectory
from prescyent.utils.dataset_manipulation import (
    expmap2rotmat_torch,
    rotmat2xyz_torch,
    update_parent_ids,
)
import prescyent.dataset.datasets.human36m.metadata as metadata
from prescyent.dataset.datasets.human36m.config import DatasetConfig


class Dataset(MotionDataset):
    """Class for data loading et preparation before the MotionDataset sampling"""

    DATASET_NAME = "H36M"

    def __init__(
        self,
        config: Union[Dict, DatasetConfig] = None,
        config_class=DatasetConfig,
        load_data_at_init: bool = True,
    ) -> None:
        self._init_from_config(config, config_class)
        super().__init__(name=self.DATASET_NAME, load_data_at_init=load_data_at_init)

    def prepare_data(self):
        """get trajectories from files or web"""
        if not Path(self.config.hdf5_path).exists():
            raise FileNotFoundError(
                "Dataset file not found at %s" % self.config.hdf5_path
            )
        self.tmp_hdf5 = tempfile.NamedTemporaryFile(suffix=".hdf5")
        hdf5_data = h5py.File(self.config.hdf5_path, "r")
        tmp_hdf5_data = h5py.File(self.tmp_hdf5.name, "w")
        trajectory_names = self.get_trajnames_from_hdf5(hdf5_data, tmp_hdf5_data)
        # keep only given actions
        if self.config.actions:
            trajectory_names = [
                key
                for key in trajectory_names
                if any([subset in key for subset in self.config.actions])
            ]
        # create subset train, test, val from ratios:
        train_trajs = [
            t for t in trajectory_names if t.split("/")[0] in self.config.subjects_train
        ]
        test_trajs = [
            t for t in trajectory_names if t.split("/")[0] in self.config.subjects_test
        ]
        val_trajs = [
            t for t in trajectory_names if t.split("/")[0] in self.config.subjects_val
        ]
        for key, trajs in {
            "train/": train_trajs,
            "test/": test_trajs,
            "val/": val_trajs,
        }.items():
            for traj_name in tqdm(
                trajs, colour="blue", desc="Writing used trajectories in temp file hdf5"
            ):
                tensor = torch.from_numpy(np.array(hdf5_data[traj_name]))
                context = {}
                # update frequency
                tensor, context = update_tensor_frequency(
                    tensor,
                    metadata.BASE_FREQUENCY,
                    self.config.frequency,
                    metadata.DEFAULT_FEATURES,
                    context,
                )
                tmp_hdf5_data.create_dataset(
                    key + traj_name,
                    tensor.shape,
                    data=tensor,
                )
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
        file_patern: str,
        remove_csv: bool = False,
        compression="gzip",
    ):
        files = list(Path(data_dir).rglob(file_patern))
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
        for f_path in tqdm(files, colour="blue", desc="Iterating txt files"):
            traj_groups = list(f_path.relative_to(data_dir).parts)
            with f_path.open() as file:
                expmap = file.readlines()
            pose_info = []
            for line in expmap:
                line = line.strip().split(",")
                if len(line) > 0:
                    pose_info.append(np.array([float(x) for x in line]))
            # get expmap from file
            pose_info = np.array(pose_info)
            S = pose_info.shape[0]
            pose_info = pose_info.reshape(-1, 33, 3)
            pose_info[:, :2] = 0
            pose_info = pose_info[:, 1:, :].reshape(-1, 3)
            # get rotmatrices from expmap
            rotmatrices = expmap2rotmat_torch(
                torch.from_numpy(pose_info).float()
            ).reshape(S, 32, 3, 3)
            # get xyz and world relative rotmatrice from joint rotmatrices and bone/parent infos
            xyz_info, world_rotmatrices = rotmat2xyz_torch(
                rotmatrices.clone().detach(), metadata._get_metadata
            )
            world_rotmatrices = world_rotmatrices.reshape(S, 32, 9)
            xyz_info = xyz_info / 1000  # mm to meter conversion
            position_traj_tensor = torch.cat((xyz_info, world_rotmatrices), dim=-1)
            traj_groups = traj_groups[:-1] + [traj_groups[-1][:-4]]
            with h5py.File(hdf5_path, "a") as f:
                group = f.create_group("/".join(traj_groups))
                group.create_dataset(
                    "traj",
                    data=position_traj_tensor,
                    compression=compression,
                )
        logger.getChild(DATASET).info(f"Created new HDF5 at {hdf5_path}")
        if remove_csv:
            logger.getChild(DATASET).info(f"Removing all files in {data_dir}")
            shutil.rmtree(data_dir)
