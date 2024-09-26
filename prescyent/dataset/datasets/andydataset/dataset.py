"""Class and methods for the AndyDataset Dataset
https://andydataset.loria.fr/
"""
from pathlib import Path
import shutil
import tempfile
from typing import List, Union, Dict
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import torch
from tqdm import tqdm

from prescyent.dataset.hdf5_utils import write_metadata, get_dataset_keys
from prescyent.utils.logger import logger, DATASET
from prescyent.utils.dataset_manipulation import (
    split_array_with_ratios,
    update_parent_ids,
)
from prescyent.utils.interpolate import update_tensor_frequency
import prescyent.dataset.datasets.andydataset.metadata as metadata
from prescyent.dataset.datasets.andydataset.config import DatasetConfig
from prescyent.dataset.features import Coordinate
from prescyent.dataset.trajectories.trajectories import Trajectories
from prescyent.dataset.trajectories.trajectory import Trajectory
from prescyent.dataset.dataset import MotionDataset


class Dataset(MotionDataset):
    """
    https://andydataset.loria.fr/
    Dataset is not splitted into test / train / val
    It as to be at initialisation, through the parameters
    """

    DATASET_NAME = "AndyDataset"

    def __init__(
        self,
        config: Union[Dict, DatasetConfig, str, Path] = None,
        load_data_at_init: bool = True,
    ) -> None:
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
        # Select only trajs from given participants:
        all_trajectory_names = [key for key in all_keys if key[-5:] == "/traj"]
        if self.config.participants:
            all_trajectory_names = [
                key
                for key in all_trajectory_names
                if any([participant in key for participant in self.config.participants])
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
            for traj_name in tqdm(
                trajs, colour="blue", desc="Writing used trajectories in temp file hdf5"
            ):
                tensor = torch.from_numpy(np.array(hdf5_data[traj_name]))
                context = {
                    key: hdf5_data[traj_name[:-4] + key]
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
                if self.config.make_joints_position_relative_to is not None:
                    for feat in metadata.DEFAULT_FEATURES:
                        if isinstance(feat, Coordinate):
                            tensor[:, :, feat.ids] -= torch.index_select(
                                tensor,
                                1,
                                torch.IntTensor(
                                    [self.config.make_joints_position_relative_to]
                                ),
                            )[:, :, feat.ids]
                tmp_hdf5_data.create_dataset(key + traj_name, data=tensor)
                for context_name, context_tensor in context.items():
                    tmp_hdf5_data.create_dataset(
                        key + traj_name[:-4] + context_name,
                        data=context_tensor,
                    )
        tmp_hdf5_data.attrs["frequency"] = self.config.frequency
        self.trajectories = Trajectories.__init_from_hdf5__(self.tmp_hdf5.name)

    def _get_from_web(self) -> None:
        raise NotImplementedError(
            "This dataset must be downloaded manually, "
            "please follow the instructions in the README"
        )

    @staticmethod
    def create_hdf5(
        hdf5_path: str, data_dir: str, subsets: str, remove_csv: bool = False
    ):
        files = list(Path(data_dir).rglob(subsets))
        logger.getChild(DATASET).info(f"Found {len(files)} files")
        with h5py.File(hdf5_path, "w") as hdf5_f:
            write_metadata(hdf5_f, metadata)

        for f_path in tqdm(files, colour="blue", desc="Iterating mvxn files"):
            traj_groups = list(f_path.relative_to(data_dir).parts)
            tree = ET.parse(f_path)
            root = tree.getroot()
            subject = root.find(f"./{metadata.SCHEMA}subject")
            frames = subject.findall(
                f'./{metadata.SCHEMA}frames/{metadata.SCHEMA}frame[@type="normal"]'
            )
            frame_list = []
            context = {key: [] for key in metadata.CONTEXT_KEYS}
            for frame in tqdm(frames, desc="Iterating frames", colour="yellow"):
                # for frame in frames:
                position = torch.FloatTensor(
                    [
                        float(x)
                        for x in frame.find(f"./{metadata.SCHEMA}position").text.split()
                    ]
                ).reshape(23, 3)
                orientation = torch.FloatTensor(
                    [
                        float(x)
                        for x in frame.find(
                            f"./{metadata.SCHEMA}orientation"
                        ).text.split()
                    ]
                ).reshape(23, 4)
                # (w, x, y, z) => (x, y, z, w)
                orientation = orientation[:, [1, 2, 3, 0]]
                # Ensure we have the quaternion with a positive w to avoid double cover
                indices = torch.nonzero(orientation[:, -1] < 0, as_tuple=True)
                orientation[indices] = -orientation[indices]
                frame_tensor = torch.cat((position, orientation), dim=-1)
                frame_list.append(frame_tensor.tolist())
                # load all other context keys for given frame
                for key in metadata.CONTEXT_KEYS:
                    context[key].append(
                        [
                            float(x)
                            for x in frame.find(
                                f"./{metadata.SCHEMA}{key}"
                            ).text.split()
                        ]
                    )
            torch_tensor = torch.FloatTensor(frame_list)
            traj_groups = traj_groups[:-1] + [traj_groups[-1][:-5]]
            with h5py.File(hdf5_path, "a") as f:
                group = f.create_group("/".join(traj_groups))
                group.create_dataset("traj", data=torch_tensor)
                for key in metadata.CONTEXT_KEYS:
                    group.create_dataset(f"{key}", data=torch.FloatTensor(context[key]))
        logger.getChild(DATASET).info(f"Created new HDF5 at {hdf5_path}")
        if remove_csv:
            logger.getChild(DATASET).info(f"Removing all files in {data_dir}")
            shutil.rmtree(data_dir)
