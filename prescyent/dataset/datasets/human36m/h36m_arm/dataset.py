"""Subset of h36m with arms only"""
from typing import List, Union, Dict

import torch

import prescyent.dataset.datasets.human36m.h36m_arm.metadata as metadata
from prescyent.dataset.datasets.human36m.h36m_arm.config import DatasetConfig
from prescyent.dataset.datasets.human36m.dataset import Dataset as H36MDataset
from prescyent.dataset.datasets.human36m.metadata import POINT_LABELS
from prescyent.dataset.features import Coordinate
from prescyent.dataset.hdf5_utils import get_dataset_keys
from prescyent.dataset.trajectories.trajectory import Trajectory
from prescyent.utils.dataset_manipulation import update_parent_ids


class Dataset(H36MDataset):
    """Class for data loading et preparation before the MotionDataset sampling"""

    DATASET_NAME = "H36MArm"

    def __init__(
        self, config: Union[Dict, DatasetConfig] = None, load_data_at_init: bool = True
    ) -> None:
        if config is None:
            config = DatasetConfig()
        super().__init__(
            config=config,
            config_class=DatasetConfig,
            load_data_at_init=load_data_at_init,
        )

    def prepare_data(self):
        """util method to turn a list of pathfiles to a list of their data
        :rtype: List[Trajectory]
        """
        super().prepare_data()
        traj_point_names = (
            self.trajectories.train + self.trajectories.test + self.trajectories.val
        )[0].point_names
        if self.config.bimanual is True:
            arm_joint_names = metadata.LEFT_ARM_LABELS + metadata.RIGHT_ARM_LABELS
            relative_joint_label = metadata.RELATIVE_BOTH_ARMS_LABEL
            arm_joint_ids = [
                traj_point_names.index(name)
                for name in arm_joint_names
                if name in traj_point_names
            ]
        else:
            # Else use main_arm
            arm_joint_names, relative_joint_label = metadata.ARM_MAP.get(
                self.config.main_arm
            )
            arm_joint_ids = [
                traj_point_names.index(name)
                for name in arm_joint_names
                if name in traj_point_names
            ]
        # Subsample all trajectories
        for trajectory in (
            self.trajectories.train + self.trajectories.test + self.trajectories.val
        ):
            trajectory.tensor = trajectory.tensor[:, arm_joint_ids, :]
            trajectory.point_parents = update_parent_ids(
                arm_joint_ids, trajectory.point_parents
            )
            trajectory.point_names = [
                trajectory.point_names[idx] for idx in arm_joint_ids
            ]
            relative_joint_id = (
                trajectory.point_names.index(relative_joint_label)
                if relative_joint_label in trajectory.point_names
                else None
            )
            # Make Coordinates relative to the joint, not rotation
            if relative_joint_id is not None:
                for feat in trajectory.tensor_features:
                    if isinstance(feat, Coordinate):
                        trajectory.tensor[:, :, feat.ids] -= torch.index_select(
                            trajectory.tensor, 1, torch.IntTensor([relative_joint_id])
                        )[:, :, feat.ids]
