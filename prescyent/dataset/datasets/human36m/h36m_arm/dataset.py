"""Subset of h36m with arms only"""
from typing import Dict, Union

import torch

from prescyent.dataset.datasets.human36m.dataset import H36MDataset
from prescyent.dataset.features import Coordinate
from prescyent.utils.dataset_manipulation import update_parent_ids

from . import metadata
from .config import H36MArmDatasetConfig


class H36MArmDataset(H36MDataset):
    """Class for data loading et preparation before the TrajectoriesDataset sampling"""

    DATASET_NAME = "H36MArm"

    def __init__(self, config: Union[Dict, H36MArmDatasetConfig]) -> None:
        super().__init__(config=config, config_class=H36MArmDatasetConfig)

    def prepare_data(self):
        """generates self.trajectories from dataset's file"""
        if hasattr(self, "_trajectories"):
            return
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
