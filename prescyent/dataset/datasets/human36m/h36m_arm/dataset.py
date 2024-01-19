"""Subset of h36m with arms only"""
from typing import List, Union, Dict

import prescyent.dataset.datasets.human36m.h36m_arm.metadata as metadata
from prescyent.dataset.datasets.human36m.h36m_arm.config import DatasetConfig
from prescyent.dataset.datasets.human36m.dataset import Dataset as H36MDataset
from prescyent.dataset.trajectories.trajectory import Trajectory
from prescyent.utils.dataset_manipulation import update_parent_ids


class Dataset(H36MDataset):
    """Class for data loading et preparation before the MotionDataset sampling"""

    DATASET_NAME = "H36MSArm"

    def __init__(self, config: Union[Dict, DatasetConfig] = None) -> None:
        if config is None:
            config = DatasetConfig()
        super().__init__(config)

    def pathfiles_to_trajectories(
        self,
        files: List,
        delimiter: str = ",",
    ) -> List[Trajectory]:
        """util method to turn a list of pathfiles to a list of their data
        :rtype: List[Trajectory]
        """
        trajectory_list = super().pathfiles_to_trajectories(files, delimiter)
        if self.config.bimanual is True:
            arm_joint_names = metadata.RIGHT_ARM_LABELS + metadata.LEFT_ARM_LABELS
            arm_joint_ids = [
                trajectory_list[0].point_names.index(name)
                for name in arm_joint_names
                if name in trajectory_list[0].point_names
            ]
            relative_joint_label = metadata.RELATIVE_BOTH_ARMS_LABEL
            return subsample_trajectories(
                trajectory_list, arm_joint_ids, relative_joint_label
            )
        # Else use main_arm, and can do data augmentation from second arm
        arm_joint_names, relative_joint_label = metadata.ARM_MAP.get(
            self.config.main_arm
        )
        arm_joint_ids = [
            trajectory_list[0].point_names.index(name)
            for name in arm_joint_names
            if name in trajectory_list[0].point_names
        ]
        new_trajectory_list = subsample_trajectories(
            trajectory_list, arm_joint_ids, relative_joint_label
        )
        if self.config.use_both_arms:
            second_arm_name = "left" if self.config.main_arm == "right" else "right"
            arm_joint_names, relative_joint_label = metadata.ARM_MAP.get(
                second_arm_name
            )
            arm_joint_ids = [
                trajectory_list[0].point_names.index(name)
                for name in arm_joint_names
                if name in trajectory_list[0].point_names
            ]
            if self.config.mirror_second_arm:
                trajectory_list = mirror_trajectory(trajectory_list)
            new_trajectory_list += subsample_trajectories(
                trajectory_list, arm_joint_ids, relative_joint_label
            )
        return new_trajectory_list


def subsample_trajectories(
    trajectory_list: List[Trajectory],
    arm_joint_ids: List[int],
    relative_joint_label: str,
) -> List[Trajectory]:
    new_trajectory_list = []
    for trajectory in trajectory_list:
        new_trajectory = get_joints(trajectory, arm_joint_ids)
        relative_joint_id = (
            trajectory.point_names.index(relative_joint_label)
            if relative_joint_label in trajectory.point_names
            else None
        )
        if relative_joint_id is not None:
            new_trajectory.tensor = new_trajectory.tensor - trajectory.tensor[
                :, relative_joint_id, :
            ].detach().unsqueeze(1)
        new_trajectory_list.append(new_trajectory)
    return new_trajectory_list


def get_joints(
    trajectory: Trajectory, idx_list: List[int]
) -> Trajectory:
    """return a subset of the given trajectory on given points ids

    Args:
        trajectory (Trajectory): trajectory to split
        idx_list (List[int]): list of point ids we want to keep

    Returns:
        Trajectory: the trajectory point subset
    """
    new_trajectory = Trajectory(
        tensor=trajectory.tensor[:, idx_list, :],
        frequency=trajectory.frequency,
        tensor_features=trajectory.tensor_features,
        file_path=trajectory.file_path,
        title=trajectory.title,
        point_parents=update_parent_ids(idx_list, trajectory.point_parents),
        point_names=[trajectory.point_names[idx] for idx in idx_list],
    )
    return new_trajectory


def mirror_trajectory(trajectory: Trajectory) -> Trajectory:
    """returns a new trajectory mirrored given the MIRROR_AXIS

    Args:
        trajectory (Trajectory): trajectory to mirror

    Returns:
        Trajectory: mirrored trajectory
    """
    raise NotImplementedError("TODO, method not implemented yet")
