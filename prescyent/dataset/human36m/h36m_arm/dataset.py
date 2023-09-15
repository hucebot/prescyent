"""Subset of h36m with arms only
"""
from typing import List, Union, Dict

from prescyent.dataset.trajectories import Trajectory

from prescyent.dataset.human36m.h36m_arm.config import DatasetConfig
from prescyent.dataset.human36m.dataset import Dataset as H36MDataset


LEFT_ARM_LABELS = [
    "left_shoulder_17",
    "left_elbow_18",
    "left_wrist_19",
    "left_wrist_20",
    "left_hand_21",
    "left_hand_22",
    "left_hand_23",
]
RIGHT_ARM_LABELS = [
    "right_shoulder_25",
    "right_elbow_26",
    "right_wrist_27",
    "right_wrist_28",
    "right_hand_29",
    "right_hand_30",
    "right_hand_31",
]

RELATIVE_LEFT_ARM_LABEL = LEFT_ARM_LABELS[0]
RELATIVE_RIGHT_ARM_LABEL = RIGHT_ARM_LABELS[0]
RELATIVE_BOTH_ARMS_LABEL = "neck_13"

MIRROR_AXIS_LABELS = [
    "crotch_0",
    "crotch_11",
    "spine_12",
    "thorax_13",
    "neck_16",
    "neck_24",
]

ARM_MAP = {
    "left": (LEFT_ARM_LABELS, RELATIVE_LEFT_ARM_LABEL),
    "right": (RIGHT_ARM_LABELS, RELATIVE_RIGHT_ARM_LABEL),
}


class Dataset(H36MDataset):
    """Class for data loading et preparation before the MotionDataset sampling"""

    DATASET_NAME = "H36MSingleArm"

    def __init__(self, config: Union[Dict, DatasetConfig] = DatasetConfig()):
        super().__init__(config)

    def pathfiles_to_trajectories(
        self,
        files: List,
        delimiter: str = ",",
        subsampling_step: int = 0,
        used_joints: List[int] = None,
    ) -> list:
        """util method to turn a list of pathfiles to a list of their data
        :rtype: list
        """
        trajectory_list = super().pathfiles_to_trajectories(
            files, delimiter, subsampling_step, used_joints
        )
        if self.config.bimanual is True:
            arm_joint_names = RIGHT_ARM_LABELS + LEFT_ARM_LABELS
            arm_joint_ids = [
                trajectory_list[0].dimension_names.index(name)
                for name in arm_joint_names
                if name in trajectory_list[0].dimension_names
            ]
            relative_joint_label = RELATIVE_BOTH_ARMS_LABEL
            return subsample_trajectories(
                trajectory_list, arm_joint_ids, relative_joint_label
            )
        # Else use main_arm, and can do data augmentation from second arm
        arm_joint_names, relative_joint_label = ARM_MAP.get(self.config.main_arm)
        arm_joint_ids = [
            trajectory_list[0].dimension_names.index(name)
            for name in arm_joint_names
            if name in trajectory_list[0].dimension_names
        ]
        new_trajectory_list = subsample_trajectories(
            trajectory_list, arm_joint_ids, relative_joint_label
        )
        if self.config.use_both_arms:
            second_arm_name = "left" if self.config.main_arm == "right" else "right"
            arm_joint_names, relative_joint_label = ARM_MAP.get(second_arm_name)
            arm_joint_ids = [
                trajectory_list[0].dimension_names.index(name)
                for name in arm_joint_names
                if name in trajectory_list[0].dimension_names
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
            trajectory.dimension_names.index(relative_joint_label)
            if relative_joint_label in trajectory.dimension_names
            else None
        )
        if relative_joint_id is not None:
            new_trajectory.tensor = new_trajectory.tensor - trajectory.tensor[
                :, relative_joint_id, :
            ].detach().unsqueeze(1)
        new_trajectory_list.append(new_trajectory)
    return new_trajectory_list


def get_joints(trajectory: Trajectory, idx_list: List[int]) -> Trajectory:
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
        dimension_names=[trajectory.dimension_names[idx] for idx in idx_list],
        file_path=trajectory.file_path,
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
