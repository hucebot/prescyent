"""Ros2 topic suscriber to current positions
that predicts and publishes the N next ones"""
import copy
from typing import List

import torch
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from prescyent_msgs.msg import Trajectory
from prescyent.predictor.base_predictor import BasePredictor


def get_pose_from_list(pose_list: List[float]) -> Pose:
    """Creates a Pose object from a list of pose values"""
    pose = Pose()
    # 7 elements in Pose: Point x, y, z + Quaternion x, y, z, w
    pose.position.x = pose_list[0] if len(pose_list) >= 0 else None
    pose.position.y = pose_list[1] if len(pose_list) >= 1 else None
    pose.position.z = pose_list[2] if len(pose_list) >= 2 else None
    pose.orientation.x = pose_list[3] if len(pose_list) >= 3 else None
    pose.orientation.y = pose_list[4] if len(pose_list) >= 4 else None
    pose.orientation.z = pose_list[5] if len(pose_list) >= 5 else None
    return pose

def get_list_from_pose(pose: Pose) -> List[float]:
    return([pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
            ])


class PredictorNode(Node):
    def __init__(self,
                 predictor: BasePredictor,
                 history_size: int,
                 future_size: int,
                 time_step: int):
        super().__init__("prescyent_predictor")
        self.predictor = predictor
        self.history_size = history_size
        self.future_size = future_size
        self.time_step = time_step
        self._pose_buffer = list()
        self.prediction_publisher = self.create_publisher(
            PoseArray, "/prescyent/prediction", 10)
        self.pose_suscriber = self.create_subscription(
            Pose, "/prescyent/positions",  self.receive_pose, 10)
        self.get_logger().info("Predictor Node has been started.")


    def get_tensor_from_trajectory(self, poses: List[PoseArray]):
        pose_tensor = []
        return pose_tensor

    def get_trajectory_from_tensor(self, tensor: torch.Tensor, history: List[PoseArray]) -> List[PoseArray]:
        """Tensors are of size (seq_size, num_points, num_dims)
        We return a Trajectory of format [PoseArray[Pose]]"""
        sequence = tensor.to_list()
        trajectory = Trajectory()
        last_stamp = history[-1].header.stamp
        last_frame_id = history[-1].header.frame_id
        for pose_array_list in sequence:
            pose_array = PoseArray()
            for pose_list in pose_array_list:
                pose = get_pose_from_list(pose_list)
                pose_array.poses.append(pose)
            last_stamp = last_stamp + self.time_step
            last_frame_id += 1
            pose_array.header.stamp = last_stamp
            pose_array.header.frame_id = last_frame_id
            trajectory.pose_array_sequence.append(pose_array)
        return trajectory

    def receive_pose(self, pose:Pose):
        self._pose_buffer.append(Pose)
        if len(self._pose_buffer) == self.history_size:
            pose_buffer = copy.deepcopy(self._pose_buffer)
            history = self.get_tensor_from_trajectory(pose_buffer)
            self._pose_buffer = list()
            prediction = self.predictor(history, future_size=self.future_size)
            self.prediction_publisher.publish(self.get_trajectory_from_tensor(prediction, pose_buffer))
