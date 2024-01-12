from typing import List, Optional

import numpy as np
import torch

import prescyent.utils.torch_rotation as tr
from prescyent.dataset.trajectories.trajectory import Trajectory
from prescyent.utils.enums import RotationRepresentation
from prescyent.utils.rotation_6d import Rotation6d


DEFAULT_EULER_SEQ = "XYZ"


class PositionsTrajectory(Trajectory):
    def __init__(
        self,
        tensor: torch.Tensor,
        rotation_representation: RotationRepresentation,
        frequency: int,
        file_path: str = "trajectory_file_path",
        title: str = "trajectory_name",
        point_parents: List[int] = None,
        dimension_names: List[str] = ["y_infos"],
    ) -> None:
        super().__init__(
            tensor,
            frequency,
            file_path,
            title,
            point_parents,
            dimension_names,
        )
        self._rotation_representation = rotation_representation

    @property
    def rotation_representation(self) -> RotationRepresentation:
        return self._rotation_representation

    @rotation_representation.setter
    def rotation_representation(self, value: RotationRepresentation):
        """do tensor rotation format conversion at rotation_representation update

        Args:
            value (RotationRepresentation): the new rotation format
        """
        if self.rotation_representation == None:
            raise AttributeError(
                "Cannot convert actual tensor without rotation to a new rotation format"
            )
        coordinates = self.tensor[:, :, :3]
        rotation = self.tensor[:, :, 3:]
        if value == RotationRepresentation.EULER:
            rotation = tr.convert_to_euler(rotation)
        elif value == RotationRepresentation.QUATERNIONS:
            rotation = tr.convert_to_quat(rotation)
        elif value == RotationRepresentation.ROTMATRICES:
            rotation = tr.convert_to_rotmatrix(rotation)
        elif value == RotationRepresentation.REP6D:
            rotation = tr.convert_to_rep6d(rotation)
        else:
            raise AttributeError(f"{value} is not an handled RotationRepresentation")
        self.tensor = torch.cat((coordinates, rotation), dim=2)
        self._rotation_representation = value

    def _get_header(self) -> List[str]:
        dim_names = ["x", "y", "z"]
        if self.rotation_representation is None:  # No rotation
            pass
        elif self.rotation_representation == RotationRepresentation.QUATERNIONS:
            dim_names += ["qx", "qy", "qz", "qw"]
        elif self.rotation_representation == RotationRepresentation.EULER:
            dim_names += [f"e{dim}" for dim in DEFAULT_EULER_SEQ]
        elif self.rotation_representation == RotationRepresentation.ROTMATRICES:
            dim_names += ["x1", "x2", "x3", "y1", "y2", "y3", "z1", "z2", "z3"]
        elif self.rotation_representation == RotationRepresentation.REP6D:
            dim_names += ["x1", "x2", "x3", "y1", "y2", "y3"]
        return [
            f"{self.dimension_names[p]}_{d}"
            for d in dim_names
            for p in range(self.num_points)
        ]

    def dump(
        self,
        output_format="csv",
        output_path=None,
        rotation_representation: RotationRepresentation = None,
    ) -> None:
        self.rotation_representation = rotation_representation
        super().dump(
            output_format=output_format,
            output_path=output_path,
        )

    def visualize_3d(
        self,
        save_file: str = None,  # use "mp4" or "gif"
        min_max_layout: bool = True,
        interactive: bool = True,
        draw_bones: bool = True,
        turn_view: bool = False,
        draw_rotation: bool = None,
    ) -> None:
        from prescyent.evaluator.visualize_3d import render_3d_trajectory

        if draw_rotation is None and self.rotation_representation is not None:
            draw_rotation = True
        elif draw_rotation is None:
            draw_rotation = False
        render_3d_trajectory(
            self,
            save_file,
            min_max_layout,
            interactive,
            draw_bones,
            draw_rotation,
            turn_view,
        )

    def get_scipy_rotation(self, frame: int, point: int) -> Rotation6d:
        rotation_tensor = self.tensor[frame][point][3:]
        if self.rotation_representation == RotationRepresentation.QUATERNIONS:
            return Rotation6d.from_quat(rotation_tensor)
        if self.rotation_representation == RotationRepresentation.EULER:
            return Rotation6d.from_euler(rotation_tensor)
        if self.rotation_representation == RotationRepresentation.ROTMATRICES:
            return Rotation6d.from_matrix(rotation_tensor.reshape(3, 3))
        if self.rotation_representation == RotationRepresentation.REP6D:
            return Rotation6d.from_rep6d(rotation_tensor.reshape(3, 2))
        raise AttributeError(
            "Can't output a rotation with the actual rotation_representation "
            f"{self.rotation_representation}"
        )

    def compare(self, other: object, offsets: List[int] = [0, 0]) -> float:
        assert isinstance(other, PositionsTrajectory)
        assert len(offsets) == 2
        seq1 = self.tensor[offsets[0] :]
        seq2 = (other.tensor[offsets[1] :])[: len(seq1)]
        assert len(seq1) == len(seq2)
        assert len(seq1[0]) == len(seq2[0])
        frame_dists = []
        # TODO: some position aware loss with coordinate error and geodesic error
        for f, frame in enumerate(seq1):
            point_dists = []
            for p, point in enumerate(frame):
                coordonate_dist, rotation_dist = point.calc_distance(
                    other.sequence_of_positions[f][p]
                )
                point_dists.append((coordonate_dist, rotation_dist))
            frame_dists.append(point_dists)
        frame_means = [np.mean(point_dists, 0) for point_dists in frame_dists]
        mean_dist = np.mean(frame_means, 0)
        return mean_dist
