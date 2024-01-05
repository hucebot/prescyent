from typing import List, Optional

import numpy as np
import torch

from prescyent.dataset.trajectories.features import Position
from prescyent.dataset.trajectories.trajectory import Trajectory
from prescyent.utils.enums import RotationRepresentation


class PositionsTrajectory(Trajectory):
    def __init__(
        self,
        sequence_of_positions: List[List[Position]],
        frequency: int,
        file_path: str = "trajectory_file_path",
        title: str = "trajectory_name",
        point_parents: List[int] = None,
        dimension_names: List[str] = ["y_infos"],
    ) -> None:
        self.sequence_of_positions = sequence_of_positions
        super().__init__(
            self.get_tensor(),
            frequency,
            file_path,
            title,
            point_parents,
            dimension_names,
        )

    @property
    def sequence_of_positions(self):
        return self._sequence_of_positions

    @sequence_of_positions.setter
    def sequence_of_positions(self, value):
        self._sequence_of_positions = value
        self._tensor = self.get_tensor()

    @property
    def tensor(self):
        return self._tensor

    @tensor.setter
    def tensor(self, value):
        self._tensor = value
        self._sequence_of_positions = self.get_sequence_of_positions(
            self.rotation_representation
        )

    @property
    def rotation_representation(self) -> RotationRepresentation:
        return self.sequence_of_positions[0][0].rotation_representation

    @rotation_representation.setter
    def rotation_representation(self, value):
        assert isinstance(value, RotationRepresentation)
        for f, frame in enumerate(self._sequence_of_positions):
            for p, _ in enumerate(frame):
                self._sequence_of_positions[f][p].rotation_representation = value
        self._tensor = self.get_tensor()

    def get_tensor(self) -> torch.Tensor:
        return torch.FloatTensor(
            [
                [point.get_tensor().tolist() for point in sequence]
                for sequence in self.sequence_of_positions
            ]
        )

    def get_sequence_of_positions(
        self, rotation_rep: RotationRepresentation
    ) -> List[List[Position]]:
        return [
            [Position.get_from_tensor(point, rotation_rep) for point in sequence]
            for sequence in self.tensor
        ]

    def _get_header(self) -> List[str]:
        dims = self.sequence_of_positions[0][0].dim_names()
        return [
            f"{self.dimension_names[p]}_{d}"
            for d in dims
            for p in range(self.num_points)
        ]

    def augment_frequency(self, augmentation_ratio: int) -> None:
        super().augment_frequency(augmentation_ratio)
        self.sequence_of_positions = self.get_sequence_of_positions(
            self.rotation_representation
        )

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
        others: Optional[List[object]] = None,
    ) -> None:
        from prescyent.evaluator.visualize_3d import render_3d_trajectory

        if (
            draw_rotation is None
            and self.sequence_of_positions[0][0].rotation is not None
        ):
            draw_rotation = True
        elif draw_rotation is None:
            draw_rotation = False
        for other in others:
            assert isinstance(other, PositionsTrajectory)
        render_3d_trajectory(
            self,
            save_file,
            min_max_layout,
            interactive,
            draw_bones,
            draw_rotation,
            turn_view,
        )

    def compare(self, other: object, offsets: List[int] = [0, 0]) -> float:
        assert isinstance(other, PositionsTrajectory)
        assert len(offsets) == 2
        seq1 = self.sequence_of_positions[offsets[0] :]
        seq2 = (other.sequence_of_positions[offsets[1] :])[: len(seq1)]
        assert len(seq1) == len(seq2)
        assert len(seq1[0]) == len(seq2[0])
        frame_dists = []
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
