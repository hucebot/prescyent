from typing import List

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
        point_parents: List[int] = ...,
        dimension_names: List[str] = ...,
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

    def dump(
        self,
        output_format="csv",
        output_path=None,
        rotation_representation: RotationRepresentation = None,
    ) -> None:
        if rotation_representation is not None:
            for points in self.sequence_of_positions:
                for position in points:
                    position.rotation_representation = rotation_representation
        self.tensor = self.get_tensor()
        super().dump(
            output_format=output_format,
            output_path=output_path,
        )

    @property
    def rotation_representation(self) -> RotationRepresentation:
        return self.sequence_of_positions[0][0].rotation_representation

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
