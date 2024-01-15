from typing import List, Type
from prescyent.dataset.dataset import MotionDataset
from prescyent.dataset.trajectories.position_trajectory import PositionsTrajectory
from prescyent.dataset.three_dimensional_dataset.config import Dataset3dConfig
from prescyent.dataset.three_dimensional_dataset.datasamples import (
    Position3dDataSamples,
)
from prescyent.utils.enums import RotationRepresentation


class Dataset3D(MotionDataset):
    """Base classe for 3d positions and rotations motion datasets"""

    config: Dataset3dConfig
    config_class: Type[Dataset3dConfig]

    def __init__(self, name: str) -> None:
        self.sample_class = Position3dDataSamples
        self.config.in_dims = None  # force all dims for baseclass, then subsample with coordinates_in and rotation_representation_in
        self.config.out_dims = None  # force all dims for baseclass, then subsample with coordinates_out and rotation_representation_out
        assert isinstance(self.trajectories[0], PositionsTrajectory)
        # Set default values for coordinates_in and _out to all
        if self.config.coordinates_in is None:
            self.config.coordinates_in = [0, 1, 2]
        if self.config.coordinates_out is None:
            self.config.coordinates_out = [0, 1, 2]
        # Check rotation representation according to the config
        if (
            self.config.rotation_representation_in is not None
            or self.config.rotation_representation_out is not None
        ) and self.rotation_representation is None:
            raise AttributeError(
                "Could not prepare a dataset with trajectories having "
                f"rotation_representation={self.rotation_representation} for config"
                f"rotation_representation_in={self.config.rotation_representation_in} "
                f" and rotation_representation_out={self.config.rotation_representation_out}"
            )
        if (
            self.config.rotation_representation_in
            == self.config.rotation_representation_out
            and self.rotation_representation != self.config.rotation_representation_in
        ):
            self.update_trajectories_rotation(self.config.rotation_representation_in)
        elif (
            self.config.rotation_representation_in
            != self.config.rotation_representation_out
        ):
            self.update_trajectories_rotation(RotationRepresentation.ROTMATRICES)
        super().__init__(name)

    @property
    def rotation_representation(self) -> RotationRepresentation:
        return self.trajectories.train[0].rotation_representation

    def update_trajectories_rotation(self, rotation: RotationRepresentation):
        """update all dataset's trajectories rotation representations

        Args:
            rotation (RotationRepresentation): Enum value for the used values to rep rotations
        """
        for t in self.trajectories.train:
            t.rotation_representation = rotation
        for t in self.trajectories.test:
            t.rotation_representation = rotation
        for t in self.trajectories.val:
            t.rotation_representation = rotation
