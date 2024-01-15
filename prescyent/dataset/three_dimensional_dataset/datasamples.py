from typing import List

import torch

from prescyent.dataset.three_dimensional_dataset.config import Dataset3dConfig
from prescyent.dataset.datasamples import MotionDataSamples
from prescyent.dataset.trajectories.position_trajectory import PositionsTrajectory
from prescyent.utils.torch_rotation import (
    convert_rotation_tensor_to,
    get_tensor_rotation_representation,
)


class Position3dDataSamples(MotionDataSamples):
    """Class storing x,y pairs for ML trainings on motion data
    With specific rotation conversion methods according to datasetconfig
    """

    trajectories: List[PositionsTrajectory]

    def __init__(
        self, trajectories: List[PositionsTrajectory], config: Dataset3dConfig
    ) -> None:
        super().__init__(trajectories, config)

    def __getitem__(self, index: int):
        _in, _out = super().__getitem__(index=index)
        # Convert tensor's rotation representation if needed
        coordinates = _in[:, :, self.config.coordinates_in]
        rotation = _in[:, :, 3:]
        rotation = convert_rotation_tensor_to(
            rotation, self.config.rotation_representation_in
        )
        _in = torch.cat((coordinates, rotation), dim=2)
        # Convert tensor's rotation representation if needed
        coordinates = _out[:, :, self.config.coordinates_out]
        rotation = _out[:, :, 3:]
        rotation = convert_rotation_tensor_to(
            rotation, self.config.rotation_representation_out
        )
        _out = torch.cat((coordinates, rotation), dim=2)
        return _in, _out
