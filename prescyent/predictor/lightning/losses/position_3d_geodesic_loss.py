import time

import numpy as np
import torch

from prescyent.predictor.lightning.losses.geodesic_loss import GeodesicLoss
from prescyent.utils.tensor_manipulation import reshape_position_tensor


class Position3DGeodesicLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self,
        geodesic_ratio: float = 0.5,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super(Position3DGeodesicLoss, self).__init__(size_average, reduce, reduction)
        self.geodesic_loss = GeodesicLoss(size_average, reduce, reduction)
        self.mse_loss = torch.nn.MSELoss(size_average, reduce, reduction)
        self.geodesic_ratio = geodesic_ratio

    def forward(
        self, pos_tensor_preds: torch.Tensor, pos_tensor_targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute a rotation aware loss using geodesic loss on rotation and mse on coordinates
        """
        # ensure we manipulate positions with coordinates and rotmatrices
        # it is way too long as is, as we convert rotations 1 by 1
        # TODO: improve reshaping to be batched
        # or have a loss function for each representation, not just rotmatrices
        # start_time = time.time()
        pos_tensor_preds = reshape_position_tensor(pos_tensor_preds)
        pos_tensor_targets = reshape_position_tensor(pos_tensor_targets)
        # intermediate_time = time.time()
        # mse on coordinates
        mse_loss = self.mse_loss(
            pos_tensor_preds[:, :, :, :3], pos_tensor_targets[:, :, :, :3]
        )
        # geodesic on rotations
        geodesic_loss = self.geodesic_loss(
            pos_tensor_preds[:, :, :, 3:], pos_tensor_targets[:, :, :, 3:]
        )
        position_loss = geodesic_loss * self.geodesic_ratio + mse_loss * (
            1 - self.geodesic_ratio
        )
        # final_time = time.time()
        # print(f"Reshaping_time: {intermediate_time - start_time}")
        # print(f"Loss_calculation_time: {final_time - intermediate_time}")
        return position_loss
