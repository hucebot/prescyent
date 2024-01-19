import time
from typing import List

import numpy as np
import torch

from prescyent.predictor.lightning.losses.geodesic_loss import GeodesicLoss
from prescyent.dataset.features import Rotation, RotationRotMat, Feature, convert_to_rotmatrix


class Position3DGeodesicLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self,
        out_features: List[Feature],
        geodesic_ratio: float = 0.5,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super(Position3DGeodesicLoss, self).__init__(size_average, reduce, reduction)
        self.geodesic_loss = GeodesicLoss(size_average, reduce, reduction)
        self.mse_loss = torch.nn.MSELoss(size_average, reduce, reduction)
        self.geodesic_ratio = geodesic_ratio
        self.out_features = out_features

    def forward(
        self, pos_tensor_preds: torch.Tensor, pos_tensor_targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute a rotation aware loss using geodesic loss on rotation and mse on other feats
        """
        geodesic_loss = torch.zeros([1], dtype=pos_tensor_preds.dtype, device=pos_tensor_preds.device)
        mse_loss = torch.zeros([1], dtype=pos_tensor_preds.dtype, device=pos_tensor_preds.device)
        for feat in self.out_features:
            if isinstance(feat, Rotation):
                if not isinstance(feat, RotationRotMat):
                    pos_tensor_preds_matrix = convert_to_rotmatrix(pos_tensor_preds[:,:,:,feat.ids])
                    pos_tensor_targets_matrix = convert_to_rotmatrix(pos_tensor_targets[:,:,:,feat.ids])
                else:
                    pos_tensor_preds_matrix = pos_tensor_preds[:,:,:,feat.ids]
                    pos_tensor_targets_matrix = pos_tensor_targets[:,:,:,feat.ids]
                geodesic_loss += self.geodesic_loss(pos_tensor_preds_matrix,
                                                    pos_tensor_targets_matrix)
            else:
                mse_loss += self.mse_loss(pos_tensor_preds[:,:,:,feat.ids],
                                          pos_tensor_targets[:,:,:,feat.ids])
        position_loss = geodesic_loss * self.geodesic_ratio + mse_loss * (1 - self.geodesic_ratio)
        return position_loss


        # # ensure we manipulate positions with coordinates and rotmatrices
        # # or have a loss function for each representation, not just rotmatrices
        # B = pos_tensor_preds.shape[0]
        # S = pos_tensor_preds.shape[1]
        # P = pos_tensor_preds.shape[2]
        # coordinate_preds, rotation_preds = (
        #     pos_tensor_preds[:, :, :, :3],
        #     pos_tensor_preds[:, :, :, 3:],
        # )
        # coordinate_targets, rotation_targets = (
        #     pos_tensor_targets[:, :, :, :3],
        #     pos_tensor_targets[:, :, :, 3:],
        # )
        # # ensure we manipulate positions with coordinates as rotmatrices
        # rotation_preds = convert_to_rotmatrix(
        #     rotation_preds.reshape(-1, rotation_preds.shape[-1])
        # ).reshape(B, S, P, 9)
        # rotation_targets = convert_to_rotmatrix(
        #     rotation_targets.reshape(-1, rotation_targets.shape[-1])
        # ).reshape(B, S, P, 9)
        # # mse on coordinates
        # mse_loss = self.mse_loss(coordinate_preds, coordinate_targets)
        # # geodesic on rotations
        # geodesic_loss = self.geodesic_loss(rotation_preds, rotation_targets)
        # position_loss = geodesic_loss * self.geodesic_ratio + mse_loss * (
        #     1 - self.geodesic_ratio
        # )
        # return position_loss
