"""Get feature wise distance foreach point as a torch loss class"""
import torch

from prescyent.dataset.features.feature_relative import get_relative_tensor_from
from prescyent.predictor.lightning.configs.module_config import ModuleConfig
from prescyent.predictor.lightning.losses.mtrd_loss import MeanTotalRigidDistanceLoss


class MeanTotalRigidDistanceAndVelocityLoss(MeanTotalRigidDistanceLoss):
    def __init__(
        self,
        config: ModuleConfig,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super(MeanTotalRigidDistanceAndVelocityLoss, self).__init__(
            config, size_average, reduce, reduction
        )

    def forward(self, input_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dloss = super().forward(input_tensor, target)
        target_vel = get_relative_tensor_from(
            target[:, 1:], target[:, :-1], self.out_features
        )
        input_vel = get_relative_tensor_from(
            input_tensor[:, 1:], input_tensor[:, :-1], self.out_features
        )
        vloss = super().forward(input_vel, target_vel)
        return dloss + vloss
