"""Get feature wise distance foreach point as a torch loss class"""
import torch

from prescyent.dataset.features.feature_relative import get_relative_tensor_from
from prescyent.predictor.lightning.configs.module_config import ModuleConfig
from prescyent.predictor.lightning.losses.mtrd_loss import MeanTotalRigidDistanceLoss


class MeanTotalRigidDistanceAndVelocityLoss(MeanTotalRigidDistanceLoss):
    """Get feature wise distance foreach point as a torch loss class mixed with feature's velocity"""

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
        """forward method computing the loss

        Args:
            input_tensor (torch.Tensor): traj_tensor to compare with truth
            target (torch.Tensor): truth traj tensor

        Returns:
            torch.Tensor: mean feature wise distance + velocity
        """
        dloss = super().forward(input_tensor, target)
        target_vel = get_relative_tensor_from(
            target[:, 1:], target[:, :-1], self.out_features
        )
        input_vel = get_relative_tensor_from(
            input_tensor[:, 1:], input_tensor[:, :-1], self.out_features
        )
        vloss = super().forward(input_vel, target_vel)
        return dloss + vloss
