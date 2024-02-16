"""Get feature wise distance foreach point of last frame as a torch loss class"""
import torch

from prescyent.dataset.features.feature_manipulation import cal_distance_for_feat
from prescyent.predictor.lightning.configs.module_config import ModuleConfig


class MeanFinalRigidDistanceLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self,
        config: ModuleConfig,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super(MeanFinalRigidDistanceLoss, self).__init__(
            size_average, reduce, reduction
        )
        self.out_features = config.dataset_config.out_features

    def forward(self, input_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = torch.zeros(
            len(self.out_features),
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )
        input_tensor = input_tensor[:, -1]
        target = target[:, -1]
        for f, feat in enumerate(self.out_features):
            losses[f] = torch.mean(
                cal_distance_for_feat(
                    input_tensor[:, :, feat.ids],
                    target[:, :, feat.ids],
                    feat,
                )
            )
        return losses.mean()
