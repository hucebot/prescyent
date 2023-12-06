import torch.nn as nn
from torch import Tensor

from prescyent.predictor.lightning.configs.module_config import ModuleConfig
from prescyent.utils.enums.normalizations import Normalizations


class MotionLayerNorm(nn.Module):
    def __init__(self, config: ModuleConfig):
        super(MotionLayerNorm, self).__init__()
        self.used_norm = config.used_norm
        if self.used_norm is None:
            self.norm_layer = nn.Identity()
        elif self.used_norm == Normalizations.SPATIAL:
            self.norm_layer = nn.LayerNorm(
                [
                    len(config.dataset_config.in_points),
                    len(config.dataset_config.in_dims),
                ]
            )
        elif self.used_norm == Normalizations.TEMPORAL:
            self.norm_layer = nn.LayerNorm(config.input_size)
        elif self.used_norm == Normalizations.ALL:
            self.norm_layer = nn.LayerNorm(
                [
                    config.input_size,
                    len(config.dataset_config.in_points),
                    len(config.dataset_config.in_dims),
                ]
            )
        elif self.used_norm == Normalizations.BATCH:
            self.norm_layer = nn.BatchNorm2d(config.input_size)
        else:
            raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        if self.used_norm == Normalizations.TEMPORAL:
            x = x.transpose(1, -1)
        y = self.norm_layer(x)
        if self.used_norm == Normalizations.TEMPORAL:
            y = y.transpose(1, -1)
        return y
