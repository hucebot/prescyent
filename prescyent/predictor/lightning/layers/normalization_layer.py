import torch
from torch import Tensor

from prescyent.dataset.features import Rotation
from prescyent.predictor.lightning.configs.module_config import ModuleConfig
from prescyent.utils.enums.normalizations import Normalizations


class MotionLayerNorm(torch.nn.Module):
    def __init__(self, config: ModuleConfig):
        super(MotionLayerNorm, self).__init__()
        self.used_norm = config.used_norm
        self.input_size = config.input_size
        self.in_features = config.dataset_config.in_features
        self.num_in_points = config.dataset_config.num_in_points
        self.norm_layers = []
        for feat in self.in_features:
            if self.used_norm is None or isinstance(feat, Rotation):
                self.norm_layers.append(torch.nn.Identity())
            elif self.used_norm == Normalizations.SPATIAL:
                self.norm_layers.append(
                    torch.nn.LayerNorm([self.num_in_points, len(feat.ids)])
                )
            elif self.used_norm == Normalizations.TEMPORAL:
                self.norm_layers.append(
                    torch.nn.Sequential(
                        [
                            torch.nn.Transpose(1, -1),
                            torch.nn.LayerNorm(self.input_size),
                            torch.nn.Transpose(1, -1),
                        ]
                    )
                )
            elif self.used_norm == Normalizations.ALL:
                self.norm_layers.append(
                    torch.nn.LayerNorm(
                        [self.input_size, self.num_in_points, len(feat.ids)]
                    )
                )
            elif self.used_norm == Normalizations.BATCH:
                self.norm_layers.append(torch.nn.BatchNorm2d(self.input_size))
            else:
                raise NotImplementedError()

    def forward(self, x: Tensor) -> Tensor:
        # Perform feature aware normalization
        for f, feat in enumerate(self.in_features):
            x[:, :, :, feat.ids] = self.norm_layers[f](x[:, :, :, feat.ids])
        return x
