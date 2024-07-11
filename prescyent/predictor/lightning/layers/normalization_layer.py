import torch

from prescyent.dataset.features import Rotation
from prescyent.predictor.lightning.configs.module_config import ModuleConfig
from prescyent.predictor.lightning.layers.transpose_layer import TransposeLayer
from prescyent.utils.enums.normalizations import Normalizations


class MotionLayerNorm(torch.nn.Module):
    def __init__(self, config: ModuleConfig):
        super(MotionLayerNorm, self).__init__()
        self.used_norm = config.used_norm
        self.norm_rotation = config.norm_rotation
        self.in_sequence_size = config.in_sequence_size
        self.in_features = config.dataset_config.in_features
        self.num_in_points = config.dataset_config.num_in_points
        self.norm_layers = torch.nn.ModuleList()
        if not self.in_features and self.used_norm is not None:
            raise AttributeError(
                f"Cannot perform {self.used_norm} feature wise normalization"
                " if in_features aren't in config.dataset_config.in_features"
            )
        for feat in self.in_features:
            if self.used_norm is None or (
                isinstance(feat, Rotation) and not self.norm_rotation
            ):
                self.norm_layers.append(torch.nn.Identity())
            elif self.used_norm == Normalizations.SPATIAL:
                self.norm_layers.append(
                    torch.nn.LayerNorm([self.num_in_points, len(feat.ids)])
                )
            elif self.used_norm == Normalizations.TEMPORAL:
                self.norm_layers.append(
                    torch.nn.Sequential(
                        TransposeLayer(1, -1),
                        torch.nn.LayerNorm(self.in_sequence_size),
                        TransposeLayer(1, -1),
                    )
                )
            elif self.used_norm == Normalizations.ALL:
                self.norm_layers.append(
                    torch.nn.LayerNorm(
                        [self.in_sequence_size, self.num_in_points, len(feat.ids)]
                    )
                )
            elif self.used_norm == Normalizations.BATCH:
                self.norm_layers.append(torch.nn.BatchNorm2d(self.in_sequence_size))
            else:
                raise AttributeError(
                    f"Couldn't match {self.used_norm} with a valid normalization"
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Perform feature aware normalization
        for f, feat in enumerate(self.in_features):
            x[:, :, :, feat.ids] = self.norm_layers[f](x[:, :, :, feat.ids])
        return x
