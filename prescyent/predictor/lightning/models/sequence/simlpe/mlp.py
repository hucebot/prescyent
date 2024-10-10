import torch
from torch import nn

from prescyent.utils.enums.trajectory_dimensions import TrajectoryDimensions


class CustomLayerNorm(nn.Module):
    def __init__(self, axis, size, epsilon=1e-5):
        super().__init__()
        self.axis = axis
        self.epsilon = epsilon
        _shape = [1, 1, 1]
        _shape[axis] = size
        self.alpha = nn.Parameter(torch.ones(_shape), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(_shape), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=self.axis, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=self.axis, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y


class SpatialFC(nn.Module):
    def __init__(self, dim):
        super(SpatialFC, self).__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.fc(x)
        x = torch.transpose(x, 1, 2)
        return x


class TemporalFC(nn.Module):
    def __init__(self, dim):
        super(TemporalFC, self).__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.fc(x)
        return x


class MLPBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.spatial_fc_only:
            self.fc0 = SpatialFC(config.in_points_dims)
        else:
            self.fc0 = TemporalFC(config.in_sequence_size)
        if config.mpl_blocks_norm:
            if config.mpl_blocks_norm == TrajectoryDimensions.SPATIAL:
                self.norm0 = CustomLayerNorm(1, config.in_points_dims)
            elif config.mpl_blocks_norm == TrajectoryDimensions.TEMPORAL:
                self.norm0 = CustomLayerNorm(-1, config.in_sequence_size)
            elif config.mpl_blocks_norm == TrajectoryDimensions.ALL:
                self.norm0 = nn.LayerNorm(
                    [config.in_points_dims, config.in_sequence_size]
                )
            elif config.mpl_blocks_norm == TrajectoryDimensions.BATCH:
                self.norm0 = nn.BatchNorm1d(config.in_points_dims)
            else:
                raise NotImplementedError(
                    f"{config.mpl_blocks_norm} is not a valid SiMLPe norm"
                )
        else:
            self.norm0 = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc0.fc.weight, gain=1e-8)
        nn.init.constant_(self.fc0.fc.bias, 0)

    def forward(self, x):
        x_ = self.fc0(x)
        x_ = self.norm0(x_)
        x = x + x_
        return x


class TransMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlps = nn.Sequential(*[MLPBlock(config) for i in range(config.num_layers)])

    def forward(self, x):
        x = self.mlps(x)
        return x
