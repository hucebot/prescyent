"""
simple MLP implementation
[This is a basic multi-layer perceptron, with configurable hidden layers and activation function]
"""
import torch
from torch import nn

from prescyent.predictor.lightning.module import BaseTorchModule
from .mlp import build_mlps

class TorchModule(BaseTorchModule):
    """siMLPe implementation"""
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.input_size = config.input_size
        self.motion_mlp = build_mlps(self.config)

        self.temporal_fc_in = config.motion_fc_in.temporal_fc
        self.temporal_fc_out = config.motion_fc_out.temporal_fc
        if self.temporal_fc_in:
            self.motion_fc_in = nn.Linear(self.config.input_size_dct, self.config.input_size_dct)
        else:
            self.motion_fc_in = nn.Linear(self.config.feature_size, self.config.feature_size)
        if self.temporal_fc_out:
            self.motion_fc_out = nn.Linear(self.config.input_size_dct, self.config.input_size_dct)
        else:
            self.motion_fc_out = nn.Linear(self.config.feature_size, self.config.feature_size)

        self.reset_parameters()

    @property
    def output_size(self):
        return self.input_size

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)

    @BaseTorchModule.allow_unbatched
    def forward(self, input_tensor: torch.Tensor, future_size: int = None):
        T = input_tensor.shape
        input_tensor = input_tensor.reshape(T[0], T[1], -1)
        if self.temporal_fc_in:
            motion_feats = torch.transpose(input_tensor, 1, 2)
            motion_feats = self.motion_fc_in(motion_feats)
        else:
            motion_feats = self.motion_fc_in(input_tensor)
            motion_feats = torch.transpose(motion_feats, 1, 2)

        motion_feats = self.motion_mlp(motion_feats)

        if self.temporal_fc_out:
            motion_feats = self.motion_fc_out(motion_feats)
            motion_feats = torch.transpose(motion_feats, 1, 2)
        else:
            motion_feats = torch.transpose(motion_feats, 1, 2)
            motion_feats = self.motion_fc_out(motion_feats)
        motion_feats = motion_feats.reshape(T)
        return motion_feats
