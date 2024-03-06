"""
simple MLP implementation
[This is a basic multi-layer perceptron, with configurable hidden layers and activation function]
"""
import numpy as np
import torch
from torch import nn

from prescyent.predictor.lightning.torch_module import BaseTorchModule
from prescyent.dataset.features import (
    convert_tensor_features_to,
    features_are_convertible_to,
)
from .mlp import TransMLP


class TorchModule(BaseTorchModule):
    """siMLPe implementation"""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.spatial_fc_only = config.spatial_fc_only
        if self.config.dct:
            if not features_are_convertible_to(self.in_features, self.out_features):
                raise AttributeError(
                    "We cannot apply DCT with non matching in and out features"
                )
            dct_m, idct_m = get_dct_matrix(self.config.in_sequence_size)
            self.register_buffer(
                "dct_m", torch.tensor(dct_m, requires_grad=False).float().unsqueeze(0)
            )
        if self.spatial_fc_only:
            self.motion_fc_in = nn.Linear(
                self.config.in_points_dims, self.config.hidden_size
            )
        else:
            self.motion_fc_in = nn.Linear(
                self.config.in_sequence_size, self.config.hidden_size
            )
        self.motion_mlp = TransMLP(self.config)
        if self.spatial_fc_only:
            if self.out_sequence_size > self.in_sequence_size:
                raise NotImplementedError(
                    "This model cannot output a sequence bigger than its"
                    " input without the spatial_fc_only configuration"
                )
            self.motion_fc_out = nn.Linear(
                self.config.hidden_size, self.config.out_points_dims
            )
        else:
            if self.config.out_points_dims > self.config.in_points_dims:
                raise NotImplementedError(
                    "This model cannot output feature dimensions bigger than its"
                    " input feature size with the spatial_fc_only configuration"
                )
            self.motion_fc_out = nn.Linear(
                self.config.hidden_size, self.config.out_sequence_size
            )
        if self.config.dct:
            self.register_buffer(
                "idct_m", torch.tensor(idct_m, requires_grad=False).float().unsqueeze(0)
            )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)

    @BaseTorchModule.allow_unbatched
    def forward(self, input_tensor: torch.Tensor, future_size: int = None):
        batch_size = input_tensor.shape[0]
        # (batch_size, seq_len, num_point, num_dim) => (batch_size, seq_len, num_point * num_dim)
        input_tensor = input_tensor.reshape(batch_size, self.in_sequence_size, -1)
        if self.config.dct:
            input_tensor_ = torch.matmul(
                self.dct_m[:, :, : self.in_sequence_size], input_tensor
            )
        else:
            input_tensor_ = input_tensor.clone().to(input_tensor.device)
        if self.spatial_fc_only:
            motion_feats = self.motion_fc_in(input_tensor_)
            motion_feats = torch.transpose(motion_feats, 1, 2)
        else:
            motion_feats = torch.transpose(input_tensor_, 1, 2)
            motion_feats = self.motion_fc_in(motion_feats)
        motion_feats = self.motion_mlp(motion_feats)
        if self.spatial_fc_only:
            motion_feats = torch.transpose(motion_feats, 1, 2)
            motion_feats = self.motion_fc_out(motion_feats)
        else:
            motion_feats = self.motion_fc_out(motion_feats)
            motion_feats = torch.transpose(motion_feats, 1, 2)

        if self.config.dct:
            motion_feats = torch.matmul(
                self.idct_m[:, : self.in_sequence_size, :], motion_feats
            )
            offset = input_tensor[:, -1:].to(motion_feats.device)
            offset = convert_tensor_features_to(
                offset, self.in_features, self.out_features
            )
            motion_feats = motion_feats[:, : self.out_sequence_size] + offset
        motion_pred = motion_feats[:, : self.out_sequence_size]
        motion_pred = motion_pred.reshape(
            batch_size, self.out_sequence_size, self.num_out_points, self.num_out_dims
        )
        return motion_pred


def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m
