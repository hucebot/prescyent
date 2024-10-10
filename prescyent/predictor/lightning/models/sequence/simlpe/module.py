"""
simple MLP implementation
[This is a basic multi-layer perceptron, with configurable hidden layers and activation function]
"""
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn

from prescyent.dataset.features import (
    convert_tensor_features_to,
    features_are_convertible_to,
)
from prescyent.predictor.lightning.layers.transpose_layer import TransposeLayer
from prescyent.predictor.lightning.torch_module import BaseTorchModule
from prescyent.utils.tensor_manipulation import self_auto_batch
from .mlp import TransMLP


class TorchModule(BaseTorchModule):
    """siMLPe implementation"""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        if self.config.out_sequence_size > self.config.in_sequence_size:
            raise AttributeError(
                "This model cannot be used with an output sequence bigger than input's sequence size"
            )
        self.motion_mlp = TransMLP(self.config)
        # Configure DCT
        if self.config.dct:
            if not features_are_convertible_to(self.in_features, self.out_features):
                raise AttributeError(
                    "We cannot apply DCT with non matching in and out features"
                )
            dct_m, idct_m = get_dct_matrix(self.in_sequence_size)
            self.register_buffer(
                "dct_m", torch.tensor(dct_m, requires_grad=False).float().unsqueeze(0)
            )
            self.register_buffer(
                "idct_m", torch.tensor(idct_m, requires_grad=False).float().unsqueeze(0)
            )
        # Configure in/out feats according to chosen FC
        if self.config.temporal_fc_in:
            self.motion_fc_in = nn.Linear(
                self.config.in_sequence_size, self.config.in_sequence_size
            )
        else:
            self.motion_fc_in = nn.Linear(
                self.config.in_points_dims, self.config.in_points_dims
            )
        if self.config.temporal_fc_out:
            self.motion_fc_out = nn.Linear(
                self.config.in_sequence_size, self.config.in_sequence_size
            )
            if self.config.in_points_dims != self.config.out_points_dims:
                self.motion_fc_out = nn.Sequential(
                    TransposeLayer(1, 2),
                    nn.Linear(self.config.in_points_dims, self.config.out_points_dims),
                    TransposeLayer(1, 2),
                    self.motion_fc_out,
                )
        else:
            self.motion_fc_out = nn.Linear(
                self.config.in_points_dims, self.config.out_points_dims
            )
        self.reset_parameters()

    def reset_parameters(self):
        try:
            nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
            nn.init.constant_(self.motion_fc_out.bias, 0)
        except AttributeError:  # If motion_fc_out is nn.Sequential
            for layer in self.motion_fc_out.children():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=1e-8)
                    nn.init.constant_(layer.bias, 0)

    @self_auto_batch
    @BaseTorchModule.deriv_tensor
    def forward(
        self,
        input_tensor: torch.Tensor,
        future_size: int = None,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ):
        batch_size = input_tensor.shape[0]
        # (batch_size, seq_len, num_point, num_dim) => (batch_size, seq_len, num_point * num_dim)
        input_tensor_ = input_tensor.reshape(batch_size, self.in_sequence_size, -1)
        if self.config.dct:
            input_tensor_ = torch.matmul(
                self.dct_m[:, :, : self.in_sequence_size], input_tensor_
            )
        if self.config.temporal_fc_in:
            input_tensor_ = torch.transpose(input_tensor_, 1, 2)
            motion_feats = self.motion_fc_in(input_tensor_)
        else:
            motion_feats = self.motion_fc_in(input_tensor_)
            motion_feats = torch.transpose(motion_feats, 1, 2)
        # (batch_size, num_point * num_dim, seq_len)
        motion_pred = self.motion_mlp(motion_feats)
        if self.config.temporal_fc_out:
            motion_pred = self.motion_fc_out(motion_pred)
            motion_pred = torch.transpose(motion_pred, 1, 2)
        else:
            motion_pred = torch.transpose(motion_pred, 1, 2)
            motion_pred = self.motion_fc_out(motion_pred)
        if self.config.dct:
            motion_pred = torch.matmul(
                self.idct_m[:, : self.in_sequence_size, :], motion_pred
            )
        motion_pred = motion_pred[:, : self.out_sequence_size]
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
