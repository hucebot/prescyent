"""
simple MLP implementation
[This is a basic multi-layer perceptron, with configurable hidden layers and activation function]
"""
import numpy as np
import torch
from torch import nn

from prescyent.predictor.lightning.torch_module import BaseTorchModule
from .mlp import TransMLP


class TorchModule(BaseTorchModule):
    """siMLPe implementation"""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.hidden_size = config.hidden_size
        self.temporal_fc_in = config.temporal_fc_in
        self.temporal_fc_out = config.temporal_fc_out
        if self.config.dct:
            dct_m, idct_m = get_dct_matrix(self.config.input_size)
            self.register_buffer(
                "dct_m", torch.tensor(dct_m, requires_grad=False).float().unsqueeze(0)
            )
        if self.temporal_fc_in:
            self.motion_fc_in = nn.Linear(
                self.config.input_size, self.config.hidden_size
            )
        else:
            self.motion_fc_in = nn.Linear(
                self.config.feature_size, self.config.hidden_size
            )
        self.motion_mlp = TransMLP(self.config)
        if self.temporal_fc_out:
            self.motion_fc_out = nn.Linear(
                self.config.hidden_size, self.config.output_size
            )
        else:
            if self.output_size > self.input_size:
                raise NotImplementedError(
                    "This model cannot output a sequence bigger than its"
                    " input without the temporal_fc_out configuration"
                )
            self.motion_fc_out = nn.Linear(
                self.config.hidden_size, self.config.feature_size
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
        T = input_tensor.shape
        input_tensor = input_tensor.reshape(T[0], T[1], -1)
        if self.config.dct:
            input_tensor_ = torch.matmul(
                self.dct_m[:, :, : self.input_size], input_tensor
            )
        else:
            input_tensor_ = input_tensor.clone().to(input_tensor.device)
        if self.temporal_fc_in:
            motion_feats = torch.transpose(input_tensor_, 1, 2)
            motion_feats = self.motion_fc_in(motion_feats)
        else:
            motion_feats = self.motion_fc_in(input_tensor_)
            motion_feats = torch.transpose(motion_feats, 1, 2)
        motion_feats = self.motion_mlp(motion_feats)
        if self.temporal_fc_out:
            motion_feats = self.motion_fc_out(motion_feats)
            motion_feats = torch.transpose(motion_feats, 1, 2)
        else:
            motion_feats = torch.transpose(motion_feats, 1, 2)
            motion_feats = self.motion_fc_out(motion_feats)

        if self.config.dct:
            motion_feats = torch.matmul(
                self.idct_m[:, : self.input_size, :], motion_feats
            )
            offset = input_tensor[:, -1:].to(motion_feats.device)
            motion_feats = motion_feats[:, : self.output_size] + offset
        motion_pred = motion_feats[:, : self.output_size]
        motion_pred = motion_pred.reshape(T[0], self.output_size, T[2], T[3])
        return motion_pred

    def criterion(self, motion_pred, motion_truth):
        dimensions = motion_truth.shape[-1]
        pred = motion_pred.reshape(-1, dimensions)
        truth = motion_truth.reshape(-1, dimensions)
        return torch.mean(torch.norm(pred - truth, 2, 1))


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
