"""
Simple Auto Regressive LSTM implementation,
for benchmark, example and tests of autoregressive method
inspired by pytorch Time Sequence prediction:
https://github.com/pytorch/examples/tree/main/time_sequence_prediction
"""
from typing import Dict, Optional
import torch
from torch import nn

from prescyent.dataset.features import (
    convert_tensor_features_to,
    features_are_convertible_to,
)
from prescyent.predictor.lightning.torch_module import BaseTorchModule
from prescyent.utils.logger import logger, PREDICTOR
from prescyent.utils.tensor_manipulation import self_auto_batch

from .config import SARLSTMConfig


class SARLSTMTorchModule(BaseTorchModule):
    """Torch implementation of a LSTM autoregressive model"""

    def __init__(self, config: SARLSTMConfig):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.num_in_features = self.num_in_dims * self.num_in_points
        self.num_out_features = self.num_out_dims * self.num_out_points
        if self.num_in_features != self.num_out_features:
            raise AttributeError(
                "We cannot use autoregressive models if we cannot recurse on the model output ! Please ajdust your input or chose another model"
            )
        self.lstms = nn.ModuleList(
            [nn.LSTMCell(self.num_in_features, self.hidden_size)]
        )
        self.lstms.extend(
            [
                nn.LSTMCell(self.hidden_size, self.hidden_size)
                for i in range(self.num_layers - 1)
            ]
        )
        self.linear = nn.Linear(self.hidden_size, self.num_out_features)
        self.convert_output = sorted(self.in_features) != sorted(self.out_features)

    @self_auto_batch
    @BaseTorchModule.deriv_tensor
    def forward(
        self,
        input_tensor: torch.Tensor,
        future_size: int = 1,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """lstm's autoregressive forward method

        Args:
            input_tensor (torch.Tensor): input traj_tensor
            future_size (int, optional): number of frames to predict as output. Defaults to 1.
            context (Optional[Dict[str, torch.Tensor]], optional): additionnal context to the trajectory.
            Note that there is no default implementation to integrate the context to the prediction. Defaults to None.

        Returns:
            torch.Tensor: predicted traj
        """
        if (
            self.convert_output
            and future_size > 1
            and not features_are_convertible_to(self.out_features, self.in_features)
        ):
            raise AttributeError(
                "Cannot output a future greater than 1 with this models features "
                f"as we cannot convert {self.out_features} to {self.in_features}"
                "and use our outputs as inputs"
            )
        if context is None:
            context = {}
        if context:
            logger.getChild(PREDICTOR).warning(
                "Context is not taken in account in SARLSTMPredictor's module"
            )
        # init the output
        predictions = []
        batch_size = input_tensor.shape[0]
        input_tensor = input_tensor.reshape(batch_size, self.in_sequence_size, -1)
        # input shape is (batch_size, seq_len, num_feature)
        # init the hidden states
        hs = [
            torch.zeros(batch_size, self.hidden_size, device=input_tensor.device)
            for i in range(self.num_layers)
        ]
        cs = [
            torch.zeros(batch_size, self.hidden_size, device=input_tensor.device)
            for i in range(self.num_layers)
        ]

        for input_frame in input_tensor.split(1, dim=1):
            # input_frame shape is (batch_size, 1, num_feature)
            # the lstmcell is called for each item of the sequence
            # we want (batch_size, 1, num_feature) => (batch_size, num_feature)
            next_input = torch.squeeze(input_frame, 1)
            for i in range(self.num_layers):
                hs[i], cs[i] = self.lstms[i](next_input, (hs[i], cs[i]))
                next_input = hs[i]
            prediction = self.linear(next_input)
            predictions.append(torch.unsqueeze(prediction, 1))

        for _ in range(future_size - 1):
            next_input = prediction
            if self.convert_output:
                next_input = convert_tensor_features_to(
                    next_input, self.in_features, self.out_features
                )
            for i in range(self.num_layers):
                hs[i], cs[i] = self.lstms[i](next_input, (hs[i], cs[i]))
                next_input = hs[i]
            prediction = self.linear(next_input)
            # reshape to (batch_size, 1, num_feature)
            predictions.append(torch.unsqueeze(prediction, 1))

        predictions = torch.cat(predictions, dim=1)
        predictions = predictions.reshape(
            batch_size,
            self.in_sequence_size + future_size - 1,
            self.num_out_points,
            self.num_out_dims,
        )
        return predictions
