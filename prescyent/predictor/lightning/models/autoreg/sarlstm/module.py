"""
Simple Auto Regressive LSTM implementation,
for benchmark, example and tests of autoregressive method
inspired by pytorch Time Sequence prediction:
https://github.com/pytorch/examples/tree/main/time_sequence_prediction
"""
import torch
from torch import nn

from prescyent.predictor.lightning.torch_module import BaseTorchModule


class TorchModule(BaseTorchModule):
    """
    feature_size - The number of dimensions to predict in parallel
    hidden_size - Can be chosen to dictate how much hidden "long term memory" the network will have
    """

    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.in_features = self.num_in_dims * self.num_in_points
        self.out_features = self.num_out_dims * self.num_out_points
        self.lstms = nn.ModuleList([nn.LSTMCell(self.in_features, self.hidden_size)])
        self.lstms.extend(
            [
                nn.LSTMCell(self.hidden_size, self.hidden_size)
                for i in range(self.num_layers - 1)
            ]
        )
        self.linear = nn.Linear(self.hidden_size, self.out_features)

    @BaseTorchModule.allow_unbatched
    @BaseTorchModule.normalize_tensor
    def forward(self, input_tensor: torch.Tensor, future_size: int = 1):
        # init the output
        predictions = []
        T = input_tensor.shape
        input_tensor = input_tensor.reshape(T[0], T[1], -1)
        # input shape is (batch_size, seq_len, num_feature)
        batch_size = T[0]
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
            for i in range(self.num_layers):
                hs[i], cs[i] = self.lstms[i](next_input, (hs[i], cs[i]))
                next_input = hs[i]
            prediction = self.linear(next_input)
            # reshape to (batch_size, 1, num_feature)
            predictions.append(torch.unsqueeze(prediction, 1))

        predictions = torch.cat(predictions, dim=1)
        predictions = predictions.reshape(
            predictions.shape[0],
            predictions.shape[1],
            self.num_out_points,
            self.num_out_dims,
        )
        return predictions
