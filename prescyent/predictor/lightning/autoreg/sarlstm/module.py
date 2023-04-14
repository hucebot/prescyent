"""
Simple Auto Regressive LSTM implementation
Inspired by: https://github.com/pytorch/examples/tree/main/time_sequence_prediction
"""
import torch
from torch import nn

from prescyent.predictor.lightning.module import BaseTorchModule


class TorchModule(BaseTorchModule):
    """
    feature_size - The number of dimensions to predict in parallel
    hidden_size - Can be chosen to dictate how much hidden "long term memory" the network will have
    """
    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.feature_size = config.feature_size

        self.lstm1 = nn.LSTMCell(self.feature_size, self.hidden_size)
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.feature_size)

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
        self.register_buffer("h1", torch.zeros(batch_size, self.hidden_size))
        self.register_buffer("c1", torch.zeros(batch_size, self.hidden_size))
        self.register_buffer("h2", torch.zeros(batch_size, self.hidden_size))
        self.register_buffer("c2", torch.zeros(batch_size, self.hidden_size))
        # h1 = torch.zeros(batch_size, self.hidden_size, device=input_tensor.device)
        # c1 = torch.zeros(batch_size, self.hidden_size, device=input_tensor.device)
        # h2 = torch.zeros(batch_size, self.hidden_size, device=input_tensor.device)
        # c2 = torch.zeros(batch_size, self.hidden_size, device=input_tensor.device)

        # encoding
        for input_frame in input_tensor.split(1, dim=1):
            # input_frame shape is (batch_size, 1, num_feature)
            # the lstmcell is called for each item of the sequence
            # we want (batch_size, 1, num_feature) => (batch_size, num_feature)
            input_frame = torch.squeeze(input_frame, 1)
            h1, c1 = self.lstm1(input_frame, (h1, c1))
            h2, c2 = self.lstm2(h1, (h2, c2))
            prediction = self.linear(h2)
            predictions.append(torch.unsqueeze(prediction, 1))

        # decoding
        # we loop over the layers for each output desired
        for _ in range(future_size - 1):
            h1, c1 = self.lstm1(prediction, (h1, c1))
            h2, c2 = self.lstm2(h1, (h2, c2))
            prediction = self.linear(h2)
            # reshape to (batch_size, 1, num_feature)
            predictions.append(torch.unsqueeze(prediction, 1))

        predictions = torch.cat(predictions, dim=1)
        predictions = predictions.reshape(predictions.shape[0],
                                          predictions.shape[1],
                                          T[2],
                                          T[3])
        return predictions
