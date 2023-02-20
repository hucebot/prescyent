"""
Simple Auto Regressive LSTM implementation
Inspired by: https://github.com/pytorch/examples/tree/main/time_sequence_prediction
"""
import torch
from torch import nn

from prescyent.predictor.lightning.module import BaseLightningModule, allow_unbatched


class TorchModule(nn.Module):
    """
    feature_size - The number of dimensions to predict in parrallel
    hidden_size - Can be chosen to dictate how much hidden "long term memory" the network will have
    """
    def __init__(self, feature_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.feature_size = feature_size

        self.lstm1 = nn.LSTMCell(self.feature_size, self.hidden_size)
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(hidden_size, feature_size)

    @allow_unbatched
    def forward(self, input_tensor: torch.Tensor, future: int = 1):
        # init the output
        predictions = []
        # input shape is (batch_size, seq_len, num_feature)
        batch_size = input_tensor.shape[0]
        # init the hidden states
        h1 = torch.zeros(batch_size, self.hidden_size, device=input_tensor.device)
        c1 = torch.zeros(batch_size, self.hidden_size, device=input_tensor.device)
        h2 = torch.zeros(batch_size, self.hidden_size, device=input_tensor.device)
        c2 = torch.zeros(batch_size, self.hidden_size, device=input_tensor.device)

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
        for _ in range(future - 1):
            h1, c1 = self.lstm1(prediction, (h1, c1))
            h2, c2 = self.lstm2(h1, (h2, c2))
            prediction = self.linear(h2)
            # reshape to (batch_size, 1, num_feature)
            predictions.append(torch.unsqueeze(prediction, 1))

        predictions = torch.cat(predictions, dim=1)
        return predictions


class LightningModule(BaseLightningModule):
    """pl module for the simple ar lstm implementation"""
    def __init__(self, feature_size: int, hidden_size: int):
        super().__init__()
        self.torch_model = TorchModule(feature_size, hidden_size)
        self.criterion = nn.MSELoss()
        self.save_hyperparameters()

    @classmethod
    def load_from_binary(cls, path: str):
        """Retrieve model infos from torch binary"""
        model = torch.load(path)
        lstm_module = cls(model.feature_size, model.hidden_size)
        lstm_module.torch_model = model
        return lstm_module
