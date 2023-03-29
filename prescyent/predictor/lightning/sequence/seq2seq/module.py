"""
simple Seq2Seq implementation
[short description]
[link to the paper]
"""
import torch
from torch import nn

from prescyent.predictor.lightning.module import BaseTorchModule


class TorchModule(BaseTorchModule):
    """
    feature_size - The number of dimensions to predict in parrallel
    hidden_size - Can be chosen to dictate how much hidden "long term memory" the network will have
    output_size - This will be equal to the prediction_periods input to get_x_y_pairs
    """
    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.feature_size = config.feature_size
        self.output_size = config.output_size
        self.num_layers = config.num_layers

        self.encoder = nn.LSTM(input_size=self.feature_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               dropout=0)
        self.decoder = nn.LSTM(input_size=self.feature_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               dropout=0)
        self.linear = nn.Linear(self.hidden_size, self.feature_size)

    @BaseTorchModule.allow_unbatched
    @BaseTorchModule.normalize_tensor
    def forward(self, input_tensor: torch.Tensor, future_size: int = None):
        T = input_tensor.shape
        # (batch_size, seq_len, num_point, num_dim) => (seq_len, batch_size, num_point * num_dim)
        batch_size = T[0]
        input_tensor = input_tensor.reshape(T[0], T[1], -1)
        input_tensor = torch.transpose(input_tensor, 0, 1)
        # take hidden state as the encoding of the whole input sequence
        _, hidden_state = self.encoder(input_tensor)
        # we take as input for the decoder the last input form the input sample
        dec_input = input_tensor[-1].unsqueeze(0)
        # we prepare the output tensor that will be fed by the decoding loop
        predictions = torch.zeros(self.output_size, batch_size,
                                  self.feature_size, device=input_tensor.device)
        # decoding loop must update the hidden state and input for each wanted output
        for i in range(self.output_size):
            dec_output, hidden_state = self.decoder(dec_input, hidden_state)
            prediction = self.linear(dec_output)
            dec_input = prediction
            predictions[i] = prediction
        # (seq_len, batch_size, num_point * num_dim) => (batch_size, seq_len, num_point, num_dim)
        predictions = torch.transpose(predictions, 0, 1)
        predictions = predictions.reshape(T[0], -1, T[2], T[3])
        return predictions
