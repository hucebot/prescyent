"""
simple Seq2Seq implementation
[short description]
[link to the paper]
"""
from typing import Dict, Optional
import torch
from torch import nn

from prescyent.predictor.lightning.torch_module import BaseTorchModule
from prescyent.utils.tensor_manipulation import self_auto_batch


class TorchModule(BaseTorchModule):
    """
    feature_size - The number of dimensions to predict in parrallel
    hidden_size - Can be chosen to dictate how much hidden "long term memory" the network will have
    out_sequence_size - This will be equal to the prediction_periods input to get_x_y_pairs
    """

    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.dropout_value = config.dropout_value if config.dropout_value else 0
        self.num_in_features = self.num_in_dims * self.num_in_points
        self.num_out_features = self.num_out_dims * self.num_out_points

        self.encoder = nn.GRU(
            input_size=self.num_in_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_value,
        )
        self.decoder = nn.GRU(
            input_size=self.num_in_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_value,
        )
        self.linear = nn.Linear(self.hidden_size, self.num_out_features)
        # If input sequence differs from the predicted
        if self.num_in_features != self.num_out_features:
            # We predict the next input sequence alongside what we want as output
            self.in_decoder_linear = nn.Linear(self.hidden_size, self.num_in_features)

    @self_auto_batch
    @BaseTorchModule.deriv_tensor
    def forward(
        self,
        input_tensor: torch.Tensor,
        future_size: int = None,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ):
        # (batch_size, seq_len, num_point, num_dim) => (seq_len, batch_size, num_point * num_dim)
        batch_size = input_tensor.shape[0]
        input_tensor = input_tensor.reshape(batch_size, self.in_sequence_size, -1)
        input_tensor = input_tensor.transpose(0, 1)
        # take hidden state as the encoding of the whole input sequence
        _, hidden_state = self.encoder(input_tensor)
        # we take as input for the decoder the last input form the input sample
        dec_input = input_tensor[-1].unsqueeze(0)
        # we prepare the output tensor that will be fed by the decoding loop
        predictions = torch.zeros(
            self.out_sequence_size,
            batch_size,
            self.num_out_features,
            device=input_tensor.device,
        )
        # decoding loop must update the hidden state and input for each wanted output
        for i in range(self.out_sequence_size):
            dec_output, hidden_state = self.decoder(dec_input, hidden_state)
            prediction = self.linear(dec_output)
            if self.num_in_features != self.num_out_features:
                dec_input = self.in_decoder_linear(dec_output)
            else:
                dec_input = prediction
            predictions[i] = prediction
        # (seq_len, batch_size, num_point * num_dim) => (batch_size, seq_len, num_point, num_dim)
        predictions = predictions.transpose(0, 1)
        predictions = predictions.reshape(
            batch_size, self.out_sequence_size, self.num_out_points, self.num_out_dims
        )
        return predictions
