"""
simple Seq2Seq implementation
[short description]
[link to the paper]
"""
import torch
from torch import nn

from prescyent.predictor.lightning.module import BaseLightningModule, allow_unbatched


class Seq2Seq(nn.Module):
    """
    feature_size - The number of dimensions to predict in parrallel
    hidden_size - Can be chosen to dictate how much hidden "long term memory" the network will have
    output_size - This will be equal to the prediction_periods input to get_x_y_pairs
    """
    def __init__(self, feature_size, hidden_size, output_size, num_layers):
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(input_size=feature_size,
                               hidden_size=hidden_size,
                               num_layers=self.num_layers,
                               batch_first=True,
                               dropout=0)
        self.decoder = nn.LSTM(input_size=feature_size,
                               hidden_size=hidden_size,
                               num_layers=self.num_layers,
                               dropout=0)
        self.linear = nn.Linear(hidden_size, feature_size)

    @allow_unbatched
    def forward(self, input_tensor):
        # take hidden state as the encoding of the whole input sequence
        _, hidden_state = self.encoder(input_tensor)
        batch_size = input_tensor.shape[0]
        # (batch_size, seq_len, features) => (seq_len, batch_size, features)
        input_tensor = torch.transpose(input_tensor, 0, 1)
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
        # (seq_len, batch_size, features) => (batch_size, seq_len, features)
        predictions = torch.transpose(predictions, 0, 1)
        return predictions


class Seq2SeqModule(BaseLightningModule):
    """[short description]
       [usage]
       [detail of the implementation]
    """
    def __init__(self, feature_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.torch_model = Seq2Seq(feature_size, hidden_size, output_size, num_layers)
        self.criterion = nn.MSELoss()
        self.save_hyperparameters()

    @classmethod
    def load_from_binary(cls, path: str):
        """Retrieve model infos from torch binary"""
        model = torch.load(path)
        seq2seq_module = cls(model.input_size, model.hidden_size,
                             model.output_size, model.num_layers)
        seq2seq_module.torch_model = model
        return seq2seq_module
