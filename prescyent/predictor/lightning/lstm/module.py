"""
simple LSTM implementation
[short description]
[link to the paper]
"""
import torch
from torch import nn

from prescyent.predictor.lightning.module import BaseLightningModule, allow_unbatched


class TorchModule(nn.Module):
    """
    feature_size - The number of dimensions to predict in parrallel
    hidden_size - Can be chosen to dictate how much hidden "long term memory" the network will have
    output_size - This will be equal to the prediction_periods input to get_x_y_pairs
    """
    def __init__(self, feature_size: int, hidden_size: int, output_size: int, num_layers: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.output_size = output_size
        self.num_layers = num_layers

        # input -> output
        # batch_first=False : (seq_len, batch_size, features) -> (seq_len, batch_size, hidden_size).
        # batch_first=True: (batch_size, seq_len, features) -> (batch_size, seq_len, hidden_size).
        # unbatched: (seq_length, feature_size) -> (seq_len, hidden_size)
        self.lstm = nn.LSTM(input_size=feature_size,
                            hidden_size=hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=0)

        # we use the sequence of all the hidden state to predict the output
        # linear expect [batch_size, *, nb_features]
        self.linear = nn.Linear(hidden_size, feature_size)

    @allow_unbatched
    def forward(self, input_tensor: torch.Tensor):
        """
        inputs need to be in the right shape as defined in documentation
        - https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

        lstm_out - will contain the hidden states from all times in the sequence
        hidden - will contain the current hidden state and cell state, it is ommited here
        """
        lstm_out, _ = self.lstm(input_tensor)
        predictions = self.linear(lstm_out)
        return predictions


class LightningModule(BaseLightningModule):
    """[short description]
       [usage]
       [detail of the implementation]
    """
    def __init__(self, feature_size: int, hidden_size: int, output_size: int, num_layers: int):
        super().__init__()
        self.torch_model = TorchModule(feature_size, hidden_size, output_size, num_layers)
        self.criterion = nn.MSELoss()
        self.save_hyperparameters()

    @classmethod
    def load_from_binary(cls, path: str):
        """Retrieve model infos from torch binary"""
        model = torch.load(path)
        lstm_module = cls(model.feature_size, model.hidden_size,
                          model.output_size, model.num_layers)
        lstm_module.torch_model = model
        return lstm_module
