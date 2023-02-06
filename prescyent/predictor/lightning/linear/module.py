"""
simple Linear implementation
[short description]
[link to the paper]
"""
import torch
from torch import nn

from prescyent.predictor.lightning.module import BaseLightningModule, allow_unbatched


class Linear(nn.Module):
    """Simple linear layer with flatten input"""
    def __init__(self, feature_size, input_size, output_size):
        super(Linear, self).__init__()
        self.feature_size = feature_size
        self.input_size = input_size
        self.output_size = output_size

        self.linear = nn.Linear(input_size, output_size)

    @allow_unbatched
    def forward(self, x):
        # simple single feature prediction of the next item in sequence
        # (batch, seq_len, features) -> (batch, features, seq_len)
        x = torch.transpose(x, 1, 2)
        predictions = self.linear(x)
        predictions = torch.transpose(predictions, 1, 2)
        return predictions


class LinearModule(BaseLightningModule):
    """Lightning Module initializing Linear NN"""
    def __init__(self, feature_size, input_size, output_size):
        super().__init__()
        self.torch_model = Linear(feature_size, input_size, output_size)
        self.criterion = nn.MSELoss()
        self.save_hyperparameters()

    @classmethod
    def load_from_binary(cls, path: str):
        """Retrieve model infos from torch binary"""
        model = torch.load(path)
        linear_module = cls(model.feature_size, model.input_size, model.output_size)
        linear_module.torch_model = model
        return linear_module
