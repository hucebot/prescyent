"""
simple Linear implementation
[short description]
[link to the paper]
"""
import torch
from torch import nn

from prescyent.predictor.lightning.module import BaseTorchModule


class TorchModule(BaseTorchModule):
    """Simple linear layer with flatten input"""
    def __init__(self, config):
        super().__init__(config)
        self.input_size = config.input_size
        self.output_size = config.output_size

        self.linear = nn.Linear(self.input_size, self.output_size)

    @BaseTorchModule.allow_unbatched
    @BaseTorchModule.normalize_tensor_from_last_value
    def forward(self, input_tensor: torch.Tensor, future_size: int = None):
        # simple single feature prediction of the next item in sequence
        # (batch, seq_len, features) -> (batch, features, seq_len)
        input_tensor = torch.transpose(input_tensor, 1, 2)
        predictions = self.linear(input_tensor)
        predictions = torch.transpose(predictions, 1, 2)
        return predictions
