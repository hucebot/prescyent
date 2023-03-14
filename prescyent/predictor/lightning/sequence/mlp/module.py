"""
simple MLP implementation
[This is a basic multi-layer perceptron, with configurable hidden layers and activation function]
"""
import torch
from torch import nn

from prescyent.predictor.lightning.module import BaseTorchModule


class TorchModule(BaseTorchModule):
    """Simple Multi-Layer Perceptron with flatten input"""
    def __init__(self, config):
        super().__init__(config)
        self.input_size = config.input_size
        self.output_size = config.output_size

        # select the activation function
        if config.activation == 'ReLu':
            act_fun = nn.ReLU
        elif config.activation == 'Sigmoid':
            act_fun = nn.Sigmoid
        else:
            print('ERROR: no activation function:', config.activation)
            act_fun = None

        # create the layers
        layers = [nn.Linear(config.input_size, config.hidden_size[0])]
        layers += [act_fun()]
        for i in range(0, len(config.hidden_size) - 1):
            layers +=  [nn.Linear(config.hidden_size[i], config.hidden_size[i + 1])]
            layers += [act_fun()]
        layers +=  [nn.Linear(config.hidden_size[-1], config.output_size)]
        self.layers = nn.Sequential(*layers)

    @BaseTorchModule.allow_unbatched
    @BaseTorchModule.normalize_tensor_from_last_value
    def forward(self, input_tensor: torch.Tensor, future_size: int = None):
        # simple single feature prediction of the next item in sequence
        # (batch, seq_len, num_point, num_dim) -> (batch, num_point, num_dim, seq_len)
        T = input_tensor.shape
        input_tensor = torch.transpose(torch.transpose(input_tensor, 1, 2), 2, 3)
        predictions = self.layers(input_tensor)
        predictions = torch.transpose(torch.transpose(predictions, 2, 3), 1, 2)
        return predictions
