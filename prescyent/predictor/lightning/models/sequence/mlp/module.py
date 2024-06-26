"""
simple MLP implementation
[This is a basic multi-layer perceptron, with configurable hidden layers and activation function]
"""
import torch
from torch import nn

from prescyent.predictor.lightning.torch_module import BaseTorchModule
from prescyent.utils.enums import ActivationFunctions
from prescyent.utils.logger import logger, PREDICTOR


class TorchModule(BaseTorchModule):
    """Simple Multi-Layer Perceptron with flatten input"""

    def __init__(self, config):
        super().__init__(config)
        self.mlps_in_size = (
            self.in_sequence_size * self.num_in_dims * self.num_in_points
        )
        self.mlps_out_size = (
            self.out_sequence_size * self.num_out_dims * self.num_out_points
        )

        # select the activation function
        if config.activation == ActivationFunctions.RELU:
            act_fun = nn.ReLU
        elif config.activation == ActivationFunctions.GELU:
            act_fun = nn.GELU
        elif config.activation == ActivationFunctions.SIGMOID:
            act_fun = nn.Sigmoid
        else:
            logger.getChild(PREDICTOR).info(
                "No activation function for: %s" % config.activation,
            )
            act_fun = nn.Identity

        # create the layers
        if config.num_layers == 1:
            self.layers = nn.Linear(self.mlps_in_size, self.mlps_out_size)
        else:
            layers = [nn.Linear(self.mlps_in_size, config.hidden_size)]
            layers += [act_fun()]
            for _ in range(0, config.num_layers - 2):
                layers += [nn.Linear(config.hidden_size, config.hidden_size)]
                layers += [act_fun()]
            layers += [nn.Linear(config.hidden_size, self.mlps_out_size)]
            self.layers = nn.Sequential(*layers)

    @BaseTorchModule.allow_unbatched
    @BaseTorchModule.normalize_tensor
    def forward(self, input_tensor: torch.Tensor, future_size: int = None):
        # simple single feature prediction of the next item in sequence
        batch_size = input_tensor.shape[0]
        # (batch, seq_len, num_point, num_dim) -> (batch, num_point * num_dim * seq_len)
        input_tensor = torch.reshape(input_tensor, (batch_size, -1))
        prediction = self.layers(input_tensor)
        output_tensor = torch.reshape(
            prediction,
            (
                batch_size,
                self.out_sequence_size,
                self.num_out_points,
                self.num_out_dims,
            ),
        )
        return output_tensor
