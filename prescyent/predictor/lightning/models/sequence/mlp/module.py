"""
simple MLP implementation
[This is a basic multi-layer perceptron, with configurable hidden layers and activation function]
"""
from typing import Dict, Optional
import torch
from torch import nn

from prescyent.predictor.lightning.torch_module import BaseTorchModule
from prescyent.utils.enums import ActivationFunctions
from prescyent.utils.logger import logger, PREDICTOR
from prescyent.utils.tensor_manipulation import self_auto_batch
from .config import Config as MlpConfig


class TorchModule(BaseTorchModule):
    """Simple Multi-Layer Perceptron with flatten input"""

    def __init__(self, config: MlpConfig):
        super().__init__(config)
        if config.context_size is not None:
            self.mlps_in_size = self.in_sequence_size * (
                config.context_size + self.num_in_dims * self.num_in_points
            )
        else:
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

    @self_auto_batch
    @BaseTorchModule.deriv_tensor
    def forward(
        self,
        input_tensor: torch.Tensor,
        future_size: int = None,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ):
        # simple single feature prediction of the next item in sequence
        batch_size, sequence_size = input_tensor.shape[0:2]
        input_tensor = torch.reshape(input_tensor, (batch_size, sequence_size, -1))
        if (
            context
        ):  # Cat context and input with (batch, seq_len, in_feat + context_feats)
            context_tensor = torch.cat(
                [context[c_key] for c_key in context.keys()], dim=2
            )
            input_tensor = torch.cat((input_tensor, context_tensor), dim=2)
        # (batch, seq_len, num_point, num_dim) -> (batch, seq_len * num_point * num_dim)
        input_tensor = torch.reshape(input_tensor, (batch_size, self.mlps_in_size))
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
