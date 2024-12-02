"""
simple MLP implementation
[This is a basic multi-layer perceptron, with configurable hidden layers and activation function]
"""
from typing import Dict, Optional
import torch
from torch import nn

from prescyent.predictor.lightning.torch_module import BaseTorchModule
from prescyent.utils.enums import ActivationFunctions
from prescyent.utils.tensor_manipulation import self_auto_batch

from .config import MlpConfig


ACT_FUNCTION_MAP = {
    ActivationFunctions.RELU: nn.ReLU,
    ActivationFunctions.GELU: nn.GELU,
    ActivationFunctions.SIGMOID: nn.Sigmoid,
}


class MlpTorchModule(BaseTorchModule):
    """Simple Multi-Layer Perceptron with flatten input"""

    def __init__(self, config: MlpConfig):
        super().__init__(config)
        # If we have a context, add its size to the feature size
        if config.context_size is not None:
            self.mlps_in_size = self.in_sequence_size * (
                config.context_size + self.num_in_dims * self.num_in_points
            )
        # Else feature size is points * dims
        else:
            self.mlps_in_size = (
                self.in_sequence_size * self.num_in_dims * self.num_in_points
            )
        self.mlps_out_size = (
            self.out_sequence_size * self.num_out_dims * self.num_out_points
        )

        # select the activation function, default is identity function
        act_fun = ACT_FUNCTION_MAP.get(config.activation, nn.Identity)
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
        future_size: Optional[int] = None,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """mlp's forward method

        Args:
            input_tensor (torch.Tensor): input traj_tensor
            future_size (int, optional): number of frames to predict as output. Uused in that .
            context (Optional[Dict[str, torch.Tensor]], optional): additionnal context to the trajectory.
                Default behavior is adding all of contexts to input tensor's features. Defaults to None.

        Returns:
            torch.Tensor: predicted traj
        """
        if future_size is None:
            future_size = self.out_sequence_size
        elif future_size > self.out_sequence_size:
            raise AttributeError(
                f"module cannot output a future bigger than its configured future_size {self.out_sequence_size}"
            )
        if context is None:
            context = {}
        # simple single feature prediction of the next item in sequence
        batch_size, sequence_size = input_tensor.shape[0:2]
        input_tensor = torch.reshape(input_tensor, (batch_size, sequence_size, -1))
        if context:
            # Cat context and input with (batch, seq_len, in_feat + context_feats)
            context_tensor = torch.cat(
                [
                    context[c_key].reshape(batch_size, sequence_size, -1)
                    for c_key in context.keys()
                ],
                dim=2,
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
        return output_tensor[:, -future_size:]
