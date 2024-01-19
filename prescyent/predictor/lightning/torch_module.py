from abc import abstractmethod
import copy
import functools
import torch

from prescyent.dataset.features import convert_tensor_features_to
from prescyent.predictor.lightning.layers.normalization_layer import MotionLayerNorm
from prescyent.predictor.lightning.layers.relative_norm import get_absolute_tensor_from, get_relative_tensor_from


class BaseTorchModule(torch.nn.Module):
    """Common methods for any torch module to be a lightning predictor"""

    def __init__(self, config) -> None:
        super().__init__()
        self.norm_on_last_input = config.norm_on_last_input
        self.used_norm = config.used_norm
        self.dropout_value = config.dropout_value
        self.input_size = config.input_size
        self.num_in_points = config.dataset_config.num_in_points
        self.num_in_dims = config.dataset_config.num_in_dims
        self.output_size = config.output_size
        self.num_out_points = config.dataset_config.num_out_points
        self.num_out_dims = config.dataset_config.num_out_dims
        self.in_features = config.dataset_config.in_features
        self.out_features = config.dataset_config.out_features
        if self.dropout_value is not None and self.dropout_value > 0:
            self.dropout = torch.nn.Dropout(self.dropout_value)
        if self.used_norm is not None:
            self.norm = MotionLayerNorm(config)

    @abstractmethod
    def forward(self, input_tensor: torch.Tensor, future_size: int) -> torch.Tensor:
        raise NotImplementedError("This method must be overriden")

    @classmethod
    def normalize_tensor(cls, function):
        """decorator for normalization of the input tensor before forward method"""

        @functools.wraps(function)
        def normalize(*args, **kwargs):
            self = args[0]
            input_tensor = args[1]
            if self.norm_on_last_input:
                # TODO: Add some relative_norm_layer
                seq_last = copy.deepcopy(input_tensor[:, -1:, :, :].detach())
                input_tensor = get_relative_tensor_from(
                    input_tensor, seq_last, self.in_features
                )
            if self.used_norm:
                input_tensor = self.norm(input_tensor)
            if self.dropout_value is not None and self.dropout_value > 0:
                input_tensor = self.dropout(input_tensor)
            predictions = function(self, input_tensor, **kwargs)
            if self.norm_on_last_input:
                # TODO: Add some relative_norm_layer
                seq_last = convert_tensor_features_to(seq_last, copy.deepcopy(self.in_features), copy.deepcopy(self.out_features))
                predictions = get_absolute_tensor_from(
                    predictions, seq_last, self.out_features
                )
            return predictions

        return normalize

    @classmethod
    def allow_unbatched(cls, function):
        """decorator for seemless batched/unbatched forward methods"""

        @functools.wraps(function)
        def reshape(*args, **kwargs):
            self = args[0]
            input_tensor = args[1]
            unbatched = len(input_tensor.shape) == 3
            if unbatched:
                input_tensor = torch.unsqueeze(input_tensor, dim=0)
            predictions = function(self, input_tensor, **kwargs)
            if unbatched:
                predictions = torch.squeeze(predictions, dim=0)
            return predictions

        return reshape
