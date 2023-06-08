from abc import abstractmethod
import functools
import torch

from prescyent.predictor.lightning.layers.standard_scaler import MotionLayerNorm


class BaseTorchModule(torch.nn.Module):
    """Common methods for any torch module to be a lightning predictor"""

    def __init__(self, config) -> None:
        super().__init__()
        self.norm_on_last_input = config.norm_on_last_input
        self.used_norm = config.used_norm
        self.output_size = config.dict().get("output_size", None)
        self.input_size = config.dict().get("input_size", None)
        self.num_points = config.dict().get("num_points", None)
        self.dropout_value = config.dict().get("dropout_value", None)
        self.num_dims = config.dict().get("num_dims", None)
        if self.dropout_value is not None and self.dropout_value > 0:
            self.dropout = torch.nn.Dropout(self.dropout_value)
        if self.used_norm is not None:
            self.norm = MotionLayerNorm(config)

    @abstractmethod
    def forward(self, input_tensor: torch.Tensor, future_size: int):
        pass

    @classmethod
    def normalize_tensor(cls, function):
        """decorator for normalization of the input tensor before forward method"""

        @functools.wraps(function)
        def normalize(*args, **kwargs):
            self = args[0]
            input_tensor = args[1]
            if self.norm_on_last_input:
                seq_last = input_tensor[:, -1:, :, :].detach()
                input_tensor = input_tensor - seq_last
            if self.used_norm:
                input_tensor = self.norm(input_tensor)
            if self.dropout_value is not None and self.dropout_value > 0:
                input_tensor = self.dropout(input_tensor)
            predictions = function(self, input_tensor, **kwargs)
            if self.norm_on_last_input:
                predictions = predictions + seq_last
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
