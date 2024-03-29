from abc import abstractmethod
import copy
import functools
import torch

from prescyent.dataset.features import (
    convert_tensor_features_to,
    features_are_convertible_to,
)
from prescyent.predictor.lightning.layers.normalization_layer import MotionLayerNorm
from prescyent.dataset.features.feature_relative import (
    get_relative_tensor_from,
    get_absolute_tensor_from,
)


class BaseTorchModule(torch.nn.Module):
    """Common methods for any torch module to be a lightning predictor"""

    def __init__(self, config) -> None:
        super().__init__()
        self.norm_on_last_input = config.norm_on_last_input
        self.used_norm = config.used_norm
        self.dropout_value = config.dropout_value
        self.in_sequence_size = config.in_sequence_size
        self.num_in_points = config.dataset_config.num_in_points
        self.num_in_dims = config.dataset_config.num_in_dims
        self.out_sequence_size = config.out_sequence_size
        self.num_out_points = config.dataset_config.num_out_points
        self.num_out_dims = config.dataset_config.num_out_dims
        self.in_features = config.dataset_config.in_features
        self.out_features = config.dataset_config.out_features
        self.in_points = config.dataset_config.in_points
        self.out_points = config.dataset_config.out_points
        if self.norm_on_last_input and (
            not features_are_convertible_to(self.in_features, self.out_features)
        ):
            raise AttributeError(
                "Cannot use 'norm_on_last_input' with non equivalent"
                f"in_features {self.in_features} and "
                f"out_features {self.out_features}"
            )
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
                seq_last = input_tensor[:, -1:, :, :].clone()
                input_tensor = get_relative_tensor_from(
                    input_tensor, seq_last, self.in_features
                )
            if self.used_norm:
                input_tensor = self.norm(input_tensor.clone())
            if self.dropout_value is not None and self.dropout_value > 0:
                input_tensor = self.dropout(input_tensor)
            predictions = function(self, input_tensor, **kwargs)
            for feat in self.out_features:
                if feat.must_post_process:
                    predictions[:, :, :, feat.ids] = feat.post_process(
                        predictions[:, :, :, feat.ids]
                    )
            if self.norm_on_last_input:
                seq_last = convert_tensor_features_to(
                    seq_last,
                    self.in_features,
                    self.out_features,
                )
                try:
                    out_points_ids = torch.LongTensor(
                        [self.in_points.index(out) for out in self.out_points]
                    )
                    out_points_ids = out_points_ids.to(device=input_tensor.device)
                except ValueError as err:
                    raise AttributeError(
                        "You cannot use norm_on_last_input if output points are not included in input!"
                    ) from err
                seq_last = torch.index_select(seq_last, 2, out_points_ids)
                # seq_last.to(device=input_tensor.device, dtype=input_tensor.dtype)
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
