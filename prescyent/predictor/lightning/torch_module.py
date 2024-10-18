from abc import abstractmethod
import copy
import functools
from typing import Dict, Optional
import torch

from prescyent.dataset.features import (
    convert_tensor_features_to,
    features_are_convertible_to,
)
from prescyent.dataset.features.feature_relative import (
    get_relative_tensor_from,
    get_absolute_tensor_from,
)


class BaseTorchModule(torch.nn.Module):
    """Common methods for any torch module to be a lightning predictor with decorators for the forward method"""

    def __init__(self, config) -> None:
        super().__init__()
        self.deriv_on_last_frame = config.deriv_on_last_frame
        self.deriv_output = config.deriv_output
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
        if (self.deriv_on_last_frame or self.deriv_output) and (
            not features_are_convertible_to(self.in_features, self.out_features)
        ):
            raise AttributeError(
                "Cannot use 'deriv_on_last_frame' with non equivalent"
                f"in_features {self.in_features} and "
                f"out_features {self.out_features}"
            )
        if self.dropout_value is not None and self.dropout_value > 0:
            self.dropout = torch.nn.Dropout(self.dropout_value)

    @abstractmethod
    def forward(
        self,
        input_tensor: torch.Tensor,
        future_size: int,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("This method must be overriden")

    @staticmethod
    def deriv_tensor(function):
        """decorator to deriv input and/or output tensor before at forward method of the torch module.
        input/output tensor are derivated from the input tensor's last frame"""

        @functools.wraps(function)
        def deriv_from_last_frame(self, input_tensor, *args, **kwargs):
            if self.deriv_on_last_frame or self.deriv_output:
                seq_last = input_tensor[:, -1:, :, :].clone()
                if self.deriv_on_last_frame:
                    input_tensor = get_relative_tensor_from(
                        input_tensor, seq_last, self.in_features
                    )
            if self.dropout_value is not None and self.dropout_value > 0:
                input_tensor = self.dropout(input_tensor)
            predictions = function(self, input_tensor, *args, **kwargs)
            for feat in self.out_features:
                if feat.must_post_process:
                    predictions[:, :, :, feat.ids] = feat.post_process(
                        predictions[:, :, :, feat.ids]
                    )
            if self.deriv_on_last_frame or self.deriv_output:
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
                        "You cannot deriv from input tensor's last frame if output points are not included in input"
                    ) from err
                seq_last = torch.index_select(seq_last, 2, out_points_ids)
                # seq_last.to(device=input_tensor.device, dtype=input_tensor.dtype)
                predictions = get_absolute_tensor_from(
                    predictions, seq_last, self.out_features
                )
            return predictions

        return deriv_from_last_frame
