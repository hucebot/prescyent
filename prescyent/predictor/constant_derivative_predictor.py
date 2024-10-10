"""simple predictor to use as a baseline"""
from typing import Dict, Optional

import torch

from prescyent.dataset.features import convert_tensor_features_to
from prescyent.dataset.features.feature_relative import (
    get_absolute_tensor_from,
    get_relative_tensor_from,
)
from prescyent.predictor.config import PredictorConfig
from prescyent.predictor.constant_predictor import ConstantPredictor
from prescyent.utils.tensor_manipulation import is_tensor_is_batched


class ConstantDerivativePredictor(ConstantPredictor):
    """simple predictor that simply return the last input"""

    def __init__(
        self,
        config: Optional[PredictorConfig],
    ) -> None:
        if config is None:
            raise AttributeError(
                "You need a valid dataset config with out_features to use this dataset"
            )
        super().__init__(config)

    def predict(
        self, input_t: torch.Tensor, future_size: int, *args, **kwargs
    ) -> torch.Tensor:
        unbatch = False
        if not is_tensor_is_batched(input_t):
            unbatch = True
            input_t = input_t.unsqueeze(0)
        input_t_vel_at_last_frame = get_relative_tensor_from(
            input_t[:, -1], input_t[:, -2], self.config.dataset_config.out_features
        )
        predictions = []
        next_pred = input_t[:, -1]
        for _ in range(future_size):
            next_pred = get_absolute_tensor_from(
                next_pred,
                input_t_vel_at_last_frame,
                self.config.dataset_config.out_features,
            )
            predictions.append(next_pred.unsqueeze(1))
        output_t = torch.cat(predictions, dim=1)
        try:
            out_points_ids = torch.LongTensor(
                [
                    self.config.dataset_config.in_points.index(out)
                    for out in self.config.dataset_config.out_points
                ]
            )
            out_points_ids = out_points_ids.to(device=input_t.device)
        except ValueError as err:
            raise AttributeError(
                "You cannot use this predictor if output points are not included in input!"
            ) from err
        output_t = torch.index_select(output_t, 2, out_points_ids)
        output_t = convert_tensor_features_to(
            output_t,
            self.config.dataset_config.in_features,
            self.config.dataset_config.out_features,
        )
        if unbatch:
            output_t = output_t.squeeze(0)
        return output_t
