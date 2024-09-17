"""simple class for evaluation result"""
from typing import Dict, List

import torch

from prescyent.dataset.features import Features
from prescyent.dataset.features.feature_manipulation import get_distance


class EvaluationResult:
    average_prediction_error: Dict[str, float]
    max_prediction_error: Dict[str, float]
    rtf: float
    truth: torch.Tensor
    pred: torch.Tensor
    features: Features

    def __init__(
        self,
        truth: torch.Tensor,
        pred: torch.Tensor,
        rtf: float,
        features: Features,
    ) -> None:
        self.rtf = rtf
        self.truth = truth
        self.pred = pred
        self.features = features

    @property
    def prediction_error(self) -> Dict[str, torch.Tensor]:
        return get_distance(
            self.pred[: len(self.truth)], self.features, self.truth, self.features
        )

    @property
    def average_prediction_error(self) -> Dict[str, float]:
        return {key: value.mean() for key, value in self.prediction_error.items()}

    @property
    def max_prediction_error(self) -> Dict[str, float]:
        return {key: value.max() for key, value in self.prediction_error.items()}
