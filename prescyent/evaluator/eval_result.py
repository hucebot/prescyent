"""simple class for evaluation result"""
from typing import Dict, List

import torch

from prescyent.dataset.features import Feature
from prescyent.dataset.features.feature_manipulation import get_distance


class EvaluationResult:
    average_prediction_error: Dict[str, float]
    max_prediction_error: Dict[str, float]
    rtf: float
    sample: torch.Tensor
    truth: torch.Tensor
    pred: torch.Tensor
    features: List[Feature]

    def __init__(
        self,
        sample: torch.Tensor,
        truth: torch.Tensor,
        pred: torch.Tensor,
        rtf: float,
        features: List[Feature],
    ) -> None:
        self.rtf = rtf
        self.sample = sample
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
