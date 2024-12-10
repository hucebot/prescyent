"""simple class for evaluation summary"""
from statistics import mean
from typing import Dict, List

import numpy as np

from prescyent.evaluator.eval_result import EvaluationResult


class EvaluationSummary:
    """class to store a list of evaluation results, and summarize metrics"""

    results: List[EvaluationResult]
    predictor_name: str
    predicted_future: float

    def __init__(
        self,
        results: List[EvaluationResult] = None,
        predictor_name: str = None,
        predicted_future: float = None,
    ) -> None:
        if results is None:
            results = []
        self.results = results
        self.predictor_name = predictor_name
        self.predicted_future = predicted_future

    @property
    def features(self):
        return self.results[0].features

    def __getitem__(self, item):
        return self.results[item]

    def __len__(self):
        return len(self.results)

    def __str__(self) -> str:
        return (
            f"\n- **Result Metrics over test set for {self.predictor_name} prediction at {self.predicted_future}s **: "
            + f"\n\t- **Average Prediction Error**: {self.average_prediction_error}"
            + f"\n\t- **Max Prediction Error**: {self.max_prediction_error}"
            + f"\n\t- **Mean Real Time Factor (process time for one second)**: {self.mean_rtf:.6f}"
            + f"\n\t- **Max Real Time Factor (process time for one second)**: {self.max_rtf:.6f}"
        )

    @property
    def headers(self) -> List[str]:
        headers = ["Predictor"]
        for feat in self.features:
            headers.append(
                f"Average {feat.name} error [{feat.distance_unit}] at {self.predicted_future}s"
            )
            headers.append(
                f"Max {feat.name} error [{feat.distance_unit}] at {self.predicted_future}s"
            )
        headers.append("Mean RTF")
        headers.append("Max RTF")
        return headers

    def as_array(self) -> List[str]:
        result_array = [self.predictor_name]
        for feat in self.features:
            result_array.append(str(self.average_prediction_error[feat.name]))
            result_array.append(str(self.max_prediction_error[feat.name]))
        result_array.append(f"{self.mean_rtf:.6f}")
        result_array.append(f"{self.max_rtf:.6f}")
        return result_array

    @property
    def average_prediction_error(self) -> Dict[str, float]:
        average_prediction_errors = [
            evaluation.average_prediction_error for evaluation in self.results
        ]
        return {
            feat_name: np.mean(
                [
                    average_prediction_error[feat_name]
                    for average_prediction_error in average_prediction_errors
                ]
            )
            for feat_name in average_prediction_errors[0].keys()
        }

    @property
    def mean_rtf(self):
        return mean([eval.rtf for eval in self.results])

    @property
    def max_prediction_error(self) -> Dict[str, float]:
        max_prediction_errors = [
            evaluation.max_prediction_error for evaluation in self.results
        ]
        return {
            feat_name: np.max(
                [
                    max_prediction_error[feat_name]
                    for max_prediction_error in max_prediction_errors
                ]
            )
            for feat_name in max_prediction_errors[0].keys()
        }

    @property
    def max_rtf(self):
        return max([eval.rtf for eval in self.results])
