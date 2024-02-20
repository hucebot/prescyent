"""simple class for evaluation summary"""
from statistics import mean
from typing import Dict, List

import numpy as np

from prescyent.evaluator.eval_result import EvaluationResult


class EvaluationSummary:
    results: List[EvaluationResult]

    def __init__(self, results: List[EvaluationResult] = None) -> None:
        if results is None:
            results = []
        self.results = results

    @property
    def features(self):
        return self.results[0].features

    def __getitem__(self, item):
        return self.results[item]

    def __len__(self):
        return len(self.results)

    def __str__(self) -> str:
        return (
            f"\n- **Average Prediction Error**: {self.average_prediction_error}"
            + f"\n- **Max Prediction Error**: {self.max_prediction_error}"
            + f"\n- **Mean Real Time Factor (process time for one second)**: {self.mean_rtf:.6f}"
            + f"\n- **Max Real Time Factor (process time for one second)**: {self.max_rtf:.6f}"
        )

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
