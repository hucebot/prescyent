"""simple class for evaluation result"""
from statistics import mean
from typing import List

import torch

from prescyent.evaluator.metrics import get_fde, get_ade


class EvaluationResult():
    ade: float
    fde: float
    inference_time_ms: float
    sample: torch.Tensor
    truth: torch.Tensor
    pred: torch.Tensor

    def __init__(self, sample: torch.Tensor, truth: torch.Tensor,
                 pred: torch.Tensor, inference_time_ms: float) -> None:
        self.inference_time_ms = inference_time_ms
        self.sample = sample
        self.truth = truth
        self.pred = pred


    @property
    def ade(self) -> float:
        return get_ade(self.truth, self.pred[:len(self.truth)]).item()
    @property
    def fde(self) -> float:
        return get_fde(self.truth, self.pred[:len(self.truth)]).item()

class EvaluationSummary():
    results: List[EvaluationResult]

    def __init__(self, results: List[EvaluationResult] = None) -> None:
        if results is None:
            results = []
        self.results = results

    def __getitem__(self, item):
         return self.results[item]

    def __len__(self):
        return len(self.results)

    @property
    def mean_ade(self) -> float:
        return mean([eval.ade for eval in self.results])

    @property
    def mean_fde(self):
        return mean([eval.fde for eval in self.results])

    @property
    def mean_inference_time_ms(self):
        return mean([eval.inference_time_ms for eval in self.results])

    @property
    def max_ade(self) -> float:
        return max([eval.ade for eval in self.results])

    @property
    def max_fde(self):
        return max([eval.fde for eval in self.results])

    @property
    def max_inference_time_ms(self):
        return max([eval.inference_time_ms for eval in self.results])
