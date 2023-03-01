"""simple class for evaluation result"""
from statistics import mean
from typing import List


class EvaluationResult():
    ade: float
    fde: float
    inference_time_ms: float

    def __init__(self, ade: float, fde: float, inference_time_ms: float) -> None:
        self.ade = ade
        self.fde = fde
        self.inference_time_ms = inference_time_ms


class EvaluationSummary():
    results: List[EvaluationResult]

    def __init__(self, results: List[EvaluationResult] = None) -> None:
        if results is None:
            results = []
        self.results = results

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
