"""simple class for evaluation result"""
import torch

from prescyent.evaluator.metrics import get_fde, get_ade, get_mpjpe


class EvaluationResult:
    ade: float
    fde: float
    inference_time_ms: float
    sample: torch.Tensor
    truth: torch.Tensor
    pred: torch.Tensor

    def __init__(
        self,
        sample: torch.Tensor,
        truth: torch.Tensor,
        pred: torch.Tensor,
        inference_time_ms: float,
    ) -> None:
        self.inference_time_ms = inference_time_ms
        self.sample = sample
        self.truth = truth
        self.pred = pred

    @property
    def ade(self) -> float:
        return get_ade(self.truth, self.pred[: len(self.truth)]).detach().item()

    @property
    def fde(self) -> float:
        return get_fde(self.truth, self.pred[: len(self.truth)]).detach().item()

    @property
    def mpjpe(self) -> float:
        return get_mpjpe(self.truth, self.pred[: len(self.truth)])[-1].detach().item()
