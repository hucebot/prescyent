"""simple predictor to use as a baseline"""

from typing import Dict, Iterable, Optional

import torch

from prescyent.predictor.base_predictor import BasePredictor
from prescyent.predictor.lightning.configs.training_config import TrainingConfig
from prescyent.utils.logger import logger, PREDICTOR
from prescyent.utils.tensor_manipulation import is_tensor_is_batched


class ConstantPredictor(BasePredictor):
    """simple predictor that simply return the last input"""

    def __init__(self, log_path: str = "data/models") -> None:
        super().__init__(log_path, version=0)

    def _build_from_config(self, config: Dict):
        """build predictor from a config"""
        logger.getChild(PREDICTOR).warning(
            "No config necessary for this predictor %s",
            self.__class__.__name__,
        )

    def train(
        self,
        train_dataloader: Iterable,
        train_config: Optional[TrainingConfig] = None,
        val_dataloader: Optional[Iterable] = None,
    ):
        """train predictor"""
        logger.getChild(PREDICTOR).warning(
            "No training necessary for this predictor %s",
            self.__class__.__name__,
        )

    def finetune(
        self,
        train_dataloader: Iterable,
        train_config: Optional[TrainingConfig] = None,
        val_dataloader: Optional[Iterable] = None,
    ):
        """finetune predictor"""
        logger.getChild(PREDICTOR).warning(
            "No training necessary for this predictor %s",
            self.__class__.__name__,
        )

    def save(self, save_path: str):
        """train predictor"""
        logger.getChild(PREDICTOR).warning(
            "No save necessary for this predictor %s",
            self.__class__.__name__,
        )

    def predict(self, input_t: torch.Tensor, future_size: int) -> torch.Tensor:
        if is_tensor_is_batched(input_t):
            input_t = torch.transpose(input_t, 0, 1)
            output = [input_t[-1].unsqueeze(0) for _ in range(future_size)]
            output_t = torch.cat(output, dim=0)
            return torch.transpose(output_t, 0, 1)
        output = [input_t[-1].unsqueeze(0) for _ in range(future_size)]
        return torch.cat(output, dim=0)
