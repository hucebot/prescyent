"""simple predictor to use as a baseline"""

from typing import Dict, Iterable

import torch
from pydantic import BaseModel

from prescyent.evaluator import get_ade, get_fde
from prescyent.predictor.base_predictor import BasePredictor
from prescyent.utils.logger import logger, PREDICTOR


class DelayedPredictor(BasePredictor):
    """simple predictor that simply return the input"""

    def __init__(self, log_path: str = "data/models") -> None:
        super().__init__(log_path, version=0)

    def _build_from_config(self, config: Dict):
        """build predictor from a config"""
        logger.warning("No config necessary for this predictor %s",
                       self.__class__.__name__,
                       group=PREDICTOR)

    def train(self, train_dataloader: Iterable,
              train_config: BaseModel = None,
              val_dataloader: Iterable = None):
        """train predictor"""
        logger.warning("No training necessary for this predictor %s",
                       self.__class__.__name__,
                       group=PREDICTOR)

    def save(self, save_path: str):
        """train predictor"""
        logger.warning("No save necessary for this predictor %s",
                       self.__class__.__name__,
                       group=PREDICTOR)

    def get_prediction(self, input_t, future_size):
        return input_t
