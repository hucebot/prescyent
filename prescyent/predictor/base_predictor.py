"""Interface for the library's Predictors
The predictor can be trained and predict
"""
from typing import Dict, Iterable

from pydantic import BaseModel
from pytorch_lightning.loggers import TensorBoardLogger


class BasePredictor():
    """abstract class for any predictor"""

    def __init__(self, log_path: str) -> None:
        self._init_logger(log_path)

    def _init_logger(self, log_path: str):
        self.tb_logger = TensorBoardLogger(log_path, name=self.__class__.__name__)

    def __call__(self, input_batch):
        return self.run(input_batch)

    def _build_from_config(self, config: Dict):
        """build predictor from a config"""
        raise NotImplementedError("This method must be overriden by the inherited predictor")

    def train(self, train_dataloader: Iterable,
              train_config: BaseModel=None,
              val_dataloader: Iterable=None):
        """train predictor"""
        raise NotImplementedError("This method must be overriden by the inherited predictor")

    def test(self, test_dataloader: Iterable):
        """test predictor"""
        raise NotImplementedError("This method must be overriden by the inherited predictor")

    def run(self, input_batch: Iterable):
        """run predictor"""
        raise NotImplementedError("This method must be overriden by the inherited predictor")
