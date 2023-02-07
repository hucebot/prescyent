"""Interface for the library's Predictors
The predictor can be trained and predict
"""
from typing import Dict, Iterable

from pydantic import BaseModel
from pytorch_lightning.loggers import TensorBoardLogger


class BasePredictor():
    """abstract class for any predictor"""
    log_root_path: str = None
    name: str = None
    version: int = None

    def __init__(self) -> None:
        if self.log_root_path is None:
            raise NotImplementedError("Child class must define a logger path before init parent")
        if self.name is None:
            self.name=self.__class__.__name__
        self._init_logger()

    def _init_logger(self):
        self.tb_logger = TensorBoardLogger(self.log_root_path, name=self.name, version=self.version)
        # determine version from tb logger logic
        if self.version is None:
            self.version = self.tb_logger._version

    def __call__(self, input_batch, input_size=None, input_step=1):
        return self.run(input_batch, input_size, input_step)

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

    def run(self, input_batch: Iterable, input_size:int, input_step:int):
        """run predictor"""
        raise NotImplementedError("This method must be overriden by the inherited predictor")
