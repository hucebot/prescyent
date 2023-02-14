"""Interface for the library's Predictors
The predictor can be trained and predict
"""
from typing import Dict, Iterable, Union

from pydantic import BaseModel
from pytorch_lightning.loggers import TensorBoardLogger


class BasePredictor():
    """abstract class for any predictor"""
    log_root_path: str
    name: str
    version: int

    def __init__(self, log_root_path: str,
                 name: str = None, version: Union[str, int] = None) -> None:
        self.log_root_path = log_root_path
        if name is None:
            name = self.__class__.__name__
        self.name = name
        self.version = version
        self._init_logger()

    def _init_logger(self):
        self.tb_logger = TensorBoardLogger(self.log_root_path,
                                           name=self.name,
                                           version=self.version)
        # redetermine version from tb logger logic if None
        if self.version is None:
            self.version = self.tb_logger._version

    def __call__(self, input_batch, input_size: int = None, input_step: int = 1):
        return self.run(input_batch, input_size, input_step)

    def __str__(self) -> str:
        return f"{self.name}_v{self.version}"

    def _build_from_config(self, config: Dict):
        """build predictor from a config"""
        raise NotImplementedError("This method must be overriden by the inherited predictor")

    def train(self, train_dataloader: Iterable,
              train_config: BaseModel = None,
              val_dataloader: Iterable = None):
        """train predictor"""
        raise NotImplementedError("This method must be overriden by the inherited predictor")

    def test(self, test_dataloader: Iterable):
        """test predictor"""
        raise NotImplementedError("This method must be overriden by the inherited predictor")

    def run(self, input_batch: Iterable, input_size: int, input_step: int):
        """run predictor"""
        raise NotImplementedError("This method must be overriden by the inherited predictor")
