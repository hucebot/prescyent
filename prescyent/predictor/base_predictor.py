"""Interface for the library's Predictors
The predictor can be trained and predict
"""
import copy
from pathlib import Path
from typing import Dict, Iterable, Union

from pydantic import BaseModel
from pytorch_lightning.loggers import TensorBoardLogger


class BasePredictor():
    """abstract class for any predictor"""
    log_root_path: str
    name: str
    version: int

    def __init__(self, log_root_path: str,
                 name: str = None, version: Union[str, int] = None,
                 no_sub_dir_log: bool = False) -> None:
        self.log_root_path = log_root_path
        if name is None:
            name = self.__class__.__name__
        self.name = name
        self.version = version
        self._init_logger(no_sub_dir_log)

    def _init_logger(self, no_sub_dir_log=False):
        if no_sub_dir_log:
            name = ""
            version = ""
        else:
            name = self.name
            version = self.version
        self.tb_logger = TensorBoardLogger(self.log_root_path,
                                           name=name,
                                           version=version)
        # redetermine version from tb logger logic if None
        if self.version is None:
            self.version = copy.deepcopy(self.tb_logger.version)

    def __call__(self, input_batch, history_size: int = None, history_step: int = 1,
                 future_size: int = 0, output_only_future: bool = True):
        return self.run(input_batch, history_size, history_step, future_size, output_only_future)

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

    def run(self, input_batch: Iterable, history_size: int, history_step: int,
            future_size: int, output_only_future: bool):
        """run predictor"""
        raise NotImplementedError("This method must be overriden by the inherited predictor")

    def save(self, save_path: str):
        """save predictor"""
        raise NotImplementedError("This method must be overriden by the inherited predictor")
