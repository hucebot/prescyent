"""Interface for the library's Predictors
The predictor can be trained and predict
"""

from typing import Dict, Iterable
from abc import abstractmethod, ABCMeta


class BasePredictor(metaclass=ABCMeta):
    """abstract class for any predictor"""

    @abstractmethod
    def _build_from_config(self, config: Dict):
        """build predictor from a config"""

    @abstractmethod
    def train(self):
        """train predictor"""

    @abstractmethod
    def run(self, input_batch: Iterable):
        """run predictor"""
