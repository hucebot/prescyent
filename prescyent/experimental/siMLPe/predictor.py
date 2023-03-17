"""Pytorch module et Lightning module for MLP"""

from prescyent.predictor.lightning.sequence.predictor import SequencePredictor
from .module import TorchModule
from .config import Config


class Predictor(SequencePredictor):
    """Upper class to train and use a MLP Model"""

    def __init__(self, model_path=None, config=None):
        self.module_class = TorchModule
        self.config_class = Config
        super().__init__(model_path, config, "siMLPe")
