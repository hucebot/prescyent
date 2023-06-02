"""Pytorch module et Lightning module for MLP"""

from prescyent.predictor.lightning.models.sequence.predictor import SequencePredictor
from .module import TorchModule
from .config import Config


class Predictor(SequencePredictor):
    """Upper class to train and use a MLP Model"""
    PREDICTOR_NAME = "siMLPe"
    module_class = TorchModule
    config_class = Config

    def __init__(self, model_path=None, config=None):
        super().__init__(model_path, config, self.PREDICTOR_NAME)
