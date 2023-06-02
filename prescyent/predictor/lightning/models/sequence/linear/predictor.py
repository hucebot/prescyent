"""Pytorch module et Lightning module for Linears"""

from prescyent.predictor.lightning.models.sequence.predictor import SequencePredictor
from prescyent.predictor.lightning.models.sequence.linear.module import TorchModule
from prescyent.predictor.lightning.models.sequence.linear.config import Config


class Predictor(SequencePredictor):
    """Upper class to train and use a Linear Model"""
    PREDICTOR_NAME = "LinearPredictor"
    module_class = TorchModule
    config_class = Config

    def __init__(self, model_path=None, config=None):
        super().__init__(model_path, config, self.PREDICTOR_NAME)
