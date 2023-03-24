"""Pytorch module et Lightning module for MLP"""

from prescyent.predictor.lightning.sequence.predictor import SequencePredictor
from prescyent.predictor.lightning.sequence.mlp.module import TorchModule
from prescyent.predictor.lightning.sequence.mlp.config import Config


class Predictor(SequencePredictor):
    """Upper class to train and use a MLP Model"""
    PREDICTOR_NAME = "MlpPredictor"

    def __init__(self, model_path=None, config=None):
        self.module_class = TorchModule
        self.config_class = Config
        super().__init__(model_path, config, self.PREDICTOR_NAME)
