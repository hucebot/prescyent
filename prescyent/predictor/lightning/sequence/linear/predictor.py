"""Pytorch module et Lightning module for Linears"""

from prescyent.predictor.lightning.sequence.predictor import SequencePredictor
from prescyent.predictor.lightning.sequence.linear.module import LightningModule
from prescyent.predictor.lightning.sequence.linear.config import Config


class Predictor(SequencePredictor):
    """Upper class to train and use a Linear Model"""

    def __init__(self, model_path=None, config=None):
        self.module_class = LightningModule
        self.config_class = Config
        super().__init__(model_path, config, "LinearPredictor")
