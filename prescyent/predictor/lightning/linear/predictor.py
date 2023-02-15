"""Pytorch module et Lightning module for Linears"""

from prescyent.predictor.lightning.predictor import LightningPredictor
from prescyent.predictor.lightning.linear.module import LightningModule
from prescyent.predictor.lightning.linear.config import Config


class Predictor(LightningPredictor):
    """Upper class to train and use a Linear Model"""

    def __init__(self, model_path=None, config=None):
        self.module_class = LightningModule
        self.config_class = Config
        super().__init__(model_path, config, "LinearPredictor")
