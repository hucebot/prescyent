"""Pytorch module et Lightning module for Seq2Seq"""

from prescyent.predictor.lightning.predictor import LightningPredictor
from prescyent.predictor.lightning.seq2seq.module import LightningModule
from prescyent.predictor.lightning.seq2seq.config import Config


class Predictor(LightningPredictor):
    """Upper class to train and use a Seq2Seq Model"""

    def __init__(self, model_path=None, config=None):
        self.module_class = LightningModule
        self.config_class = Config
        super().__init__(model_path, config, "Seq2SeqPredictor")
