"""Pytorch module et Lightning module for Seq2Seq"""

from prescyent.predictor.lightning.models.sequence.predictor import SequencePredictor
from prescyent.predictor.lightning.models.sequence.seq2seq.module import TorchModule
from prescyent.predictor.lightning.models.sequence.seq2seq.config import Config


class Predictor(SequencePredictor):
    """Upper class to train and use a Seq2Seq Model"""

    PREDICTOR_NAME = "Seq2SeqPredictor"
    module_class = TorchModule
    config_class = Config

    def __init__(self, model_path=None, config=None):
        super().__init__(model_path, config, self.PREDICTOR_NAME)
