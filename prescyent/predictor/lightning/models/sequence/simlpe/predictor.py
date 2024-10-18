"""Pytorch module et Lightning module for SiMLPe's implementation"""

from prescyent.predictor.lightning.models.sequence.predictor import SequencePredictor
from .module import TorchModule
from .config import Config


class Predictor(SequencePredictor):
    """Upper class to train and use a SiMLPe Model"""

    PREDICTOR_NAME = "siMLPe"
    """unique name for this predictror"""
    module_class = TorchModule
    """LightningModule class used in this predictor"""
    config_class = Config
    """PredictorConfig class used in this predictor"""

    def __init__(self, config: Config, skip_build: bool = False):
        super().__init__(config=config, name=self.PREDICTOR_NAME, skip_build=skip_build)
