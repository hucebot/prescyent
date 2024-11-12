"""Pytorch module et Lightning module for SiMLPe's implementation"""

from prescyent.predictor.lightning.models.sequence.predictor import SequencePredictor
from .module import SiMLPeTorchModule
from .config import SiMLPeConfig


class SiMLPePredictor(SequencePredictor):
    """Upper class to train and use a SiMLPe Model"""

    PREDICTOR_NAME = "siMLPe"
    """unique name for this predictror"""
    module_class = SiMLPeTorchModule
    """LightningModule class used in this predictor"""
    config_class = SiMLPeConfig
    """PredictorConfig class used in this predictor"""

    def __init__(self, config: SiMLPeConfig, skip_build: bool = False):
        super().__init__(config=config, name=self.PREDICTOR_NAME, skip_build=skip_build)
