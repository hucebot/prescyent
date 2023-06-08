"""
Core Package with the methods to run a prediction
Create a predictor object from a config
Predictors can be a specific NN architecture or an algorithm
Predictors can be trained, loaded from a checkpoint, and runned

Built with pytorch_lightning and pydantic (for now)
"""
from prescyent.predictor.lightning.configs.training_config import TrainingConfig
from prescyent.predictor.delayed_predictor import DelayedPredictor
from prescyent.predictor.constant_predictor import ConstantPredictor
from prescyent.predictor.lightning.models.sequence.linear import (
    LinearConfig,
    LinearPredictor,
)
from prescyent.predictor.lightning.models.sequence.seq2seq import (
    Seq2SeqConfig,
    Seq2SeqPredictor,
)
from prescyent.predictor.lightning.models.sequence.mlp import MlpPredictor, MlpConfig
from prescyent.predictor.lightning.models.autoreg.sarlstm import (
    SARLSTMConfig,
    SARLSTMPredictor,
)

from prescyent.utils.logger import logger, PREDICTOR


try:
    from prescyent.experimental.simlpe import SiMLPePredictor

    use_experimental = True
except ModuleNotFoundError:
    use_experimental = False
    logger.warning(
        "modules from experimental package will not be instanciable", group=PREDICTOR
    )


PREDICTOR_LIST = [LinearPredictor, SARLSTMPredictor, Seq2SeqPredictor, MlpPredictor]
if use_experimental:
    PREDICTOR_LIST.append(SiMLPePredictor)


PREDICTOR_MAP = {p.PREDICTOR_NAME: p for p in PREDICTOR_LIST}
