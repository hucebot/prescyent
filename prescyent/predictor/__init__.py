"""
Core Package with the methods to run a prediction
Create a predictor object from a config
Predictors can be a specific NN architecture or an algorithm
Predictors can be trained, loaded from a checkpoint, and run prediction over trajectories

Built with pytorch_lightning and pydantic
"""
from prescyent.predictor.lightning.configs.training_config import TrainingConfig
from prescyent.predictor.config import PredictorConfig
from prescyent.predictor.delayed_predictor import DelayedPredictor
from prescyent.predictor.constant_predictor import ConstantPredictor
from prescyent.predictor.constant_derivative_predictor import (
    ConstantDerivativePredictor,
)
from prescyent.predictor.lightning.models.sequence.seq2seq import (
    Seq2SeqConfig,
    Seq2SeqPredictor,
)
from prescyent.predictor.lightning.models.sequence.simlpe import (
    SiMLPeConfig,
    SiMLPePredictor,
)
from prescyent.predictor.lightning.models.sequence.mlp import MlpPredictor, MlpConfig
from prescyent.predictor.lightning.models.autoreg.sarlstm import (
    SARLSTMConfig,
    SARLSTMPredictor,
)
from prescyent.predictor.promp import PrompConfig, PrompPredictor
from prescyent.utils.logger import logger, PREDICTOR


# No more experimental predictor for now, kept logic for later use
# try:
#     # from prescyent.experimental.simlpe import SiMLPePredictor

#     use_experimental = True
# except ModuleNotFoundError:
#     use_experimental = False
#     logger.getChild(PREDICTOR).warning(
#         "modules from experimental package will not be instanciable"
#     )


PREDICTOR_LIST = [
    SARLSTMPredictor,
    Seq2SeqPredictor,
    MlpPredictor,
    SiMLPePredictor,
]
# if use_experimental:
#     PREDICTOR_LIST.append(SiMLPePredictor)


# Map used for AutoPredictor
PREDICTOR_MAP = {p.PREDICTOR_NAME: p for p in PREDICTOR_LIST}
