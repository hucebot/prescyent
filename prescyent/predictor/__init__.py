"""
Core Package with the methods to run a prediction
Create a predictor object from a config
Predictors can be a specific NN architecture or an algorithm
Predictors can be trained, loaded from a checkpoint, and runned

Built with pytorch_lightning and pydantic (for now)
"""

from prescyent.predictor.lightning.training_config import TrainingConfig
from prescyent.predictor.auto_predictor import AutoPredictor

from prescyent.predictor.delayed_predictor import DelayedPredictor
from prescyent.predictor.constant_predictor import ConstantPredictor

from prescyent.predictor.lightning.sequence.linear import LinearConfig, LinearPredictor
from prescyent.predictor.lightning.sequence.seq2seq import Seq2SeqConfig, Seq2SeqPredictor
from prescyent.predictor.lightning.sequence.mlp import MlpPredictor, MlpConfig

from prescyent.predictor.lightning.autoreg.sarlstm import SARLSTMConfig, SARLSTMPredictor
