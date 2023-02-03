"""
Core Package with the methods to run a prediction
Create a predictor object from a config
Predictors can be a specific NN architecture or an algorithm
Predictors can be trained, loaded from a checkpoint, and runned

Built with pytorch_lightning and pydantic (for now)
"""

from prescyent.predictor.delayed_predictor import DelayedPredictor
from prescyent.predictor.lightning.training_config import TrainingConfig

from prescyent.predictor.lightning.lstm import LSTMConfig, LSTMPredictor
from prescyent.predictor.lightning.linear import LinearConfig, LinearPredictor
from prescyent.predictor.lightning.seq2seq import Seq2SeqConfig, Seq2SeqPredictor
