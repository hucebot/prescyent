"""
Core Package with the methods to run a prediction
Create a predictor object from a config
Predictors can be a specific NN architecture or an algorithm
Predictors can be trained, loaded from a checkpoint, and runned

Built with pytorch_lightning and pydantic (for now)
"""

from prescyent.predictor.lightning import LSTMConfig, LSTMPredictor, TrainingConfig
