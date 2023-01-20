"""Pytorch module et Lightning module for LSTMs
"""
from typing import Union
import inspect

from prescyent.predictor.lightning.predictor import LightningPredictor
from prescyent.predictor.lightning.lstm.module import LSTMModule, LSTM
from prescyent.predictor.lightning.lstm.config import LSTMConfig


class LSTMPredictor(LightningPredictor):
    """Upper class to train and use a LSTM Model"""

    def __init__(self, model_path=None, config=None):
        if model_path is not None:
            self.model = self._load_from_path(model_path)
            # reload a config instead
            model_name = "lstm"
        elif config is not None:
            self._build_from_config(config)
            model_name = config.identifier
        else:
            # In later versions we can imagine a pretrained or config free version of the model
            raise NotImplementedError("No default implementation for now")
        self._init_logger(model_name=model_name)
        self._init_trainer()

    def _build_from_config(self, config: Union[dict, LSTMConfig]):
        # -- We check that the input config is valid through pydantic model
        if isinstance(config, dict):
            config = LSTMConfig(**config)
        self.config = config

        # -- Check if a checkpoint or file exist:
        if config.model_path:
            self.model = LSTMPredictor._load_from_path(config.model_path)
            return

        # -- Build from Scratch
        # The relevant items from "config" are passed as the args for the pytorch module
        self.model = LSTMModule(**config.dict(include=set(inspect.getfullargspec(LSTMModule)[0])))

    @classmethod
    def _load_from_path(cls, path: str, *args, **kwargs):
        return super()._load_from_path(path, LSTMModule)

    @classmethod
    def _load_from_checkpoint(cls, path: str, *args, **kwargs):
        return LSTMModule.load_from_checkpoint(path)

    @classmethod
    def _load_from_state_dict(cls, path: str, *args, **kwargs):
        return LSTMModule.load_from_state_dict(path)

    @classmethod
    def _load_from_binary(cls, path: str, *args, **kwargs):
        return LSTMModule.load_from_binary(path)
