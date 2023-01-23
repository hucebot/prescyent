"""Pytorch module et Lightning module for LSTMs"""
from pathlib import Path
from typing import Union
import inspect

from prescyent.predictor.lightning.predictor import LightningPredictor
from prescyent.predictor.lightning.lstm.module import LSTMModule
from prescyent.predictor.lightning.lstm.config import LSTMConfig


class LSTMPredictor(LightningPredictor):
    """Upper class to train and use a LSTM Model"""

    def __init__(self, model_path=None, config=None):
        if model_path is not None:
            model = self._load_from_path(model_path)
            root_path = model_path if Path(model_path).is_dir() else str(Path(model_path).parent)
            self._load_config(Path(root_path) / "config.json")
        elif config is not None:
            model = self._build_from_config(config)
            root_path = config.model_path
        else:
            # In later versions we can imagine a pretrained or config free version of the model
            raise NotImplementedError("No default implementation for now")
        super().__init__(model, root_path)

    def _load_config(self, config_path: Union[Path, str]):
        super()._load_config(config_path)
        self.config = LSTMConfig(**self.config_data.get("model_config", None))
        del self.config_data

    def _build_from_config(self, config: Union[dict, LSTMConfig]):
        # -- We check that the input config is valid through pydantic model
        if isinstance(config, dict):
            config = LSTMConfig(**config)
        self.config = config

        # -- Build from Scratch
        # The relevant items from "config" are passed as the args for the pytorch module
        return LSTMModule(**config.dict(include=set(inspect.getfullargspec(LSTMModule)[0])))

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
