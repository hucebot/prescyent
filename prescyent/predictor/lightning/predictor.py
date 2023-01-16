from collections.abc import Iterable, Callable
from typing import Dict
from pathlib import Path

import pytorch_lightning as pl

from prescyent.predictor.base_predictor import BasePredictor
from prescyent.predictor.lightning.training_config import TrainingConfig


class LightningPredictor(BasePredictor):
    """Predictor to run any lightning module
    """
    model: pl.LightningModule
    training_config: TrainingConfig
    trainer: pl.Trainer = None

    def __call__(self, input_batch):
        return self.run(input_batch)

    def _build_from_id(self, identifier: str):
        raise NotImplementedError()

    def _build_from_config(self, config: Dict):
        raise NotImplementedError()

    def _init_training_config(self, config):
        if isinstance(config, dict):
            config = TrainingConfig(**config)
        self.training_config = config

    @classmethod
    def _load_from_path(cls, path: str, module_class: Callable):
        supported_extentions = [".ckpt", ".pb"]   # prefered order
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError("No file or directory at %s" % model_path)

        if model_path.is_dir():
            found_model = None
            for extention in supported_extentions:
                if list(model_path.rglob(f'*{extention}')):
                    # WARNING : Chosing last file if there are many
                    found_model = sorted(model_path.rglob(f'*{extention}'))[-1]
                    break
            if found_model is None:
                raise FileNotFoundError("No file matching %s was found in directory %s"
                                        % (supported_extentions, model_path))
            model_path = found_model

        if model_path.suffix == ".ckpt":
            return cls._load_from_checkpoint(model_path, module_class)
        # TODO: State Dict loading not supported yet
        # elif model_path.suffix == ".pt":
        #     return cls._load_from_state_dict(model_path, module_class)
        elif model_path.suffix == ".pb":
            return cls._load_from_binary(model_path, module_class)
        else:
            raise NotImplementedError("Given file extention %s is not supported. "
                                      "Models exported by this module and imported to it can be %s"
                                      % (model_path.suffix, supported_extentions))

    @classmethod
    def _load_from_state_dict(cls, path: str, module_class: Callable):
        return module_class.load_from_state_dict(path)

    @classmethod
    def _load_from_binary(cls, path: str, module_class: Callable):
        return module_class.load_from_binary(path)

    @classmethod
    def _load_from_checkpoint(cls, path: str, module_class: Callable):
        return module_class.load_from_checkpoint(path)

    def train(self, train_dataloader: Iterable, train_config: TrainingConfig = None):
        """train the model"""
        if not train_config:
            train_config = TrainingConfig()
        self._init_training_config(train_config)
        self.training_config = train_config
        self.trainer = pl.Trainer(max_epochs=train_config.epoch,
                                  accelerator=train_config.accelerator,
                                  devices=train_config.devices,
                                  )
        # where do we store checkpoints and models
        # probably we'll need a more complex constructor
        self.trainer.fit(model=self.model, train_dataloaders=train_dataloader)

    def test(self, test_dataloader: Iterable):
        """test the model"""
        self.trainer = pl.Trainer(devices=1)
        self.trainer.test(model=self.model, dataloaders=test_dataloader)

    def run(self, input_batch: Iterable):
        """run method/model inference on the input batch"""
        self.model.eval()
        return self.model.torch_model(input_batch)

    def save(self, save_path):
        """save model to path"""
        if isinstance(save_path, str):
            save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        if self.trainer is not None:
            self.trainer.save_checkpoint(save_path / "trainer_checkpoint.ckpt")
        self.model.save(save_path)
