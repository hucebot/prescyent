"""Lightning Predictor class for ML predictors"""
from collections.abc import Iterable, Callable
import inspect
import json
import math
import shutil
from typing import Type, Union
from pathlib import Path

from pydantic import BaseModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import torch

from prescyent.predictor.base_predictor import BasePredictor
from prescyent.predictor.lightning.training_config import TrainingConfig
from prescyent.utils.logger import logger, PREDICTOR
from prescyent.utils.tensorboard_utils import retreive_log_version_from_path


class LightningPredictor(BasePredictor):
    """Predictor to run any lightning module
    This class should not be called as is
    You must instanciate a class from wich LightningPredictor is a parent
    """
    module_class = Type[BasePredictor]
    config_class = Type[BaseModel]
    model: pl.LightningModule
    training_config: TrainingConfig
    trainer: pl.Trainer
    tb_logger: TensorBoardLogger

    def __init__(self, model_path=None, config=None) -> None:
        # -- Init Model and root path
        if model_path is not None:
            self.model = self._load_from_path(model_path)
            self.log_root_path = model_path if Path(model_path).is_dir() \
                else str(Path(model_path).parent)
            self._load_config(Path(self.log_root_path) / "config.json")
            self.log_root_path, self.name, self.version = retreive_log_version_from_path(model_path)
        elif config is not None:
            self.name, self.version = None, None
            self.model = self._build_from_config(config)
            self.log_root_path = config.model_path
        else:
            # In later versions we can imagine a pretrained or config free version of the model
            raise NotImplementedError("No default implementation for now")

        # -- Init trainer and related args
        if not hasattr(self, "training_config"):
            self.training_config = None
        if not hasattr(self, "trainer"):
            self.trainer = None
        super().__init__()
        self._init_trainer()

    def _build_from_config(self, config):
        # -- We check that the input config is valid through pydantic model
        if isinstance(config, dict):
            config = self.config_class(**config)
        self.config = config

        # -- Build from Scratch
        # The relevant items from "config" are passed as the args for the pytorch module
        return self.module_class(**config.dict(include=set(
            inspect.getfullargspec(self.module_class)[0]
        )))

    def _load_from_path(self, path: str):
        supported_extentions = [".ckpt", ".pb"]   # prefered order
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"No file or directory at {model_path}")

        if model_path.is_dir():
            found_model = None
            for extention in supported_extentions:
                if list(model_path.glob(f'*{extention}')):
                    # WARNING : Chosing last file if there are many
                    found_model = sorted(model_path.glob(f'*{extention}'))[-1]
                    break
            if found_model is None:
                raise FileNotFoundError(f"No file matching {supported_extentions}"
                                        f" was found in directory {model_path}")
            model_path = found_model

        if model_path.suffix == ".ckpt":
            return self.module_class.load_from_checkpoint(model_path)
        #     return self.module_class._load_from_state_dict(model_path, module_class)
        elif model_path.suffix == ".pb":
            return self.module_class.load_from_binary(model_path)
        else:
            raise NotImplementedError(f"Given file extention {model_path.suffix} "
                                      "is not supported. Models exported by this module "
                                      f"and imported to it can be {supported_extentions}")

    @classmethod
    def _load_from_binary(cls, path: str, module_class: Callable):
        return module_class.load_from_binary(path)

    @classmethod
    def _load_from_checkpoint(cls, path: str, module_class: Callable):
        return module_class.load_from_checkpoint(path)

    def _init_training_config(self, config):
        if isinstance(config, dict):
            config = TrainingConfig(**config)
        self.training_config = config

    def _init_trainer(self):
        if self.training_config is None:
            self.training_config = TrainingConfig()
        cls_default_params = {arg for arg in inspect.signature(pl.Trainer).parameters}
        kwargs = self.training_config.dict(include=cls_default_params)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        self.trainer = pl.Trainer(logger=self.tb_logger,
                                  max_epochs=self.training_config.epoch,
                                  callbacks=[lr_monitor],
                                  log_every_n_steps=1,
                                  **kwargs)
        logger.info("Predictor logger initialised at %s", self.tb_logger.log_dir,
                    group=PREDICTOR)

    def _save_config(self, save_path: Path):
        res = dict()
        self.config.model_path = str(save_path.parent)
        if self.training_config is not None :
            res["training_config"] = self.training_config.dict()
        if self.config is not None :
            res["model_config"] = self.config.dict()
        with (save_path).open('w', encoding="utf-8") as conf_file:
            json.dump(res, conf_file, indent=4, sort_keys=True)

    def _load_config(self, config_path: Union[Path, str]):
        if config_path is None:
            logger.info("No config were found near model at %s", config_path, group=PREDICTOR)
            return
        if isinstance(config_path, str):
            config_path = Path(config_path)
        with (config_path).open(encoding="utf-8") as conf_file:
            config_data = json.load(conf_file)
        self.training_config = TrainingConfig(**config_data.get("training_config", None))
        self.config = self.config_class(**config_data.get("model_config", None))
        logger.info("Config loaded from %s", config_path, group=PREDICTOR)

    def _init_module_optimizer(self):
        self.model.training_config = self.training_config

    def train(self, train_dataloader: Iterable,
              train_config: TrainingConfig = None,
              val_dataloader: Iterable = None):
        """train the model"""
        if not train_config:
            train_config = TrainingConfig()
        self._init_training_config(train_config)
        self._init_trainer()
        self._init_module_optimizer()
        self.trainer.fit(model=self.model,
                         train_dataloaders=train_dataloader,
                         val_dataloaders=val_dataloader)

    def test(self, test_dataloader: Iterable):
        """test the model"""
        if self.trainer is None:
            logger.info("New trainer as been created at %s", self.tb_logger.log_dir,
                        group=PREDICTOR)
            self._init_trainer()
        self.trainer.test(self.model, test_dataloader)

    def run(self, input_batch: Iterable, input_size: int = None,
            input_step: int = 1) -> torch.Tensor:
        """run method/model inference on the input batch
        The output is either the list of predictions for each defined subpart of the input batch,
        or the single prediction for the whole input

        Args:
            input_batch (Iterable): Input for the predictor's model
            input_size (int|None, optional): If an input size is provided, the input batch will
                be splitted sequences of len == input_size. Defaults to None.
            input_step (int, optional): When splitting the input_batch (input_size != None)
                defines the step of the iteration. Defaults to 1.

        Returns:
            torch.Tensor | List[torch.Tensor]: the model prediction or list of model predictions
        """
        with torch.no_grad():
            self.model.eval()
            # return pred for all input if no input_size was given
            if input_size is None or input_size >= input_batch.shape[0]:
                return self.model.torch_model(input_batch)
            # otherwise we iterate over inputs of len input_size and return a list of predictions
            prediction_list = torch.zeros(input_batch.shape[0] - input_size,
                                          input_size,
                                          input_batch.shape[1])
            for i in range(0, input_batch.shape[0] - input_size, input_step):
                input_sub_batch = input_batch[i:i + input_size]
                prediction_list[i] = self.model.torch_model(input_sub_batch)
            return prediction_list

    def save(self, save_path: Union[str, Path] = None):
        """save model to path"""
        if save_path is None:
            save_path = self.tb_logger.log_dir
        else:  # we cp the tensorflow logger content first
            shutil.copytree(self.tb_logger.log_dir, save_path, dirs_exist_ok=True)
        if isinstance(save_path, str):
            save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        if self.trainer is not None:
            logger.info("Saving checkpoint at %s", (save_path / "trainer_checkpoint.ckpt"),
                        group=PREDICTOR)
            self.trainer.save_checkpoint(save_path / "trainer_checkpoint.ckpt")

        logger.info("Saving model at %s", save_path, group=PREDICTOR)
        self.model.save(save_path)

        logger.info("Saving config at %s", (save_path / "config.json"), group=PREDICTOR)
        self._save_config(save_path / "config.json")
