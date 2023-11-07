"""Lightning Predictor class for ML predictors"""
import gc
import inspect
import json
import os
import shutil
from collections.abc import Iterable, Callable
from pathlib import Path
from typing import Dict, Type, Union

import pytorch_lightning as pl
import torch
from pydantic import BaseModel
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.profilers import (
    AdvancedProfiler,
    PyTorchProfiler,
    SimpleProfiler,
)
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    DeviceStatsMonitor,
    EarlyStopping,
)

from prescyent.predictor.base_predictor import BasePredictor
from prescyent.predictor.lightning.module import LightningModule
from prescyent.predictor.lightning.torch_module import BaseTorchModule
from prescyent.predictor.lightning.configs.module_config import ModuleConfig
from prescyent.predictor.lightning.callbacks.progress_bar import LightningProgressBar
from prescyent.predictor.lightning.configs.training_config import TrainingConfig
from prescyent.utils.logger import logger, PREDICTOR
from prescyent.utils.tensor_manipulation import is_tensor_is_batched
from prescyent.predictor.lightning.layers.reshaping_layer import ReshapingLayer


class LightningPredictor(BasePredictor):
    """Predictor to run any lightning module
    This class should not be called as is
    You must instanciate a class from wich LightningPredictor is a parent
    """

    module_class: Type[BaseTorchModule]
    config_class: Type[ModuleConfig]
    model: pl.LightningModule
    training_config: TrainingConfig
    trainer: pl.Trainer
    tb_logger: TensorBoardLogger

    def __init__(self, model_path=None, config=None, name=None) -> None:
        # -- Init Model and root path
        if model_path is not None:
            log_root_path = (
                model_path
                if Path(model_path).is_dir()
                else str(Path(model_path).parent)
            )
            self._load_config(Path(log_root_path) / "config.json", config_data=config)
            name, version = self.name, self.version
            self.model = self._load_from_path(model_path)
            super().__init__(log_root_path, name, version, no_sub_dir_log=True)
        elif config is not None:
            self.model = self._build_from_config(config)
            version = self.config.version
            log_root_path = self.config.save_path
            super().__init__(log_root_path, name, version)
        else:
            # In later versions we can imagine a pretrained or config free version of the model
            raise NotImplementedError(
                "No default implementation for now, "
                "please provide a config or a path to init predictor"
            )
        # -- Init trainer related args
        if not hasattr(self, "training_config"):
            self.training_config = None
        if not hasattr(self, "trainer"):
            self.trainer = None

    def _build_from_config(self, config: Union[Dict, ModuleConfig]):
        # -- We check that the input config is valid through pydantic model
        if isinstance(config, dict):
            config = self.config_class(**config)
        self.config = config

        # -- Build from Scratch
        return LightningModule(self.module_class, config)

    def _load_from_path(self, path: str):
        supported_extentions = [".ckpt"]  # prefered order
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"No file or directory at {model_path}")

        if model_path.is_dir():
            found_model = None
            for extention in supported_extentions:
                if list(model_path.glob(f"*{extention}")):
                    # WARNING : Chosing last file if there are many
                    found_model = sorted(model_path.glob(f"*{extention}"))[-1]
                    break
            if found_model is None:
                raise FileNotFoundError(
                    f"No file matching {supported_extentions}"
                    f" was found in directory {model_path}"
                )
            model_path = found_model

        if model_path.suffix == ".ckpt":
            try:
                return LightningModule.load_from_checkpoint(
                    model_path, torch.device("gpu")
                )
            except RuntimeError:
                return LightningModule.load_from_checkpoint(
                    model_path, torch.device("cpu")
                )
        else:
            raise NotImplementedError(
                f"Given file extention {model_path.suffix} "
                "is not supported. Models exported by this module "
                f"and imported to it can be {supported_extentions}"
            )

    def _init_training_config(self, config):
        if isinstance(config, dict):
            config = TrainingConfig(**config)
        self.training_config = config

    def _init_trainer(self, devices=None):
        logger.getChild(PREDICTOR).info(
            "Creating new Trainer instance at %s", self.log_path
        )
        if self.training_config is None:
            self.training_config = TrainingConfig()
        cls_default_params = {arg for arg in inspect.signature(pl.Trainer).parameters}
        kwargs = self.training_config.model_dump(include=cls_default_params)
        lr_monitor = LearningRateMonitor(logging_interval="step")
        progress_bar = LightningProgressBar()
        callbacks = [lr_monitor, progress_bar]
        if devices is not None:
            kwargs["devices"] = devices
        callbacks, profiler = self._init_profilers(callbacks)
        if self.training_config.early_stopping_patience:
            early_stopping = EarlyStopping(
                monitor=self.training_config.early_stopping_value,
                patience=self.training_config.early_stopping_patience,
                mode=self.training_config.early_stopping_mode,
            )
            callbacks.append(early_stopping)
        if self.training_config.seed is not None:
            pl.seed_everything(self.training_config.seed, workers=True)
        if self.training_config.use_deterministic_algorithms:
            torch.use_deterministic_algorithms(True)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            kwargs["deterministic"] = True
        torch.set_float32_matmul_precision("high")
        self.trainer = pl.Trainer(
            default_root_dir=self.log_path,
            logger=self.tb_logger,
            max_epochs=self.training_config.epoch,
            callbacks=callbacks,
            profiler=profiler,
            **kwargs,
        )
        logger.getChild(PREDICTOR).info(
            "Predictor logger initialised at %s", self.log_path
        )

    def _init_profilers(self, callbacks):
        if self.config.used_profiler == "advanced":
            profiler = AdvancedProfiler(
                dirpath=self.log_path, filename="advanced_profiler"
            )
        elif self.config.used_profiler == "simple":
            profiler = SimpleProfiler(dirpath=self.log_path, filename="simple_profiler")
        elif self.config.used_profiler == "torch":
            profiler = PyTorchProfiler(
                dirpath=self.log_path, filename="torch_profiler", emit_nvtx=True
            )
        else:
            profiler = None
        if self.config.used_profiler is not None:
            callbacks.append(DeviceStatsMonitor())
        return callbacks, profiler

    def _free_trainer(self):
        del self.trainer
        self.trainer = None
        torch.cuda.empty_cache()
        gc.collect()

    def _save_config(
        self, save_path: Path, dataset_config: Union[dict, BaseModel, None] = None
    ):
        res = dict()
        if dataset_config:
            if isinstance(dataset_config, BaseModel):
                dataset_config = dataset_config.model_dump(exclude_defaults=False)
            res["dataset_config"] = dataset_config
        if self.training_config is not None:
            res["training_config"] = self.training_config.model_dump(
                exclude_defaults=False
            )
        if self.config is not None:
            res["model_config"] = self.config.model_dump(exclude_defaults=False)
        res["model_config"]["name"] = self.name
        res["model_config"]["version"] = self.version
        with (save_path).open("w", encoding="utf-8") as conf_file:
            json.dump(res, conf_file, indent=4, sort_keys=True)

    def _load_config(self, config_path: Union[Path, str], config_data=None):
        if not config_data:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"No file or directory at {config_path}")
            with config_path.open(encoding="utf-8") as conf_file:
                config_data = json.load(conf_file)
        self.training_config = TrainingConfig(**config_data.get("training_config", {}))
        if self.training_config.accelerator == "cuda" and not torch.cuda.is_available():
            self.training_config.accelerator = "auto"
        self.config = self.config_class(**config_data.get("model_config", {}))
        self.name = config_data.get("model_config", {}).get("name", None)
        self.version = config_data.get("model_config", {}).get("version", None)
        logger.getChild(PREDICTOR).info("Config loaded from %s", config_path)

    def _init_module_optimizer(self):
        self.model.training_config = self.training_config

    def train(
        self,
        train_dataloader: Iterable,
        train_config: TrainingConfig = None,
        val_dataloader: Iterable = None,
    ):
        """train the model"""
        if not train_config:
            train_config = TrainingConfig()
        self._init_training_config(train_config)
        self._init_trainer()
        self._init_module_optimizer()
        if train_config.use_auto_lr:
            # Run learning rate finder
            tuner = Tuner(self.trainer)
            lr_finder = tuner.lr_find(self.model, train_dataloader)
            fig = lr_finder.plot(suggest=True)  # Plot
            self.tb_logger.experiment.add_figure("lr_finder", fig)
            self.model.hparams.lr = lr_finder.suggestion()
            self.training_config.lr = lr_finder.suggestion()
        # Add hyperparams to Tensorboard and init HP Metrics
        self.tb_logger.log_hyperparams(
            {
                **self.model.hparams,
                **self.training_config.model_dump(),
                **self.config.model_dump(),
            },
            {"hp/ADE": -1, "hp/FDE": -1, "hp/MPJPE": -1},
        )
        sample_tensor = train_dataloader.dataset[0][0]
        self.config.num_dims = sample_tensor.shape[2]
        self.config.num_points = sample_tensor.shape[1]
        self.trainer.fit(
            model=self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        # Always save after training
        self.trainer.save_checkpoint(Path(self.log_path) / "trainer_checkpoint.ckpt")
        self._free_trainer()

    def finetune(
        self,
        train_dataloader: Iterable,
        train_config: TrainingConfig = None,
        val_dataloader: Iterable = None,
    ):
        """finetune the model"""
        self.version = None
        self.name = self.name + "_finetuned"
        self._init_logger()
        input_t, truth_t = next(iter(train_dataloader))
        input_shape = input_t.shape
        output_shape = truth_t.shape
        try:
            # try inference
            self.predict(input_t, len(input_t[0]))
        except RuntimeError:
            # adapt model
            model_input_shape = torch.Size(
                (
                    input_shape[0],
                    self.config.input_size,
                    self.config.num_points,
                    self.config.num_dims,
                )
            )
            model_output_shape = torch.Size(
                (
                    input_shape[0],
                    self.config.output_size,
                    self.config.num_points,
                    self.config.num_dims,
                )
            )
            # we update first and last layer with new feature_size
            first_layer = ReshapingLayer(input_shape, model_input_shape)
            last_layer = ReshapingLayer(model_output_shape, output_shape)
            self.model.torch_model = torch.nn.Sequential(
                first_layer, self.model.torch_model, last_layer
            )
        # we update model config with new input output infos
        self.config.input_size = input_t.shape[1]
        self.config.output_size = truth_t.shape[1]
        self.config.num_points = input_t.shape[2]
        self.config.num_dims = input_t.shape[3]
        # train on new dataset
        self.train(train_dataloader, train_config, val_dataloader)

    def test(self, test_dataloader: Iterable):
        """test the model"""
        if self.trainer is None:
            self._init_trainer(devices=1)
        self.trainer.test(self.model, test_dataloader)
        self._free_trainer()

    def save(
        self,
        save_path: Union[str, Path, None] = None,
        dataset_config: Union[dict, BaseModel, None] = None,
    ):
        """save model to path"""
        save_path = str(save_path)
        if save_path is None:
            save_path = self.log_path
        else:  # we cp the tensorflow logger content first
            while True:
                try:
                    shutil.copytree(
                        self.log_path,
                        save_path,
                        ignore=shutil.ignore_patterns("checkpoints"),
                    )
                    break
                except FileExistsError:
                    save_path += "_"
        if isinstance(save_path, str):
            save_path = Path(save_path)
        # save model & config
        save_path.mkdir(parents=True, exist_ok=True)
        logger.getChild(PREDICTOR).info(
            "Saving config at %s", (save_path / "config.json")
        )
        self._save_config(save_path / "config.json", dataset_config)
        # reload logger at new location
        self.log_root_path = save_path
        super()._init_logger(no_sub_dir_log=True)

    def predict(self, input_t: torch.Tensor, future_size: int):
        with torch.no_grad():
            self.model.eval()
            output = self.model.torch_model(input_t, future_size=future_size)
            if is_tensor_is_batched(output):
                return output[:, -future_size:]
            return output[-future_size:]
