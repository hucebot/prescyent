"""Lightning Predictor class for ML predictors"""
import gc
import inspect
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Type, Union

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
    ModelCheckpoint,
)
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from prescyent.predictor.base_predictor import BasePredictor
from prescyent.predictor.lightning.module import LightningModule
from prescyent.predictor.lightning.torch_module import BaseTorchModule
from prescyent.predictor.lightning.configs.module_config import ModuleConfig
from prescyent.predictor.lightning.callbacks.progress_bar import LightningProgressBar
from prescyent.predictor.lightning.configs.training_config import TrainingConfig
from prescyent.scaler.scaler import Scaler
from prescyent.utils.logger import logger, PREDICTOR
from prescyent.utils.tensor_manipulation import is_tensor_is_batched
from prescyent.predictor.lightning.layers.reshaping_layer import ReshapingLayer


MODEL_CHECKPOINT_NAME = "model_checkpoint.ckpt"


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

    def __init__(self, config, name=None, skip_build: bool = False) -> None:
        self.config = config
        if not skip_build:
            if config is None:
                raise AttributeError("We cannot build a new predictor without a config")
            self.model = self._build_from_config(config)
        if self.config.name is None:  # Default name if none in self.config
            self.config.name = name
        super().__init__(
            self.config,
        )
        # -- Init trainer related args
        if not hasattr(self, "training_config"):
            self.training_config = None
        if not hasattr(self, "trainer"):
            self.trainer = None
        if self.config.name is None:
            self.config.name = self.name

    @classmethod
    def load_pretrained(cls, model_dir: Union[Path, str]) -> object:
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)
        # ensure we have the folder and not the file
        model_dir = model_dir if model_dir.is_dir() else model_dir.parent
        config = cls._load_config(model_dir / "config.json")
        predictor = cls(config, skip_build=True)
        predictor.model = predictor._load_from_path(model_dir)
        # Load scaler if any
        if config.scaler_config:
            try:
                predictor.scaler = Scaler.load(Path(model_dir) / "scaler.pkl")
            except FileNotFoundError:
                logger.getChild(PREDICTOR).warning(
                    f"Could not retreive scaler at path {Path(model_dir) / 'scaler.pkl'}"
                )
        return predictor

    @classmethod
    def _load_config(cls, config_path: Path):
        if not config_path.exists():
            raise FileNotFoundError(f"No file or directory at {config_path}")
        with config_path.open(encoding="utf-8") as conf_file:
            config_data = json.load(conf_file)
        config = cls.config_class(**config_data.get("model_config", {}))
        return config

    def _build_from_config(self, config: Union[Dict, ModuleConfig]):
        # -- We check that the input config is valid through pydantic model
        if isinstance(config, dict):
            config = self.config_class(**config)
        self.config = config
        # -- Build from Scratch
        return LightningModule(self.module_class, config)

    def _load_from_path(self, path: str):
        supported_extentions = [".ckpt"]
        model_path = Path(path)
        if model_path.name == "config.json":
            model_path = model_path.parent
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
        logger.getChild(PREDICTOR).info(f"Loading model from {model_path}")
        if model_path.suffix == ".ckpt":
            try:
                return LightningModule.load_from_checkpoint(
                    model_path, map_location=torch.device("gpu")
                )
            except RuntimeError:
                return LightningModule.load_from_checkpoint(
                    model_path, map_location=torch.device("cpu")
                )
        else:
            raise NotImplementedError(
                f"Given file extention {model_path.suffix} "
                "is not supported. Models exported by this module "
                f"and imported to it can be {supported_extentions}"
            )

    def _init_training_config(self, config):
        if self.config is None:
            config = TrainingConfig()
        elif isinstance(config, dict):
            config = TrainingConfig(**config)
        self.training_config = config
        self.model.lr = self.training_config.lr

    def _init_trainer(self, devices=None):
        logger.getChild(PREDICTOR).info(
            "Creating new Trainer instance at %s", self.log_path
        )
        if self.training_config is None:
            self._init_training_config(None)
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
            self.checkpoint_callback = ModelCheckpoint(
                save_top_k=self.training_config.early_stopping_patience,
                monitor=self.training_config.early_stopping_value,
                mode=self.training_config.early_stopping_mode,
            )
            callbacks.append(self.checkpoint_callback)
            kwargs["enable_checkpointing"] = (True,)
        if self.training_config.seed is not None:
            pl.seed_everything(self.training_config.seed, workers=True)
        if self.training_config.use_deterministic_algorithms:
            torch.use_deterministic_algorithms(True)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            kwargs["deterministic"] = True
        torch.set_float32_matmul_precision("high")
        try:
            self.trainer = pl.Trainer(
                default_root_dir=self.log_path,
                logger=self.tb_logger,
                callbacks=callbacks,
                profiler=profiler,
                **kwargs,
            )
        except MisconfigurationException:
            kwargs["accelerator"] = "auto"
            self.trainer = pl.Trainer(
                default_root_dir=self.log_path,
                logger=self.tb_logger,
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

    def free_trainer(self):
        del self.trainer
        self.trainer = None
        torch.cuda.empty_cache()
        gc.collect()

    def _save_config(self, save_path: Path):
        res = dict()
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

    def _init_module_optimizer(self):
        self.model.training_config = self.training_config

    def train(
        self,
        datamodule: pl.LightningDataModule,
        train_config: TrainingConfig = None,
    ):
        """train the model"""
        super().train(datamodule=datamodule, train_config=train_config)
        if not train_config:
            train_config = TrainingConfig()
        self._init_training_config(train_config)
        self._init_trainer()
        self._init_module_optimizer()
        self.model.scaler = self.scaler
        if train_config.use_auto_lr:
            # Run learning rate finder
            tuner = Tuner(self.trainer)
            lr_finder = tuner.lr_find(self.model, datamodule=datamodule)
            fig = lr_finder.plot(suggest=True)  # Plot
            self.tb_logger.experiment.add_figure("lr_finder", fig)
            self.model.hparams.lr = lr_finder.suggestion()
            self.training_config.lr = lr_finder.suggestion()
        # Add hyperparams to Tensorboard and init HP Metrics
        hp_metrics = {}
        for feat in self.config.dataset_config.out_features:
            hp_metrics[f"hp/{feat.name}/ADE"] = -1
            hp_metrics[f"hp/{feat.name}/FDE"] = -1
            hp_metrics[f"hp/{feat.name}/MPJPE"] = -1
        self.tb_logger.log_hyperparams(
            {
                **self.model.hparams,
                **self.training_config.model_dump(),
                **self.config.model_dump(exclude=["dataset"]),
            },
            hp_metrics,
        )
        # pass scaler to module before training
        self.trainer.fit(
            model=self.model,
            datamodule=datamodule,
        )
        # Save model checkpoint
        # Retreive best checkpoint if one exists
        if hasattr(self, "checkpoint_callback"):
            logger.getChild(PREDICTOR).info(
                f"Reloading best model from checkpoints {self.checkpoint_callback.best_model_path}"
            )
            self.model = self._load_from_path(self.checkpoint_callback.best_model_path)
            shutil.copy(
                self.checkpoint_callback.best_model_path,
                Path(self.log_path) / MODEL_CHECKPOINT_NAME,
            )
        else:
            self.trainer.save_checkpoint(Path(self.log_path) / MODEL_CHECKPOINT_NAME)
        self.free_trainer()

    def finetune(
        self,
        datamodule: pl.LightningDataModule,
        train_config: TrainingConfig = None,
    ):
        """finetune the model"""
        self.version = None
        self.name = self.name + "_finetuned"
        self._init_logger()
        input_t, context_t, truth_t = next(iter(datamodule.train_dataloader()))
        input_shape = input_t.shape
        output_shape = truth_t.shape
        try:
            # try inference
            self.predict(input_t, future_size=len(input_t[0]), context=context_t)
        except RuntimeError:
            # adapt model
            model_input_shape = torch.Size(
                (
                    input_shape[0],
                    self.config.in_sequence_size,
                    len(self.config.dataset_config.in_points),
                    len(self.config.dataset_config.in_dims),
                )
            )
            model_output_shape = torch.Size(
                (
                    input_shape[0],
                    self.config.out_sequence_size,
                    len(self.config.dataset_config.out_points),
                    len(self.config.dataset_config.out_dims),
                )
            )
            # we update first and last layer with new feature_size
            first_layer = ReshapingLayer(input_shape, model_input_shape)
            last_layer = ReshapingLayer(model_output_shape, output_shape)
            self.model.torch_model = torch.nn.Sequential(
                first_layer, self.model.torch_model, last_layer
            )
        # we update model config with new input output infos
        self.config.in_sequence_size = input_t.shape[1]
        self.config.out_sequence_size = truth_t.shape[1]
        self.config.dataset_config.in_points = list(range(input_t.shape[2]))
        self.config.dataset_config.in_dims = list(range(input_t.shape[3]))
        self.config.dataset_config.out_points = list(range(truth_t.shape[2]))
        self.config.dataset_config.out_dims = list(range(truth_t.shape[3]))
        # train on new dataset
        self.train(train_config=train_config, datamodule=datamodule)

    def test(self, datamodule: pl.LightningDataModule):
        """test the model"""
        if self.trainer is None:
            self._init_trainer(devices=1)
        # pass scaler to module before training
        self.model.scaler = self.scaler
        losses = self.trainer.test(self.model, datamodule=datamodule)
        self.free_trainer()
        return losses

    def save(
        self,
        save_path: Union[str, Path, None] = None,
        rm_log_path: bool = True,
    ):
        """save model to path"""
        if save_path is None:
            save_path = self.log_path
        else:  # we cp the tensorflow logger content first
            save_path = str(save_path)
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
                except FileNotFoundError:
                    # Nothing to copy
                    break
        if isinstance(save_path, str):
            save_path = Path(save_path)
        # save model & config
        save_path.mkdir(parents=True, exist_ok=True)
        logger.getChild(PREDICTOR).info(
            "Saving config at %s", (save_path / "config.json")
        )
        self._save_config(save_path / "config.json")
        if rm_log_path and Path(self.log_path).resolve() != save_path.resolve():
            shutil.rmtree(self.log_path, ignore_errors=True)
        # reload logger at new location
        self.log_root_path = save_path
        super()._init_logger(no_sub_dir_log=True)
        # save scaler instance if any
        if self.scaler is not None:
            self.scaler.save(save_path / "scaler.pkl")

    @BasePredictor.use_scaler
    def predict(
        self,
        input_t: torch.Tensor,
        future_size: int,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ):
        with torch.no_grad():
            self.model.eval()
            output = self.model.torch_model(
                input_t, future_size=future_size, context=context
            )
            if is_tensor_is_batched(output):
                return output[:, -future_size:]
            return output[-future_size:]
