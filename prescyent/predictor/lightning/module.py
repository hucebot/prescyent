"""Module with methods common to every lightning modules"""
import copy
import inspect
from typing import Any, Dict, List, Type

import pytorch_lightning as pl
import torch

from prescyent.dataset.features.feature_manipulation import cal_distance_for_feat
from prescyent.predictor.lightning.configs.module_config import LossFunctions
from prescyent.predictor.lightning.configs.training_config import TrainingConfig
from prescyent.predictor.lightning.losses.mtrd_loss import MeanTotalRigidDistanceLoss
from prescyent.predictor.lightning.losses.mfrd_loss import MeanFinalRigidDistanceLoss
from prescyent.predictor.lightning.torch_module import BaseTorchModule
from prescyent.utils.logger import logger, PREDICTOR


CRITERION_MAPPING = {
    LossFunctions.L1LOSS: torch.nn.L1Loss,
    LossFunctions.MSELOSS: torch.nn.MSELoss,
    LossFunctions.NLLLOSS: torch.nn.NLLLoss,
    LossFunctions.CROSSENTROPYLOSS: torch.nn.CrossEntropyLoss,
    LossFunctions.HINGEEMBEDDINGLOSS: torch.nn.HingeEmbeddingLoss,
    LossFunctions.MARGINRANKINGLOSS: torch.nn.MarginRankingLoss,
    LossFunctions.TRIPLETMARGINLOSS: torch.nn.TripletMarginLoss,
    LossFunctions.KLDIVLOSS: torch.nn.KLDivLoss,
    LossFunctions.MFRDLOSS: MeanFinalRigidDistanceLoss,
    LossFunctions.MTRDLOSS: MeanTotalRigidDistanceLoss,
}
DEFAULT_LOSS = torch.nn.MSELoss


def apply_spectral_norm(model):
    for module_name, module in copy.copy(model._modules).items():
        # recurse on sequentials
        if isinstance(module, torch.nn.Sequential):
            logger.getChild(PREDICTOR).info(
                "Applying Spectral Normalization to %s module",
                module_name,
            )
            setattr(model, module_name, apply_spectral_norm(module))
        # if the module has weights
        elif hasattr(module, "weight") and isinstance(module, torch.nn.Module):
            # apply spectral norm
            module = torch.nn.utils.spectral_norm(module)
            setattr(model, module_name, module)
            logger.getChild(PREDICTOR).info(
                "Applying Spectral Normalization to %s module",
                module_name,
            )
    return model


class LightningModule(pl.LightningModule):
    """Lightning class with methods for modules training, saving, logging"""

    torch_model: BaseTorchModule
    training_config: TrainingConfig
    criterion: torch.nn.modules.loss._Loss
    lr: float
    val_output: List[Dict[str, float]]
    test_output: List[Dict[str, float]]
    train_output: List[Dict[str, float]]

    def __init__(self, torch_model_class: Type[BaseTorchModule], config) -> None:
        logger.getChild(PREDICTOR).info("Initialization of the Lightning Module...")
        super().__init__()
        self.lr = 0.999
        self.torch_model = torch_model_class(config)
        self.val_output = []
        self.test_output = []
        self.train_output = []
        if config.do_lipschitz_continuation:
            logger.getChild(PREDICTOR).info(
                "Parametrization of Lightning Module using the Lipschitz constant..."
            )
            apply_spectral_norm(self.torch_model)
        if not hasattr(self.torch_model, "criterion"):
            if config.loss_fn is None:
                criterion = DEFAULT_LOSS
                logger.getChild(PREDICTOR).info(
                    "No loss function provided in config, using default %s instead",
                    DEFAULT_LOSS,
                )
            else:
                criterion = CRITERION_MAPPING.get(config.loss_fn, None)
                if criterion is None:
                    logger.getChild(PREDICTOR).error(
                        "provided criterion %s is not handled, please use one of the"
                        " following %s.",
                        config.loss_fn,
                        list(CRITERION_MAPPING.keys()),
                    )
                    raise AttributeError(config.loss_fn)
            # Init criterion with config argument if required
            if "config" in inspect.signature(criterion).parameters:
                self.criterion = criterion(config=config)
            else:
                self.criterion = criterion()
            logger.getChild(PREDICTOR).info(
                "Using %s loss function", self.criterion.__class__.__name__
            )
        else:
            # If a child module comes with its own criterion method
            self.criterion = self.torch_model.criterion
            logger.getChild(PREDICTOR).info("Using Predictor's default loss function")
        self.save_hyperparameters(ignore=["torch_model", "criterion"])
        logger.getChild(PREDICTOR).info("Lightning Module ready.")

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    @classmethod
    def load_from_binary(cls, path: str, config):
        """Retrieve model infos from torch binary"""
        model = torch.load(path)
        return cls(model.__class__, config)

    def save(self, save_path: str):
        """Export model to state_dict and torch binary"""
        torch.save(self.torch_model, save_path / "model.pb")

    def configure_optimizers(self):
        """return module optimizer"""
        if not self.training_config.use_auto_lr:
            self.lr = self.training_config.lr
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.training_config.weight_decay,
        )
        if self.training_config.use_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.training_config.max_lr,
                total_steps=self.trainer.estimated_stepping_batches,
            )
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
        return [optimizer]

    def get_metrics(self, batch, prefix: str = "", loss_only=False):
        """get loss and accuracy metrics from batch"""
        sample, truth = batch
        pred = self.torch_model(sample)
        if torch.any(torch.isnan(pred)):
            logger.getChild(PREDICTOR).error("NAN in pred")
            # raise ValueError("Please check your loss function and architecture")
        loss = self.criterion(pred, truth)
        if torch.any(torch.isnan(loss)):
            logger.getChild(PREDICTOR).error("NAN in loss")
            # raise ValueError("Please check your loss function and architecture")
        self.log(f"{prefix}/loss", loss.detach(), prog_bar=True)
        if loss_only:
            return {"loss": loss}
        # Evaluation metrics
        # eval step
        features = self.torch_model.out_features
        feat2distances = dict()
        for feat in features:
            feat2distances[feat.name] = cal_distance_for_feat(
                pred[..., feat.ids], truth[..., feat.ids], feat
            ).detach()
        feat2distances["loss"] = loss
        return feat2distances

    def log_accuracy(self, outputs, prefix: str = "", loss_only=False):
        """log accuracy metrics from epoch"""
        mean_loss = torch.stack([x["loss"] for x in outputs]).mean().detach()
        self.logger.experiment.add_scalar(
            f"{prefix}/loss_epoch", mean_loss, self.current_epoch
        )
        if loss_only:
            return
        # eval epoch
        features = self.torch_model.out_features
        for feat in features:
            batch_feat_distances = torch.cat(
                [feat2distances[feat.name] for feat2distances in outputs]
            )
            ade = batch_feat_distances.mean()
            fde = batch_feat_distances[:, -1].mean()
            mpjpe = (
                batch_feat_distances.transpose(0, 1)
                .reshape(self.torch_model.out_sequence_size, -1)
                .mean(-1)
            )
            self.logger.experiment.add_scalar(
                f"{prefix}/{feat.name}/ADE", ade, self.current_epoch
            )
            self.logger.experiment.add_scalar(
                f"{prefix}/{feat.name}/FDE", fde, self.current_epoch
            )
            for v, val in enumerate(mpjpe):
                self.logger.experiment.add_scalar(
                    f"{prefix}/{feat.name}/MPJPE", val, v + 1
                )
            if prefix in ["Val", "Test"]:
                self.logger.experiment.add_scalar(
                    f"hp/{feat.name}/ADE", ade, self.current_epoch
                )
                self.logger.experiment.add_scalar(
                    f"hp/{feat.name}/FDE", fde, self.current_epoch
                )
                for v, val in enumerate(mpjpe):
                    self.logger.experiment.add_scalar(
                        f"hp/{feat.name}/MPJPE", val, v + 1
                    )

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.val_output = []

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.test_output = []

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self.train_output = []

    def training_step(self, *args, **kwargs):
        """run every training step"""
        batch = args[0]
        res = self.get_metrics(batch, "Train", loss_only=True)
        self.train_output.append(res)
        return res

    def test_step(self, *args, **kwargs):
        """run every test step"""
        with torch.no_grad():
            batch = args[0]
            res = self.get_metrics(batch, "Test")
            self.test_output.append(res)
            return res

    def validation_step(self, *args, **kwargs):
        """run every validation step"""
        with torch.no_grad():
            batch = args[0]
            res = self.get_metrics(batch, "Val")
            self.val_output.append(res)
            return res

    def on_training_epoch_end(self):
        """run every training epoch end"""
        with torch.no_grad():
            self.log_accuracy(self.train_output, "Train", loss_only=True)

    def on_test_epoch_end(self):
        """run every test epoch end"""
        with torch.no_grad():
            self.log_accuracy(self.test_output, "Test")

    def on_validation_epoch_end(self):
        """run every validation epoch end"""
        with torch.no_grad():
            self.log_accuracy(self.val_output, "Val")
