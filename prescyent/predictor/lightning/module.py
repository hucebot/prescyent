"""Module with methods common to every lightning modules"""
import copy
from typing import Type

import pytorch_lightning as pl
import torch

from prescyent.evaluator.metrics import get_ade, get_fde, get_mpjpe
from prescyent.predictor.lightning.configs.module_config import LossFunctions
from prescyent.predictor.lightning.configs.training_config import TrainingConfig
from prescyent.predictor.lightning.layers.mpjpe_loss import MPJPELoss
from prescyent.predictor.lightning.torch_module import BaseTorchModule
from prescyent.utils.logger import logger, PREDICTOR


CRITERION_MAPPING = {
    LossFunctions.L1LOSS: torch.nn.L1Loss(),
    LossFunctions.MSELOSS: torch.nn.MSELoss(),
    LossFunctions.NLLLOSS: torch.nn.NLLLoss(),
    LossFunctions.CROSSENTROPYLOSS: torch.nn.CrossEntropyLoss(),
    LossFunctions.HINGEEMBEDDINGLOSS: torch.nn.HingeEmbeddingLoss(),
    LossFunctions.MARGINRANKINGLOSS: torch.nn.MarginRankingLoss(),
    LossFunctions.TRIPLETMARGINLOSS: torch.nn.TripletMarginLoss(),
    LossFunctions.KLDIVLOSS: torch.nn.KLDivLoss(),
    LossFunctions.MPJPELOSS: MPJPELoss()
}
DEFAULT_LOSS = MPJPELoss()


def apply_spectral_norm(model):
    for module_name, module in copy.copy(model._modules).items():
        # recurse on sequentials
        if isinstance(module, torch.nn.Sequential):
            logger.info("Applying Spectral Normalization to %s module",
                        module_name, group=PREDICTOR)
            setattr(model, module_name, apply_spectral_norm(module))
        # if the module has weights
        elif hasattr(module, "weight") and isinstance(module, torch.nn.Module):
            # apply spectral norm
            module = torch.nn.utils.spectral_norm(module)
            setattr(model, module_name, module)
            logger.info("Applying Spectral Normalization to %s module",
                        module_name, group=PREDICTOR)
    return model


class LightningModule(pl.LightningModule):
    """Lightning class with methods for modules training, saving, logging"""
    torch_model: BaseTorchModule
    criterion: torch.nn.modules.loss._Loss
    training_config: TrainingConfig

    def __init__(self, torch_model_class: Type[BaseTorchModule], config) -> None:
        logger.info("Initialization of the Lightning Module...", group=PREDICTOR)
        super().__init__()
        self.lr = 0.999
        self.torch_model = torch_model_class(config)
        if config.do_lipschitz_continuation:
            logger.info("Parametrization of Lightning Module using the Lipschitz constant...")
            apply_spectral_norm(self.torch_model)
        if not hasattr(self.torch_model, "criterion"):
            criterion = CRITERION_MAPPING.get(config.loss_fn.lower(), None)
            if criterion is None:
                logger.warning("provided criterion %s is not handled, please use one of the"
                               "following %s.", config.loss_fn.lower(),
                               list(CRITERION_MAPPING.keys()),
                               group=PREDICTOR)
                criterion = DEFAULT_LOSS
        else:
            criterion = self.torch_model.criterion
        logger.info("Using %s loss function", criterion.__class__.__name__, group=PREDICTOR)
        self.criterion = criterion
        self.save_hyperparameters(ignore=['torch_model', 'criterion'])
        logger.info("Lightning Module ready.", group=PREDICTOR)

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
            self.lr = self.training_config.learning_rate
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                      weight_decay=self.training_config.weight_decay)
        if self.training_config.use_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.training_config.max_learning_rate,
                total_steps=self.trainer.estimated_stepping_batches
            )
            return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]
        return [optimizer]

    def get_metrics(self, batch, prefix: str = "", loss_only=False):
        """get loss and accuracy metrics from batch"""
        sample, truth = batch
        pred = self.torch_model(sample)
        loss = self.criterion(pred, truth)
        self.log(f"{prefix}/loss", loss.detach())
        if loss_only:
            return {"loss": loss}
        ade = get_ade(truth, pred).detach()
        fde = get_fde(truth, pred).detach()
        mpjpe = get_mpjpe(truth, pred).detach()[-1]
        return {"loss": loss, "ADE": ade, "FDE": fde, "MPJPE": mpjpe}

    def log_accuracy(self, outputs, prefix: str = "", loss_only=False):
        """log accuracy metrics from epoch"""
        mean_loss = torch.stack([x["loss"] for x in outputs]).mean().detach()
        self.logger.experiment.add_scalar(f"{prefix}/epoch_loss", mean_loss, self.current_epoch)
        if loss_only:
            return
        fde = torch.stack([x["FDE"] for x in outputs]).mean().detach()
        ade = torch.stack([x["ADE"] for x in outputs]).mean().detach()
        mpjpe = torch.stack([x["MPJPE"] for x in outputs]).mean().detach()
        self.logger.experiment.add_scalar(f"{prefix}/FDE", fde, self.current_epoch)
        self.logger.experiment.add_scalar(f"{prefix}/ADE", ade, self.current_epoch)
        self.logger.experiment.add_scalar(f"{prefix}/MPJPE", mpjpe, self.current_epoch)
        if prefix in ["Val", "Test"]:
            self.logger.experiment.add_scalar("hp/FDE", fde, self.current_epoch)
            self.logger.experiment.add_scalar("hp/ADE", ade, self.current_epoch)
            self.logger.experiment.add_scalar("hp/MPJPE", mpjpe, self.current_epoch)

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
