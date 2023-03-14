"""Module with methods common to every lightning modules"""
from abc import abstractmethod
import functools
from typing import Type

import pytorch_lightning as pl
import torch

from prescyent.evaluator.metrics import get_ade, get_fde, get_mpjpe
from prescyent.predictor.lightning.training_config import TrainingConfig
from prescyent.utils.logger import logger, PREDICTOR


CRITERION_MAPPING = {
    "l1loss": torch.nn.L1Loss(),
    "mseloss": torch.nn.MSELoss(),
    "nllloss": torch.nn.NLLLoss(),
    "crossentropyloss": torch.nn.CrossEntropyLoss(),
    "hingeembeddingloss": torch.nn.HingeEmbeddingLoss(),
    "marginrankingloss": torch.nn.MarginRankingLoss(),
    "tripletmarginloss": torch.nn.TripletMarginLoss(),
    "kldivloss": torch.nn.KLDivLoss()
}


class BaseTorchModule(torch.nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.do_normalization = config.do_normalization

    @abstractmethod
    def forward(self, input_tensor: torch.Tensor, future_size: int):
        pass

    @classmethod
    def normalize_tensor_from_last_value(cls, function):
        """decorator for normalization of the input tensor before forward method"""
        @functools.wraps(function)
        def normalize(*args, **kwargs):
            self = args[0]
            input_tensor = args[1]
            if self.do_normalization:
                seq_last = input_tensor[:, -1:, :, :].detach()
                input_tensor = input_tensor - seq_last
            predictions = function(self, input_tensor, **kwargs)
            if self.do_normalization:
                predictions = predictions + seq_last
            return predictions
        return normalize

    @classmethod
    def allow_unbatched(cls, function):
        """decorator for seemless batched/unbatched forward methods"""
        @functools.wraps(function)
        def reshape(*args, **kwargs):
            self = args[0]
            input_tensor = args[1]
            unbatched = len(input_tensor.shape) == 3
            if unbatched:
                input_tensor = torch.unsqueeze(input_tensor, dim=0)
            predictions = function(self, input_tensor, **kwargs)
            if unbatched:
                predictions = torch.squeeze(predictions, dim=0)
            return predictions
        return reshape


class LightningModule(pl.LightningModule):
    """Lightning class with methods for modules training, saving, logging"""
    torch_model: BaseTorchModule
    criterion: torch.nn.modules.loss._Loss
    training_config: TrainingConfig

    def __init__(self, torch_model_class: Type[BaseTorchModule], config) -> None:
        super().__init__()
        self.torch_model = torch_model_class(config)
        criterion = CRITERION_MAPPING.get(config.criterion.lower(), None)
        if criterion is None:
            logger.warning("provided criterion %s is not handled, please use one of the"
                           "following %s. Using default MSELoss instead", config.criterion.lower(),
                           list(CRITERION_MAPPING.keys()),
                           group=PREDICTOR)
            criterion = torch.nn.MSELoss()
        self.criterion = criterion
        self.save_hyperparameters(ignore=['torch_model', 'criterion'])

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.training_config.learning_rate,
                                      weight_decay=self.training_config.weight_decay)
        if self.training_config.use_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.training_config.max_learning_rate,
                total_steps=self.trainer.estimated_stepping_batches
            )
            return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]
        return [optimizer]

    def get_metrics(self, batch, prefix: str = ""):
        """get loss and accuracy metrics from batch"""
        sample, truth = batch
        pred = self.torch_model(sample)
        loss = self.criterion(pred, truth)
        ade = get_ade(truth, pred)
        fde = get_fde(truth, pred)
        mpjpe = get_mpjpe(truth, pred)
        self.log(f"{prefix}/loss", loss)
        return {"loss": loss, "ADE": ade, "FDE": fde, "MPJPE": mpjpe}

    def log_accuracy(self, outputs, prefix: str = ""):
        """log accuracy metrics from epoch"""
        mean_loss = torch.stack([x["loss"] for x in outputs]).mean()
        fde = torch.stack([x["FDE"] for x in outputs]).mean()
        ade = torch.stack([x["ADE"] for x in outputs]).mean()
        mpjpe = torch.stack([x["MPJPE"] for x in outputs]).mean()
        self.logger.experiment.add_scalar(f"{prefix}/epoch_loss", mean_loss, self.current_epoch)
        self.logger.experiment.add_scalar(f"{prefix}/FDE", fde, self.current_epoch)
        self.logger.experiment.add_scalar(f"{prefix}/ADE", ade, self.current_epoch)
        self.logger.experiment.add_scalar(f"{prefix}/MPJPE", mpjpe, self.current_epoch)

    def training_step(self, *args, **kwargs):
        """run every training step"""
        batch = args[0]
        return self.get_metrics(batch, "Train")

    def test_step(self, *args, **kwargs):
        """run every test step"""
        batch = args[0]
        return self.get_metrics(batch, "Test")

    def validation_step(self, *args, **kwargs):
        """run every validation step"""
        batch = args[0]
        return self.get_metrics(batch, "Val")

    def test_epoch_end(self, outputs):
        """run every test epoch end"""
        self.log_accuracy(outputs, "Test")

    def training_epoch_end(self, outputs):
        """run every training epoch end"""
        self.log_accuracy(outputs, "Train")

    def validation_epoch_end(self, outputs):
        """run every validation epoch end"""
        self.log_accuracy(outputs, "Val")
