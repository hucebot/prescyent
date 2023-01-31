
import functools

import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import nn

from prescyent.evaluator.metrics import get_ade, get_fde


def allow_unbatched(function):
    @functools.wraps(function)
    def reshape(*args, **kwargs):
        self = args[0]
        x = args[1]
        unbatched = len(x.shape) == 2
        if unbatched:
            x = torch.unsqueeze(x, dim=0)
        predictions = function(self, x, **kwargs)
        if unbatched:
            predictions = torch.squeeze(predictions, dim=0)
        return predictions
    return reshape


class BaseLightningModule(pl.LightningModule):
    torch_model: nn.Module
    criterion: torch.nn.modules.loss._Loss

    def save(self, save_path: str):
        """Export model to state_dict and torch binary"""
        torch.save(self.torch_model, save_path / "model.pb")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def get_metrics(self, batch, prefix: str = ""):
        """get loss and accuracy metrics from batch"""
        sample, truth = batch
        pred = self.torch_model(sample)
        loss = self.criterion(pred, truth)
        ade = get_ade(truth, pred)
        fde = get_fde(truth, pred)
        self.log(f"{prefix}/loss", loss)
        return {"loss": loss, "ADE": ade, "FDE": fde}

    def log_accuracy(self, outputs, prefix: str = ""):
        """log accuracy metrics from epoch"""
        mean_loss = torch.stack([x["loss"] for x in outputs]).mean()
        fde = torch.stack([x["FDE"] for x in outputs]).mean()
        ade = torch.stack([x["ADE"] for x in outputs]).mean()
        self.logger.experiment.add_scalar(f"{prefix}/epoch_loss", mean_loss, self.current_epoch)
        self.logger.experiment.add_scalar(f"{prefix}/FDE", fde, self.current_epoch)
        self.logger.experiment.add_scalar(f"{prefix}/ADE", ade, self.current_epoch)

    def training_step(self, batch, batch_idx):
        return self.get_metrics(batch, "Train")

    def test_step(self, batch, batch_idx):
        return self.get_metrics(batch, "Test")

    def validation_step(self, batch, batch_idx):
        return self.get_metrics(batch, "Val")

    def test_epoch_end(self, outputs):
        self.log_accuracy(outputs, "Test")

    def training_epoch_end(self, outputs):
        self.log_accuracy(outputs, "Train")

    def validation_epoch_end(self, outputs):
        self.log_accuracy(outputs, "Val")
