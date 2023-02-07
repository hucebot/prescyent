"""simple predictor to use as a baseline"""

from typing import Dict, Iterable

import torch

from prescyent.evaluator import get_ade, get_fde
from prescyent.predictor.base_predictor import BasePredictor
from prescyent.utils.logger import logger, PREDICTOR


class DelayedPredictor(BasePredictor):
    """simple predictor that simply return the input"""

    def __init__(self, log_path: str="data/models") -> None:
        self.log_root_path = log_path
        super().__init__()

    def _build_from_config(self, config: Dict):
        """build predictor from a config"""
        logger.warning("No config necessary for this predictor %s",
                       self.__class__.__name__,
                       group=PREDICTOR)
        return

    def train(self, *args, **kwargs):
        """train predictor"""
        logger.warning("No training necessary for this predictor %s",
                       self.__class__.__name__,
                       group=PREDICTOR)
        return

    def test(self, test_dataloader: Iterable):
        """test predictor"""
        # log in tensorboard
        losses, ades, fdes = list(), list(), list()
        for sample, truth in test_dataloader:
            # eval step
            pred = self.run(sample)
            losses.append(torch.nn.MSELoss()(pred, truth))
            ades.append(get_ade(truth, pred))
            fdes.append(get_fde(truth, pred))
        # eval epoch
        mean_loss = torch.stack(losses).mean()
        ade = torch.stack(ades).mean()
        fde = torch.stack(fdes).mean()
        self.tb_logger.experiment.add_scalar("Test/epoch_loss", mean_loss, 0)
        self.tb_logger.experiment.add_scalar("Test/ADE", ade, 0)
        self.tb_logger.experiment.add_scalar("Test/FDE", fde, 0)
        return mean_loss, ade, fde

    def run(self, input_batch: Iterable, input_size: int=None, input_step: int=1):
        """run predictor"""
        return input_batch
