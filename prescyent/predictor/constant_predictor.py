"""simple predictor to use as a baseline"""

from typing import Dict, Iterable

import torch
from pydantic import BaseModel

from prescyent.evaluator import get_ade, get_fde
from prescyent.predictor.base_predictor import BasePredictor
from prescyent.utils.logger import logger, PREDICTOR
from prescyent.utils.tensor_manipulation import is_tensor_batched


class ConstantPredictor(BasePredictor):
    """simple predictor that simply return the last input"""

    def __init__(self, log_path: str = "data/models") -> None:
        super().__init__(log_path, version=0)

    def _build_from_config(self, config: Dict):
        """build predictor from a config"""
        logger.warning("No config necessary for this predictor %s",
                       self.__class__.__name__,
                       group=PREDICTOR)

    def train(self, train_dataloader: Iterable,
              train_config: BaseModel = None,
              val_dataloader: Iterable = None):
        """train predictor"""
        logger.warning("No training necessary for this predictor %s",
                       self.__class__.__name__,
                       group=PREDICTOR)

    def save(self, save_path: str):
        """train predictor"""
        logger.warning("No save necessary for this predictor %s",
                       self.__class__.__name__,
                       group=PREDICTOR)

    def test(self, test_dataloader: Iterable):
        """test predictor"""
        # log in tensorboard
        losses, ades, fdes = [], [], []
        for sample, truth in test_dataloader:
            # eval step
            pred = self.get_prediction(sample, len(truth[0]))
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

    def run(self, input_batch: Iterable, history_size: int,
            history_step: int = None, future_size: int = None,
            output_only_future: bool = None):
        """run predictor"""
        # -- If no history_size is given or relevant, return the model over the whole input
        if history_size is None or history_size >= input_batch.shape[0]:
            predictions = self.get_prediction(input_batch, future_size)
            return predictions
        # -- Else we iterate over inputs of len history_size, with step history_step
        # and return a list of predictions
        prediction_list = []
        for i in range(0, input_batch.shape[0] - history_size, history_step):
            input_sub_batch = input_batch[i:i + history_size]
            prediction = self.get_prediction(input_sub_batch, future_size)
            prediction_list.append(prediction)
        return prediction_list


    def get_prediction(self, input_t, history_size):
        if is_tensor_batched(input_t):
            input_t = torch.transpose(input_t, 0, 1)
            output = [input_t[-1].unsqueeze(0) for _ in range(history_size)]
            output_t = torch.cat(output, dim=0)
            return torch.transpose(output_t, 0, 1)
        output = [input_t[-1].unsqueeze(0) for _ in range(history_size)]
        return torch.cat(output, dim=0)
