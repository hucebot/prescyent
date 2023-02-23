"""class for auto regressive lightning models"""
from typing import Iterable

import torch

from prescyent.predictor.lightning.predictor import LightningPredictor
from prescyent.utils.logger import logger, PREDICTOR


class SequencePredictor(LightningPredictor):
    """Sequence models outputs depend on their trained output_size
        We reimplement here the run function to pass a future_size arg to the model
    """

    def run(self, input_batch: Iterable, history_size: int = None,
            history_step: int = 1, future_size: int = 0,
            output_only_future: bool = None) -> torch.Tensor:
        """run method/model inference on the input batch
        The output is either the list of predictions for each defined subpart of the input batch,
        or the single prediction for the whole input

        Args:
            input_batch (Iterable): Input for the predictor's model
            history_size (int|None, optional): If an input size is provided, the input batch will
                be splitted sequences of len == history_size. Defaults to None.
            history_step (int, optional): When splitting the input_batch (history_size != None)
                defines the step of the iteration. Defaults to 1.

        Returns:
            torch.Tensor | List[torch.Tensor]: the model prediction or list of model predictions
        """
        with torch.no_grad():
            self.model.eval()
            if future_size is not None and self.model.torch_model.output_size < future_size:
                logger.warning("This predictor cannot output a sequence lower than %s",
                            self.model.torch_model.output_size,
                            group=PREDICTOR)
            elif future_size is None:
                future_size = self.model.torch_model.output_size
            list_outputs = []
            # if is tensor and batched, input_len = seq_len's dim, else len()
            input_len = input_batch.shape[1] \
                if isinstance(input_batch, torch.Tensor) and len(input_batch.shape) == 3 \
                else len(input_batch)
            # -- If no history_size is given or relevant, return the model over the whole input
            if history_size is None or history_size >= input_batch.shape[0]:
                for _ in range(0, future_size, self.model.torch_model.output_size):
                    prediction = self.model.torch_model(input_batch)
                    list_outputs.append(prediction)
                    # We iter over pred
                    input_batch = torch.cat((input_batch, prediction))[-input_len:]
                # If we predicted more, we output only future_size
                return torch.cat(list_outputs, dim=0)[:future_size]

            # -- Else we iterate over inputs of len history_size, with step history_step
            # and return a list of predictions of len == future_size
            prediction_list = []
            for j in range(0, input_batch.shape[0] - history_size, history_step):
                list_outputs = []
                input_step = input_batch[j:j + history_size]
                for _ in range(0, future_size, self.model.torch_model.output_size):
                    prediction = self.model.torch_model(input_step)
                    list_outputs.append(prediction)
                    input_step = torch.cat((input_step, prediction))[-history_size:]
                prediction_list.append(torch.cat(list_outputs, dim=0)[:future_size])
            return prediction_list
