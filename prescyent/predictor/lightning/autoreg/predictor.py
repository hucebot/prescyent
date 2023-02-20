"""class for auto regressive lightning models"""
from typing import Iterable

import torch

from prescyent.predictor.lightning.predictor import LightningPredictor


class AutoRegPredictor(LightningPredictor):
    """Auto Regressive models outputs history[1:] + (1 + future)
        We reimplement here the run function to pass a future arg to the model
        and to allow to output only the future timesteps of the trajectory
    """

    def run(self, input_batch: Iterable, history_size: int = None,
            history_step: int = 1, future_size: int = 0,
            output_only_future: bool = True) -> torch.Tensor:
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
            # -- If no history_size is given or relevant, return the model over the whole input
            if history_size is None or history_size >= input_batch.shape[0]:
                predictions = self.model.torch_model(input_batch, future=future_size)
                if output_only_future:
                    predictions = predictions[-(future_size + 1):]
                return predictions

            # -- Else we iterate over inputs of len history_size, with step history_step
            # and return a list of predictions
            prediction_list = []
            for i in range(0, input_batch.shape[0] - history_size, history_step):
                input_sub_batch = input_batch[i:i + history_size]
                prediction = self.model.torch_model(input_sub_batch, future=future_size)
                if output_only_future:
                    prediction_list.append(prediction[-(future_size):])
                else:
                    prediction_list.append(prediction)
            return prediction_list
