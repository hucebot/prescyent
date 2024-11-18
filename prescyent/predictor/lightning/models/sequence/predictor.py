"""class for sequence based lightning models"""
from typing import Dict, Optional

import torch

from prescyent.predictor.lightning.predictor import LightningPredictor
from prescyent.predictor.base_predictor import BasePredictor
from prescyent.utils.enums.learning_types import LearningTypes
from prescyent.utils.logger import logger, PREDICTOR
from prescyent.utils.tensor_manipulation import is_tensor_is_batched, self_auto_batch


class SequencePredictor(LightningPredictor):
    """Sequence models outputs depend on their trained out_sequence_size
    We reimplement here the predict function to pass a future_size arg to the model
    """

    @self_auto_batch
    @BasePredictor.use_scaler
    def predict(
        self,
        input_t: torch.Tensor,
        future_size: Optional[int] = None,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """run the model / algorithm for one input

        Args:
            input_t (torch.Tensor): tensor to predict over
            future_size (int): number of the expected predicted frames
            context (Optional[Dict[str, torch.Tensor]], optional): additional context. Defaults to None.

        Returns:
            torch.Tensor: predicted tensor
        """
        if context is None:
            context = {}
        with torch.no_grad():
            self.model.eval()
            list_outputs = []
            if future_size is None:
                future_size = self.config.dataset_config.future_size
            # if is tensor and batched, input_len = seq_len's dim, else len()
            history_size = (
                input_t.shape[1] if is_tensor_is_batched(input_t) else len(input_t)
            )
            if (
                hasattr(self.model.torch_model, "in_sequence_size")
                and history_size < self.model.torch_model.in_sequence_size
            ):
                raise AttributeError(
                    "history_size can't be lower than "
                    f"{self.model.torch_model.in_sequence_size}"
                )
            elif (
                hasattr(self.model.torch_model, "in_sequence_size")
                and history_size > self.model.torch_model.in_sequence_size
            ):
                logger.getChild(PREDICTOR).warning(
                    "Input can't be bigger than model in_sequence_size %d"
                    ", the input will be sliced",
                    self.model.torch_model.in_sequence_size,
                )
                input_t = input_t[-self.model.torch_model.in_sequence_size :]
                context = {
                    c_key: c_tensor[-self.model.torch_model.in_sequence_size :]
                    for c_key, c_tensor in context
                }
            if future_size > self.config.dataset_config.future_size and (
                self.config.dataset_config.in_features
                != self.config.dataset_config.out_features
                or self.config.dataset_config.in_points
                != self.config.dataset_config.out_points
                or context
            ):
                raise AttributeError(
                    f"We cannot predict a future_size bigger than "
                    f"{self.config.dataset_config.future_size} if we cannot recurse"
                    " on the model's output or with a context! "
                    " Please check your inputs and outputs'"
                    " in_features and out_features or in_points and out_points"
                )
            device = None
            if input_t.device != self.model.device:
                device = input_t.device
                input_t = input_t.to(self.model.device)
                context = {
                    c_key: c_tensor.to(self.model.device)
                    for c_key, c_tensor in context.items()
                }
            for i in range(0, future_size, self.config.dataset_config.future_size):
                prediction = self.model.torch_model(
                    input_t, future_size=future_size, context=context
                )
                list_outputs.append(prediction)
                if i + self.config.dataset_config.future_size < future_size:
                    if is_tensor_is_batched(input_t):
                        input_t = torch.cat((input_t, prediction), dim=1)[
                            :, -history_size:
                        ]
                    else:
                        input_t = torch.cat((input_t, prediction))[-history_size:]
            if device:
                list_outputs = [pred.to(device) for pred in list_outputs]
            if is_tensor_is_batched(input_t):
                if self.config.dataset_config.learning_type == LearningTypes.SEQ2ONE:
                    return torch.cat(list_outputs, dim=1)[:, -1].unsqueeze(1)
                return torch.cat(list_outputs, dim=1)[:, :future_size]
            else:
                if self.config.dataset_config.learning_type == LearningTypes.SEQ2ONE:
                    return torch.cat(list_outputs, dim=0)[-1].unsqueeze(0)
                return torch.cat(list_outputs, dim=0)[:future_size]
