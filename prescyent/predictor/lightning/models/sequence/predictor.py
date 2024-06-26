"""class for sequence based lightning models"""
import torch

from prescyent.predictor.lightning.predictor import LightningPredictor
from prescyent.utils.logger import logger, PREDICTOR
from prescyent.utils.tensor_manipulation import is_tensor_is_batched


class SequencePredictor(LightningPredictor):
    """Sequence models outputs depend on their trained out_sequence_size
    We reimplement here the predict function to pass a future_size arg to the model
    """

    def predict(
        self,
        input_t: torch.Tensor,
        future_size: int = None,
    ):
        with torch.no_grad():
            self.model.eval()
            list_outputs = []
            if future_size is None:
                future_size = self.model.torch_model.out_sequence_size
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
            if future_size > self.model.torch_model.out_sequence_size and (
                self.dataset_config.in_features != self.dataset_config.out_features
                or self.dataset_config.in_points != self.dataset_config.out_points
            ):
                raise AttributeError(
                    f"We cannot predict a futur_size bigger than "
                    f"{self.model.torch_model.out_sequence_size} with different "
                    "in_features and out_features or "
                    "in_points and out_points "
                    f"{self.model.torch_model.in_sequence_size}"
                )
            for i in range(0, future_size, self.model.torch_model.out_sequence_size):
                prediction = self.model.torch_model(input_t)
                list_outputs.append(prediction)
                if i + self.model.torch_model.out_sequence_size < future_size:
                    if is_tensor_is_batched(input_t):
                        input_t = torch.cat((input_t, prediction), dim=1)[
                            :, -history_size:
                        ]
                    else:
                        input_t = torch.cat((input_t, prediction))[-history_size:]
            if is_tensor_is_batched(input_t):
                return torch.cat(list_outputs, dim=0)[:, :future_size]
            else:
                return torch.cat(list_outputs, dim=0)[:future_size]
