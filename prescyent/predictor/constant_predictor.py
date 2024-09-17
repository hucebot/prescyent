"""simple predictor to use as a baseline"""

from typing import Dict, Iterable, Optional

import torch
from pytorch_lightning import LightningDataModule

from prescyent.dataset.features import convert_tensor_features_to
from prescyent.predictor.base_predictor import BasePredictor
from prescyent.predictor.config import PredictorConfig
from prescyent.predictor.lightning.configs.training_config import TrainingConfig
from prescyent.utils.logger import logger, PREDICTOR
from prescyent.utils.tensor_manipulation import is_tensor_is_batched


class ConstantPredictor(BasePredictor):
    """simple predictor that simply return the last input"""

    def __init__(
        self,
        config: Optional[PredictorConfig] = None,
    ) -> None:
        super().__init__(config)

    def _build_from_config(self, config: Dict):
        """build predictor from a config"""
        logger.getChild(PREDICTOR).warning(
            "No config necessary for this predictor %s",
            self.__class__.__name__,
        )

    def train(
        self,
        datamodule: LightningDataModule,
        train_config: Optional[TrainingConfig] = None,
    ):
        """train predictor"""
        logger.getChild(PREDICTOR).warning(
            "No training necessary for this predictor %s",
            self.__class__.__name__,
        )

    def finetune(
        self,
        datamodule: LightningDataModule,
        train_config: Optional[TrainingConfig] = None,
    ):
        """finetune predictor"""
        logger.getChild(PREDICTOR).warning(
            "No training necessary for this predictor %s",
            self.__class__.__name__,
        )

    def save(self, save_path: str):
        """train predictor"""
        logger.getChild(PREDICTOR).warning(
            "No save necessary for this predictor %s",
            self.__class__.__name__,
        )

    def predict(
        self, input_t: torch.Tensor, future_size: int, *args, **kwargs
    ) -> torch.Tensor:
        unbatch = False
        if not is_tensor_is_batched(input_t):
            unbatch = True
            input_t = input_t.unsqueeze(0)
        input_t = torch.transpose(input_t, 0, 1)
        output = [input_t[-1].unsqueeze(0) for _ in range(future_size)]
        output_t = torch.cat(output, dim=0)
        output_t = torch.transpose(output_t, 0, 1)
        if self.config is not None:
            try:
                out_points_ids = torch.LongTensor(
                    [
                        self.config.dataset_config.in_points.index(out)
                        for out in self.config.dataset_config.out_points
                    ]
                )
                out_points_ids = out_points_ids.to(device=input_t.device)
            except ValueError as err:
                raise AttributeError(
                    "You cannot use this predictor if output points are not included in input!"
                ) from err
            output_t = torch.index_select(output_t, 2, out_points_ids)
            output_t = convert_tensor_features_to(
                output_t,
                self.config.dataset_config.in_features,
                self.config.dataset_config.out_features,
            )
        if unbatch:
            output_t = output_t.squeeze(0)
        return output_t
