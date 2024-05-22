"""simple predictor to use as a baseline"""

from typing import Dict, Optional

import torch
from pytorch_lightning import LightningDataModule

from prescyent.dataset.config import MotionDatasetConfig
from prescyent.dataset.features import convert_tensor_features_to
from prescyent.predictor.base_predictor import BasePredictor
from prescyent.predictor.lightning.configs.training_config import TrainingConfig
from prescyent.utils.logger import logger, PREDICTOR
from prescyent.utils.tensor_manipulation import is_tensor_is_batched


class DelayedPredictor(BasePredictor):
    """simple predictor that simply return the input"""

    def __init__(
        self,
        dataset_config: MotionDatasetConfig,
        log_path: str = "data/models",
        version: int = 0,
    ) -> None:
        super().__init__(dataset_config, log_path, version=version)

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

    def predict(self, input_t: torch.Tensor, future_size: int) -> torch.Tensor:
        unbatch = False
        if not is_tensor_is_batched(input_t):
            unbatch = True
            input_t = input_t.unsqueeze(0)
        input_t = torch.transpose(input_t, 0, 1)
        if future_size > len(input_t):
            new_inputs = [
                input_t[0].unsqueeze(0) for _ in range(future_size - len(input_t))
            ]
            new_inputs = torch.cat(new_inputs, dim=0)
            input_t = torch.cat((new_inputs, input_t), dim=0)
        input_t = torch.transpose(input_t[-future_size:], 0, 1)
        try:
            out_points_ids = torch.LongTensor(
                [
                    self.dataset_config.in_points.index(out)
                    for out in self.dataset_config.out_points
                ]
            )
            out_points_ids = out_points_ids.to(device=input_t.device)
        except ValueError as err:
            raise AttributeError(
                "You cannot use this predictor if output points are not included in input!"
            ) from err
        input_t = torch.index_select(input_t, 2, out_points_ids)
        input_t = convert_tensor_features_to(
            input_t, self.dataset_config.in_features, self.dataset_config.out_features
        )
        if unbatch:
            input_t = input_t.squeeze(0)
        return input_t
