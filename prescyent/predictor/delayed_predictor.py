"""simple predictor to use as a baseline"""

from typing import Dict, Optional

import torch
from pytorch_lightning import LightningDataModule

from prescyent.dataset.features import convert_tensor_features_to
from prescyent.predictor.base_predictor import BasePredictor
from prescyent.predictor.config import PredictorConfig
from prescyent.predictor.lightning.configs.training_config import TrainingConfig
from prescyent.utils.logger import logger, PREDICTOR
from prescyent.utils.tensor_manipulation import self_auto_batch


class DelayedPredictor(BasePredictor):
    """simple predictor that simply return the input"""

    def __init__(
        self,
        config: Optional[PredictorConfig],
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

    @self_auto_batch
    @BasePredictor.use_scaler
    def predict(
        self, input_t: torch.Tensor, future_size: int, *args, **kwargs
    ) -> torch.Tensor:
        input_t = torch.transpose(input_t, 0, 1)
        if future_size > len(input_t):
            new_inputs = [
                input_t[0].unsqueeze(0) for _ in range(future_size - len(input_t))
            ]
            new_inputs = torch.cat(new_inputs, dim=0)
            input_t = torch.cat((new_inputs, input_t), dim=0)
        output_t = torch.transpose(input_t[-future_size:], 0, 1)
        if self.config is not None:
            try:
                out_points_ids = torch.LongTensor(
                    [
                        self.config.dataset_config.in_points.index(out)
                        for out in self.config.dataset_config.out_points
                    ]
                )
                out_points_ids = out_points_ids.to(device=output_t.device)
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
        return output_t
