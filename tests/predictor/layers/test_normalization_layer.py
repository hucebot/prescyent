import torch
from pydantic import ValidationError

from prescyent.dataset.config import MotionDatasetConfig
from prescyent.predictor.lightning.layers.normalization_layer import MotionLayerNorm
from prescyent.predictor.lightning.configs.module_config import ModuleConfig
from prescyent.utils.enums import Normalizations
from tests.custom_test_case import CustomTestCase


class MotionLayerNormTests(CustomTestCase):
    def test_all_norm(self):
        """test all norm init and forward"""
        input_size = 10
        output_size = 5
        in_dims = [0, 1, 2]
        in_points = [0, 1]
        input_t = torch.zeros(64, input_size, len(in_points), len(in_dims))
        config = ModuleConfig(
            dataset_config=MotionDatasetConfig(
                future_size=output_size,
                history_size=input_size,
                in_dims=in_dims,
                in_points=in_points,
            ),
            used_norm=Normalizations.ALL,
        )
        normalization_layer = MotionLayerNorm(config)
        output_t = normalization_layer(input_t)
        self.assertEqual(input_t.shape, output_t.shape)
        with self.assertRaises(ValidationError) as _:
            config = ModuleConfig(
                dataset_config=MotionDatasetConfig(
                    future_size=output_size,
                    history_size=input_size,
                    in_points=in_points,
                ),
                used_norm=Normalizations.ALL,
            )

    def test_spatial_norm(self):
        """test spatial norm init and forward"""
        input_size = 10
        output_size = 5
        in_dims = [0, 1, 2]
        in_points = [0, 1]
        input_t = torch.zeros(64, input_size, len(in_points), len(in_dims))
        config = ModuleConfig(
            dataset_config=MotionDatasetConfig(
                future_size=output_size,
                history_size=input_size,
                in_dims=in_dims,
                in_points=in_points,
            ),
            used_norm=Normalizations.SPATIAL,
        )
        normalization_layer = MotionLayerNorm(config)
        output_t = normalization_layer(input_t)
        self.assertEqual(input_t.shape, output_t.shape)
        with self.assertRaises(ValidationError) as _:
            config = ModuleConfig(
                dataset_config=MotionDatasetConfig(
                    future_size=output_size,
                    history_size=input_size,
                    in_points=in_points,
                ),
                used_norm=Normalizations.SPATIAL,
            )

    def test_temporal_norm(self):
        """test test temporal_norm and forward"""
        input_size = 10
        output_size = 5
        in_dims = [0, 1, 2]
        in_points = [0, 1]
        input_t = torch.zeros(64, input_size, len(in_points), len(in_dims))
        config = ModuleConfig(
            dataset_config=MotionDatasetConfig(
                future_size=output_size,
                history_size=input_size,
            ),
            used_norm=Normalizations.TEMPORAL,
        )
        normalization_layer = MotionLayerNorm(config)
        output_t = normalization_layer(input_t)
        self.assertEqual(input_t.shape, output_t.shape)
        with self.assertRaises(ValidationError) as _:
            config = ModuleConfig(
                dataset_config=MotionDatasetConfig(
                    future_size=output_size,
                ),
                used_norm=Normalizations.TEMPORAL,
            )

    def test_batch_norm(self):
        """test batch norm init and forward"""
        input_size = 10
        output_size = 5
        in_dims = [0, 1, 2]
        in_points = [0, 1]
        input_t = torch.zeros(64, input_size, len(in_points), len(in_dims))
        config = ModuleConfig(
            dataset_config=MotionDatasetConfig(
                future_size=output_size,
                history_size=input_size,
            ),
            used_norm=Normalizations.BATCH,
        )
        normalization_layer = MotionLayerNorm(config)
        output_t = normalization_layer(input_t)
        self.assertEqual(input_t.shape, output_t.shape)
        with self.assertRaises(ValidationError) as _:
            config = ModuleConfig(
                dataset_config=MotionDatasetConfig(
                    future_size=output_size,
                ),
                used_norm=Normalizations.BATCH,
            )
