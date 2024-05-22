import torch
from pydantic import ValidationError

from prescyent.dataset.dataset import MotionDatasetConfig
from prescyent.dataset.features import Any
from prescyent.predictor.lightning.layers.normalization_layer import MotionLayerNorm
from prescyent.predictor.lightning.configs.module_config import ModuleConfig
from prescyent.utils.enums import Normalizations
from tests.custom_test_case import CustomTestCase


class MotionLayerNormTests(CustomTestCase):
    def test_all_norm(self):
        """test all norm init and forward"""
        in_sequence_size = 10
        out_sequence_size = 5
        in_features = [Any(range(3))]
        in_points = [0, 1]
        input_t = torch.zeros(
            64,
            in_sequence_size,
            len(in_points),
            sum([len(feat.ids) for feat in in_features]),
        )
        config = ModuleConfig(
            dataset_config=MotionDatasetConfig(
                future_size=out_sequence_size,
                history_size=in_sequence_size,
                in_features=in_features,
                out_features=in_features,
                in_points=in_points,
                out_points=in_points,
            ),
            used_norm=Normalizations.ALL,
        )
        normalization_layer = MotionLayerNorm(config)
        output_t = normalization_layer(input_t)
        self.assertEqual(input_t.shape, output_t.shape)

    def test_spatial_norm(self):
        """test spatial norm init and forward"""
        in_sequence_size = 10
        out_sequence_size = 5
        in_features = [Any([0, 1, 2])]
        in_points = [0, 1]
        input_t = torch.zeros(
            64,
            in_sequence_size,
            len(in_points),
            sum([len(feat.ids) for feat in in_features]),
        )
        config = ModuleConfig(
            dataset_config=MotionDatasetConfig(
                future_size=out_sequence_size,
                history_size=in_sequence_size,
                in_features=in_features,
                out_features=in_features,
                in_points=in_points,
                out_points=in_points,
            ),
            used_norm=Normalizations.SPATIAL,
        )
        normalization_layer = MotionLayerNorm(config)
        output_t = normalization_layer(input_t)
        self.assertEqual(input_t.shape, output_t.shape)

    def test_temporal_norm(self):
        """test test temporal_norm and forward"""
        in_sequence_size = 10
        out_sequence_size = 5
        in_features = [Any([0, 1, 2])]
        in_points = [0, 1]
        input_t = torch.zeros(
            64,
            in_sequence_size,
            len(in_points),
            sum([len(feat.ids) for feat in in_features]),
        )
        config = ModuleConfig(
            dataset_config=MotionDatasetConfig(
                in_features=in_features,
                out_features=in_features,
                future_size=out_sequence_size,
                history_size=in_sequence_size,
                in_points=in_points,
                out_points=in_points,
            ),
            used_norm=Normalizations.TEMPORAL,
        )
        normalization_layer = MotionLayerNorm(config)
        output_t = normalization_layer(input_t)
        self.assertEqual(input_t.shape, output_t.shape)
        with self.assertRaises(ValidationError) as _:
            config = ModuleConfig(
                dataset_config=MotionDatasetConfig(
                    future_size=out_sequence_size,
                ),
                used_norm=Normalizations.TEMPORAL,
            )

    def test_batch_norm(self):
        """test batch norm init and forward"""
        in_sequence_size = 10
        out_sequence_size = 5
        in_features = [Any([0, 1, 2])]
        in_points = [0, 1]
        input_t = torch.zeros(
            64,
            in_sequence_size,
            len(in_points),
            sum([len(feat.ids) for feat in in_features]),
        )
        config = ModuleConfig(
            dataset_config=MotionDatasetConfig(
                future_size=out_sequence_size,
                history_size=in_sequence_size,
                in_features=in_features,
                out_features=in_features,
                in_points=in_points,
                out_points=in_points,
            ),
            used_norm=Normalizations.BATCH,
        )
        normalization_layer = MotionLayerNorm(config)
        output_t = normalization_layer(input_t)
        self.assertEqual(input_t.shape, output_t.shape)
        with self.assertRaises(ValidationError) as _:
            config = ModuleConfig(
                dataset_config=MotionDatasetConfig(
                    future_size=out_sequence_size,
                ),
                used_norm=Normalizations.BATCH,
            )
