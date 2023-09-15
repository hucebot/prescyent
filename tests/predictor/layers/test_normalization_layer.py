import torch
from pydantic import ValidationError

from prescyent.predictor.lightning.layers.normalization_layer import MotionLayerNorm
from prescyent.predictor.lightning.configs.module_config import ModuleConfig
from prescyent.utils.enums import Normalizations
from tests.custom_test_case import CustomTestCase


class MotionLayerNormTests(CustomTestCase):
    def test_all_norm(self):
        """test all norm init and forward"""
        input_size = 10
        num_points = 5
        num_dims = 3
        input_t = torch.zeros(64, input_size, num_points, num_dims)
        config = ModuleConfig(input_size=input_size,
                              num_points=num_points,
                              num_dims=num_dims,
                              used_norm=Normalizations.ALL,
                              )
        normalization_layer = MotionLayerNorm(config)
        output_t = normalization_layer(input_t)
        self.assertEqual(input_t.shape, output_t.shape)
        with self.assertRaises(ValidationError) as context:
            config = ModuleConfig(input_size=input_size,
                              num_points=num_points,
                              used_norm=Normalizations.ALL,
                              )

    def test_spatial_norm(self):
        """test spatial norm init and forward"""
        input_size = 10
        num_points = 5
        num_dims = 3
        input_t = torch.zeros(64, input_size, num_points, num_dims)
        config = ModuleConfig(num_points=num_points,
                              num_dims=num_dims,
                              used_norm=Normalizations.SPATIAL,
                              )
        normalization_layer = MotionLayerNorm(config)
        output_t = normalization_layer(input_t)
        self.assertEqual(input_t.shape, output_t.shape)
        with self.assertRaises(ValidationError) as context:
            config = ModuleConfig(num_points=num_points,
                              used_norm=Normalizations.SPATIAL,
                              )

    def test_temporal_norm(self):
        """test test temporal_norm and forward"""
        input_size = 10
        num_points = 5
        num_dims = 3
        input_t = torch.zeros(64, input_size, num_points, num_dims)
        config = ModuleConfig(input_size=input_size,
                              used_norm=Normalizations.TEMPORAL,
                              )
        normalization_layer = MotionLayerNorm(config)
        output_t = normalization_layer(input_t)
        self.assertEqual(input_t.shape, output_t.shape)
        with self.assertRaises(ValidationError) as context:
            config = ModuleConfig(num_dims=num_dims,
                              num_points=num_points,
                              used_norm=Normalizations.TEMPORAL,
                              )

    def test_batch_norm(self):
        """test batch norm init and forward"""
        input_size = 10
        num_points = 5
        num_dims = 3
        input_t = torch.zeros(64, input_size, num_points, num_dims)
        config = ModuleConfig(input_size=input_size,
                              used_norm=Normalizations.BATCH,
                              )
        normalization_layer = MotionLayerNorm(config)
        output_t = normalization_layer(input_t)
        self.assertEqual(input_t.shape, output_t.shape)
        with self.assertRaises(ValidationError) as context:
            config = ModuleConfig(used_norm=Normalizations.BATCH)
