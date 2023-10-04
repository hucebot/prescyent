import torch

from prescyent.predictor.lightning.layers.reshaping_layer import ReshapingLayer

from tests.custom_test_case import CustomTestCase


class ReshapingLayerTests(CustomTestCase):
    def test_trajectory_shapes(self):
        """test with trajectory likes shapes"""
        input_t = torch.zeros(64, 10, 10, 3)
        output_t = torch.zeros(64, 10, 3, 3)
        model = ReshapingLayer(input_shapes=input_t.shape, output_shapes=output_t.shape)
        self.assertEqual(model(input_t).shape, output_t.shape)
        model = ReshapingLayer(input_shapes=output_t.shape, output_shapes=input_t.shape)
        self.assertEqual(model(output_t).shape, input_t.shape)
        input_t = torch.zeros(64, 10, 10, 3)
        output_t = torch.zeros(128, 25, 3, 3)
        model = ReshapingLayer(input_shapes=input_t.shape, output_shapes=output_t.shape)
        self.assertEqual(model(input_t).shape, output_t.shape)
        model = ReshapingLayer(input_shapes=output_t.shape, output_shapes=input_t.shape)
        self.assertEqual(model(output_t).shape, input_t.shape)
        input_t = torch.zeros(1, 2, 3, 4)
        output_t = torch.zeros(4, 3, 2, 1)
        model = ReshapingLayer(input_shapes=input_t.shape, output_shapes=output_t.shape)
        self.assertEqual(model(input_t).shape, output_t.shape)
        model = ReshapingLayer(input_shapes=output_t.shape, output_shapes=input_t.shape)
        self.assertEqual(model(output_t).shape, input_t.shape)

    def test_no_shapes(self):
        """empty reshapelayer returns input tensor has output"""
        model = ReshapingLayer([], [])
        input_t = torch.zeros(64, 10, 10, 3)
        self.assertTrue(torch.equal(model(input_t), input_t))

    def test_same_shapes(self):
        """same shape reshapelayer returns input tensor has output"""
        input_t = torch.zeros(64, 10, 10, 3)
        output_t = torch.zeros(64, 10, 10, 3)
        model = ReshapingLayer(input_shapes=input_t.shape, output_shapes=output_t.shape)
        self.assertTrue(torch.equal(model(input_t), input_t))

    def test_shape_dims_mismatch(self):
        """same shape reshapelayer returns input tensor has output"""
        input_t = torch.zeros(64, 10, 10)
        output_t = torch.zeros(64, 10, 10, 3)
        model = ReshapingLayer(input_shapes=input_t.shape, output_shapes=output_t.shape)
        self.assertTrue(torch.equal(model(input_t), input_t))

    def test_extreme_dims(self):
        """same shape reshapelayer returns input tensor has output"""
        input_t = torch.zeros(2)
        output_t = torch.zeros(4)
        model = ReshapingLayer(input_shapes=input_t.shape, output_shapes=output_t.shape)
        self.assertEqual(model(input_t).shape, output_t.shape)
        input_t = torch.zeros([1] + [i for i in range(15)] + [1])
        output_t = torch.zeros([2] + [i for i in range(15)] + [2])
        model = ReshapingLayer(input_shapes=input_t.shape, output_shapes=output_t.shape)
        self.assertEqual(model(input_t).shape, output_t.shape)
