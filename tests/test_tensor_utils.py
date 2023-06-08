import unittest
import torch


from prescyent.utils.tensor_manipulation import cat_tensor_with_seq_idx


class TestFlattenListPreds(unittest.TestCase):
    def test_return_as_is(self):
        input_pred = torch.arange(3 * 3).view(3, 3)
        output_pred = cat_tensor_with_seq_idx(input_pred)
        self.assertTrue(torch.equal(input_pred, output_pred))

    def test_flatten(self):
        input_pred = torch.arange(3 * 3 * 3 * 3).view(3, 3, 3, 3)
        expected_output = torch.Tensor(
            [
                [[18.0, 19.0, 20.0], [21.0, 22.0, 23.0], [24.0, 25.0, 26.0]],
                [[45.0, 46.0, 47.0], [48.0, 49.0, 50.0], [51.0, 52.0, 53.0]],
                [[72.0, 73.0, 74.0], [75.0, 76.0, 77.0], [78.0, 79.0, 80.0]],
            ]
        )
        output_pred = cat_tensor_with_seq_idx(input_pred)
        self.assertTrue(torch.equal(expected_output, output_pred))
