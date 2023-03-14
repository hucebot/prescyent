import unittest
import torch


from prescyent.utils.tensor_manipulation import cat_tensor_with_seq_idx


class TestFlattenListPreds(unittest.TestCase):

    def test_return_as_is(self):
        input_pred = torch.arange(3 * 3).view(3, 3)
        output_pred = cat_tensor_with_seq_idx(input_pred)
        self.assertTrue(torch.equal(input_pred, output_pred))

    def test_flatten(self):
        input_pred = torch.arange(3 * 3 * 3* 3).view(3, 3, 3, 3)
        expected_output = torch.Tensor([[[18., 19., 20.],
         [21., 22., 23.],
         [24., 25., 26.]],

        [[45., 46., 47.],
         [48., 49., 50.],
         [51., 52., 53.]],

        [[72., 73., 74.],
         [75., 76., 77.],
         [78., 79., 80.]]])
        output_pred = cat_tensor_with_seq_idx(input_pred)
        self.assertTrue(torch.equal(expected_output, output_pred))
