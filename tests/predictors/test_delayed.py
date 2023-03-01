import shutil
import torch
from torch.utils.data import DataLoader

from prescyent.predictor import DelayedPredictor
from tests.custom_test_case import CustomTestCase


class DelayedPredictorTests(CustomTestCase):

    def test_prediction(self):
        predictor = DelayedPredictor("tmp")
        input_tensor = torch.rand(20, 10, 7)
        output = predictor.get_prediction(input_tensor, len(input_tensor[0]))
        self.assertTrue(torch.equal(input_tensor, output))
        input_tensor = torch.rand(3, 2)
        output = predictor.get_prediction(input_tensor, len(input_tensor))
        self.assertTrue(torch.equal(input_tensor, output))
        output = predictor.run(input_tensor)[0]
        self.assertTrue(torch.equal(input_tensor, output))
        output = predictor(input_tensor)[0]
        self.assertTrue(torch.equal(input_tensor, output))
        shutil.rmtree("tmp", ignore_errors=True)

    def test_test_loop(self):
        predictor = DelayedPredictor("tmp")
        dataloader = DataLoader([(torch.rand(64, 10, 7), torch.rand(64, 10, 7)) for i in range(50)])
        test_results = predictor.test(dataloader)
        shutil.rmtree("tmp", ignore_errors=True)
