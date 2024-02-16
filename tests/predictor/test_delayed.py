import shutil
import torch

from prescyent.dataset import (
    DatasetConfig,
    CustomDataset,
    Trajectories,
    Trajectory,
)
from prescyent.predictor import DelayedPredictor
from tests.custom_test_case import CustomTestCase


class DelayedPredictorTests(CustomTestCase):
    def test_prediction(self):
        predictor = DelayedPredictor("tmp")
        input_tensor = torch.rand(20, 10, 7, 6)
        output = predictor.predict(input_tensor, len(input_tensor[0]))
        self.assertTrue(torch.equal(input_tensor, output))
        input_tensor = torch.rand(10, 7, 6)
        output = predictor.predict(input_tensor, len(input_tensor))
        self.assertTrue(torch.equal(input_tensor, output))
        output = predictor.run(input_tensor)[0]
        self.assertTrue(torch.equal(input_tensor, output))
        output = predictor(input_tensor)[0]
        self.assertTrue(torch.equal(input_tensor, output))
        shutil.rmtree("tmp", ignore_errors=True)

    def test_test_loop(self):
        predictor = DelayedPredictor("tmp")
        trajs = Trajectories(
            [(Trajectory(torch.rand(500, 9, 7), 10)) for i in range(1)],
            [(Trajectory(torch.rand(500, 9, 7), 10)) for i in range(10)],
            [(Trajectory(torch.rand(500, 9, 7), 10)) for i in range(1)],
        )
        dataset = CustomDataset(DatasetConfig(history_size=10, future_size=10), trajs)
        predictor.test(dataset)
        shutil.rmtree("tmp", ignore_errors=True)
