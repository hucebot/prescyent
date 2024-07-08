import shutil
import torch

from prescyent.dataset import (
    DatasetConfig,
    CustomDataset,
    Trajectories,
    Trajectory,
)
from prescyent.dataset.features import Any
from prescyent.predictor import DelayedPredictor
from tests.custom_test_case import CustomTestCase


class DelayedPredictorTests(CustomTestCase):
    def test_prediction(self):
        features = [Any(range(6))]
        points = list(range(7))
        dataset_config = DatasetConfig(
            batch_size=20,
            frequency=10,
            history_size=10,
            future_size=10,
            in_features=features,
            out_features=features,
            in_points=points,
            out_points=points,
        )
        predictor = DelayedPredictor(dataset_config, "tmp")
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
        features = [Any(range(7))]
        points = list(range(9))
        dataset_config = DatasetConfig(
            batch_size=64,
            frequency=10,
            history_size=10,
            future_size=10,
            in_features=features,
            out_features=features,
            in_points=points,
            out_points=points,
        )
        predictor = DelayedPredictor(dataset_config, "tmp")
        trajs = Trajectories(
            [(Trajectory(torch.rand(500, 9, 7), 10, features)) for i in range(1)],
            [(Trajectory(torch.rand(500, 9, 7), 10, features)) for i in range(10)],
            [(Trajectory(torch.rand(500, 9, 7), 10, features)) for i in range(1)],
        )
        dataset = CustomDataset(dataset_config, trajs, load_data_at_init=True)
        predictor.test(dataset)
        shutil.rmtree("tmp", ignore_errors=True)
