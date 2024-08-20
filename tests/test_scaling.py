import torch
from torch.utils.data import DataLoader

from prescyent.dataset import TeleopIcubDataset, TeleopIcubDatasetConfig
from prescyent.dataset.features import CoordinateXY
from prescyent.predictor import PredictorConfig
from prescyent.scaler import Scaler, ScalerConfig
from prescyent.utils.enums import TrajectoryDimensions

from tests.custom_test_case import CustomTestCase


class InitTeleopIcubDatasetTest(CustomTestCase):
    def test_complex_scaling(self):
        frequency: int = 10     # subsampling -> 100 Hz to 10Hz
        history_size = 5       # 0.5 seconds
        future_size = 5        # 0.5 seconds
        batch_size = 256
        dataset_config = TeleopIcubDatasetConfig(
            history_size=history_size,
            future_size=future_size,
            frequency=frequency,
            batch_size=batch_size,
            in_points=[1, 2],
            out_points=[2],
            out_features=[CoordinateXY(range(2))]
        )
        dataset = TeleopIcubDataset(dataset_config)
        for dim in [
            TrajectoryDimensions.FEATURE,
            TrajectoryDimensions.POINT,
            TrajectoryDimensions.TEMPORAL,
            TrajectoryDimensions.SPATIAL,
                                ]:
            scaler_config=ScalerConfig(do_feature_wise_scaling=True, scaling_axis=dim)
            pred_config = PredictorConfig(dataset_config=dataset_config,
                                        scaler_config=scaler_config)
            # instanciate scaler
            scaler = Scaler(scaler_config)
            # collate all frames for train trajectories and train on theses
            scaler.train(DataLoader(torch.cat([traj.tensor for traj in dataset.trajectories.train], dim=0),
                                    batch_size=dataset.config.batch_size,
                                    # collate_fn=lambda batch: batch, #return a list instead of a tensor
                                    ),
                        dataset_features=dataset.tensor_features)
            test_t = torch.cat([traj.tensor for traj in dataset.trajectories.train])
            n_test_t = scaler.scale(test_t, features=dataset.tensor_features)
            std = n_test_t.unsqueeze(0).std(dim=scaler.scalers['Coordinate'].dim)
            mean = n_test_t.unsqueeze(0).mean(dim=scaler.scalers['Coordinate'].dim)
            std[std==0] = 1
            self.assertTrue(torch.allclose(
                std,
                torch.ones_like(scaler.scalers['Coordinate'].std),
                atol=1e-4,
            ))
            self.assertTrue(torch.allclose(
                mean,
                torch.zeros_like(scaler.scalers['Coordinate'].mean),
                atol=1e-4,
            ))
            u_test_t = scaler.unscale(n_test_t, features=dataset.tensor_features)
            self.assertTrue(torch.allclose(test_t, u_test_t, atol=1e-4))

            from prescyent.predictor import DelayedPredictor

            delayed = DelayedPredictor(pred_config)
            delayed.train_scaler(dataset)
            test_sample, _ = next(iter(dataset.test_dataloader()))
            output = delayed.predict(test_sample, future_size)
            self.assertTrue(torch.allclose(test_sample[:,-future_size:,[1],:2], output, atol=1e-4))
