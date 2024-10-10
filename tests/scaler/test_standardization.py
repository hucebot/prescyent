import shutil

import torch
from torch.utils.data import DataLoader

from prescyent.dataset import SCCDataset, SCCDatasetConfig
from prescyent.dataset.features import Features, CoordinateXY, Rotation, RotationEuler
from prescyent.predictor import (
    MlpPredictor,
    MlpConfig,
    TrainingConfig,
)
from prescyent.scaler import Scaler, ScalerConfig
from prescyent.utils.enums import TrajectoryDimensions

from tests.custom_test_case import CustomTestCase


class ScalerStandardizationTest(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        """setup dataset for the scaling tests"""
        dataset_config = SCCDatasetConfig(num_trajs=[5, 5])
        cls.dataset = SCCDataset(dataset_config)

    @classmethod
    def tearDownClass(cls):
        """setup dataset for the scaling tests"""
        shutil.rmtree("tmp", ignore_errors=True)

    def verify_standardization(self, scaler):
        test_t = torch.cat([traj.tensor for traj in self.dataset.trajectories.train])
        n_test_t = scaler.scale(
            test_t.unsqueeze(0), features=self.dataset.tensor_features
        )
        if scaler.config.do_feature_wise_scaling:
            feat_names = {}
            for feat in self.dataset.tensor_features:
                if not isinstance(feat, Rotation) or scaler.config.scale_rotations:
                    feat_names[feat.name] = feat.ids
        else:
            feat_names = {"feature": list[range(test_t.shape[-1])]}
        for feat_name, feat_ids in feat_names.items():
            std = n_test_t[..., feat_ids].std(dim=scaler.scalers[feat_name].dim)
            mean = n_test_t[..., feat_ids].mean(dim=scaler.scalers[feat_name].dim)
            std[std == 0] = 1
            self.assertTrue(
                torch.allclose(
                    std,
                    torch.ones_like(scaler.scalers[feat_name].std),
                    atol=1e-3,
                )
            )
            self.assertTrue(
                torch.allclose(
                    mean,
                    torch.zeros_like(scaler.scalers[feat_name].mean),
                    atol=1e-3,
                )
            )
            u_test_t = scaler.unscale(n_test_t, features=self.dataset.tensor_features)
            self.assertTrue(torch.allclose(test_t, u_test_t, atol=1e-3))

    def test_standardizer_all_scaling_axes(self):
        for dim in [
            TrajectoryDimensions.POINT,
            TrajectoryDimensions.FEATURE,
            TrajectoryDimensions.TEMPORAL,
            TrajectoryDimensions.SPATIAL,
        ]:
            scaler_config = ScalerConfig(
                do_feature_wise_scaling=True, scaling_axis=dim, scale_rotations=True
            )
            scaler = Scaler(scaler_config)
            # collate all frames for train trajectories and train on theses
            dataset_tensor = torch.cat(
                [traj.tensor for traj in self.dataset.trajectories.train], dim=0
            )
            scaler.train(
                DataLoader(dataset_tensor, batch_size=self.dataset.config.batch_size),
                dataset_features=self.dataset.tensor_features,
            )
            self.verify_standardization(scaler)

    def test_scaler_in_predictor(self):
        scaler_config = ScalerConfig(
            do_feature_wise_scaling=True, scaling_axis=TrajectoryDimensions.SPATIAL
        )
        mlp_config = MlpConfig(
            dataset_config=self.dataset.config,
            scaler_config=scaler_config,
            save_path="tmp",
        )
        mlp = MlpPredictor(config=mlp_config)
        training_config = TrainingConfig(max_epochs=1)
        mlp.train(self.dataset, training_config)
        self.verify_standardization(mlp.scaler)

    def test_load_and_save(self):
        # instanciate scaler
        scaler_config = ScalerConfig(
            do_feature_wise_scaling=True, scaling_axis=TrajectoryDimensions.SPATIAL
        )
        scaler = Scaler(scaler_config)
        scaler.save("tmp/scaler.pkl")
        scaler_2 = Scaler.load("tmp/scaler.pkl")
        # train scaler
        dataset_tensor = torch.cat(
            [traj.tensor for traj in self.dataset.trajectories.train], dim=0
        )
        scaler.train(
            DataLoader(dataset_tensor, batch_size=self.dataset.config.batch_size),
            dataset_features=self.dataset.tensor_features,
        )
        scaler.save("tmp/scaler_trained.pkl")
        scaler_2 = Scaler.load("tmp/scaler_trained.pkl")
        self.verify_standardization(scaler_2)
        self.verify_standardization(scaler)

    def test_load_and_save_predictor_with_scaler(self):
        # instanciate from loaded predictor
        scaler_config = ScalerConfig(
            do_feature_wise_scaling=True, scaling_axis=TrajectoryDimensions.SPATIAL
        )
        mlp_config = MlpConfig(
            dataset_config=self.dataset.config,
            scaler_config=scaler_config,
            save_path="tmp",
        )
        mlp = MlpPredictor(config=mlp_config)
        mlp.save("tmp/tmp")
        training_config = TrainingConfig(max_epochs=1)
        mlp.train(self.dataset, training_config)
        mlp.save("tmp/tmp2")
        mlp_2 = MlpPredictor.load_pretrained("tmp/tmp2")
        self.verify_standardization(mlp_2.scaler)
