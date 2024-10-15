import shutil
import unittest

import torch
from torch.utils.data import DataLoader

from prescyent.dataset import SCCDataset, SCCDatasetConfig
from prescyent.dataset.features import CoordinateXY, Features, RotationEuler
from prescyent.dataset.features.feature.rotation import Rotation
from prescyent.predictor import (
    DelayedPredictor,
    PredictorConfig,
    MlpPredictor,
    MlpConfig,
    TrainingConfig,
)
from prescyent.scaler import Scaler, ScalerConfig
from prescyent.utils.enums import TrajectoryDimensions, Scalers

from tests.custom_test_case import CustomTestCase


class ScalerNormalizationTest(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        """setup dataset for the scaling tests"""
        dataset_config = SCCDatasetConfig(num_trajs=[5, 5])
        cls.dataset = SCCDataset(dataset_config)

    @classmethod
    def tearDownClass(cls):
        """setup dataset for the scaling tests"""
        shutil.rmtree("tmp", ignore_errors=True)

    def verify_normalization(self, scaler):
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
            min_t = n_test_t[..., feat_ids].amin(dim=scaler.scalers[feat_name].dim)
            max_t = n_test_t[..., feat_ids].amax(dim=scaler.scalers[feat_name].dim)
            self.assertTrue(
                torch.allclose(
                    max_t,
                    torch.ones_like(scaler.scalers[feat_name].max_t),
                    atol=1e-4,
                )
            )
            self.assertTrue(
                torch.allclose(
                    min_t,
                    torch.zeros_like(scaler.scalers[feat_name].min_t),
                    atol=1e-4,
                )
            )
            u_test_t = scaler.unscale(n_test_t, features=self.dataset.tensor_features)
            self.assertTrue(torch.allclose(test_t, u_test_t, atol=1e-4))

    def test_normalizer_all_scaling_axes(self):
        for dim in [
            TrajectoryDimensions.SPATIAL,
            TrajectoryDimensions.TEMPORAL,
            TrajectoryDimensions.POINT,
            TrajectoryDimensions.FEATURE,
        ]:
            print(dim)
            scaler_config = ScalerConfig(
                scaler=Scalers.NORMALIZATION,
                do_feature_wise_scaling=True,
                scaling_axis=dim,
            )
            scaler = Scaler(scaler_config)
            # collate all frames for train trajectories and train on theses
            dataset_tensor = torch.cat(
                [traj.tensor for traj in self.dataset.trajectories.train], dim=0
            )
            scaler.train(
                DataLoader(
                    dataset_tensor,
                    batch_size=self.dataset.config.batch_size,
                    num_workers=self.dataset.config.num_workers,
                ),
                dataset_features=self.dataset.tensor_features,
            )
            self.verify_normalization(scaler)

    def test_scaler_in_predictor(self):
        scaler_config = ScalerConfig(
            scaler=Scalers.NORMALIZATION,
            do_feature_wise_scaling=True,
            scaling_axis=TrajectoryDimensions.SPATIAL,
        )
        mlp_config = MlpConfig(
            dataset_config=self.dataset.config,
            scaler_config=scaler_config,
            save_path="tmp",
        )
        mlp = MlpPredictor(config=mlp_config)
        training_config = TrainingConfig(max_epochs=1)
        _ = torch.cat([traj.tensor for traj in self.dataset.trajectories.train], dim=0)
        mlp.train(self.dataset, training_config)
        self.verify_normalization(mlp.scaler)

    def test_load_and_save(self):
        # instanciate scaler
        scaler_config = ScalerConfig(
            scaler=Scalers.NORMALIZATION,
            do_feature_wise_scaling=True,
            scaling_axis=TrajectoryDimensions.SPATIAL,
        )
        scaler = Scaler(scaler_config)
        scaler.save("tmp/scaler.pkl")
        scaler_2 = Scaler.load("tmp/scaler.pkl")
        # train scaler
        scaler.train(
            DataLoader(
                torch.cat(
                    [traj.tensor for traj in self.dataset.trajectories.train], dim=0
                ),
                batch_size=self.dataset.config.batch_size,
            ),
            dataset_features=self.dataset.tensor_features,
        )
        scaler.save("tmp/scaler_trained.pkl")
        scaler_2 = Scaler.load("tmp/scaler_trained.pkl")
        self.verify_normalization(scaler_2)
        self.verify_normalization(scaler)

    def test_load_and_save_predictor_with_scaler(self):
        # instanciate from loaded predictor
        scaler_config = ScalerConfig(
            scaler=Scalers.NORMALIZATION,
            do_feature_wise_scaling=True,
            scaling_axis=TrajectoryDimensions.SPATIAL,
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
        self.verify_normalization(mlp_2.scaler)
