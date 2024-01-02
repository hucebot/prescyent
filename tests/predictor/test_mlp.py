from pydantic import ValidationError
from prescyent.predictor import MlpPredictor, MlpConfig
from prescyent.dataset import DatasetConfig

from tests.custom_test_case import CustomTestCase


dataset_config = DatasetConfig(
    history_size=10,
    future_size=10,
    in_points=[0],
    out_points=[0],
    in_dims=[0],
    out_dims=[0],
)


class MlpInitTests(CustomTestCase):
    # -- INIT FROM SCRATCH AND CONFIG
    def test_init(self):
        config = MlpConfig(dataset_config=dataset_config, num_layers=10)
        predictor = MlpPredictor(config=config)
        self.assertHasAttr(predictor, "model")
        self.assertHasAttr(predictor, "config")
        self.assertHasAttr(predictor.model, "torch_model")
        self.assertHasAttr(predictor.model, "criterion")
        self.assertEqual(
            predictor.model.torch_model.layers[0].in_features,
            config.in_feature_size * config.input_size,
        )  # mandatory value
        self.assertEqual(
            predictor.model.torch_model.layers[-1].out_features,
            config.out_feature_size * config.output_size,
        )  # mandatory value
        self.assertEqual(
            len(predictor.model.torch_model.layers), config.num_layers * 2 - 1
        )

    def test_missing_config_arg_error(self):
        with self.assertRaises(ValidationError):
            MlpConfig(feature_size=1)

    def test_missing_config_error(self):
        self.assertRaises(NotImplementedError, MlpPredictor)

    # -- INIT FROM STATE
    def test_init_with_pathname(self):
        MlpPredictor(model_path="tests/mocking/mlp_model")
        MlpPredictor(model_path="tests/mocking/mlp_model/trainer_checkpoint.ckpt")
        with self.assertRaises(NotImplementedError) as context:
            MlpPredictor(model_path="tests/mocking/mlp_model/bad_model.bin")
        self.assertTrue(
            "Given file extention .bin is not supported" in str(context.exception)
        )
        with self.assertRaises(FileNotFoundError) as context:
            MlpPredictor(model_path="tests/mocking/mlp_model/non_existing.bin")
        self.assertTrue("No file or directory" in str(context.exception))

    def test_init_with_config_containing_path(self):
        model_path = "tests/mocking/mlp_model"
        config = MlpConfig(
            dataset_config=dataset_config,
            model_path=model_path,
        )
        MlpPredictor(config=config)


class MlpFunctionalTests(CustomTestCase):
    def setUp(self):
        feature_size = 1
        output_size = 10
        config = MlpConfig(
            feature_size=feature_size,
            output_size=output_size,
        )
        self.predictor = MlpPredictor(config=config)
