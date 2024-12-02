from pydantic import ValidationError

from prescyent.dataset import TrajectoriesDatasetConfig
from prescyent.dataset.features import Any, Features
from prescyent.predictor import MlpPredictor, MlpConfig
from prescyent.predictor.lightning.predictor import MODEL_CHECKPOINT_NAME


from tests.custom_test_case import CustomTestCase

features = Features([Any(range(1))])
dataset_config = TrajectoriesDatasetConfig(
    frequency=10,
    history_size=10,
    future_size=10,
    in_points=[0],
    out_points=[0],
    in_features=features,
    out_features=features,
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
            config.in_points_dims * config.in_sequence_size,
        )  # mandatory value
        self.assertEqual(
            predictor.model.torch_model.layers[-1].out_features,
            config.out_points_dims * config.out_sequence_size,
        )  # mandatory value
        self.assertEqual(
            len(predictor.model.torch_model.layers), config.num_layers * 2 - 1
        )

    def test_missing_config_arg_error(self):
        with self.assertRaises(ValidationError):
            MlpConfig(feature_size=1)

    def test_missing_config_error(self):
        self.assertRaises(TypeError, MlpPredictor)

    # -- INIT FROM STATE
    def test_init_with_pathname(self):
        MlpPredictor.load_pretrained("tests/mocking/mlp_model")
        MlpPredictor.load_pretrained(f"tests/mocking/mlp_model/{MODEL_CHECKPOINT_NAME}")
        with self.assertRaises(FileNotFoundError) as context:
            MlpPredictor.load_pretrained("tests/mocking/non_existing/")
        self.assertTrue("No file or directory" in str(context.exception))

    def test_init_with_config_containing_path(self):
        model_path = "tests/mocking/mlp_model"
        config = MlpConfig(
            dataset_config=dataset_config,
            save_path=model_path,
        )
        MlpPredictor(config=config)
