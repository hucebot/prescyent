
from pydantic import ValidationError
from prescyent.predictor import LinearPredictor, LinearConfig

from tests.custom_test_case import CustomTestCase


class LinearInitTests(CustomTestCase):

    # -- INIT FROM SCRATCH AND CONFIG
    def test_init(self):
        feature_size = 1
        output_size = 10
        input_size = 10
        config = LinearConfig(feature_size=feature_size,
                            output_size=output_size,
                            input_size=input_size
                            )
        predictor = LinearPredictor(config=config)
        self.assertHasAttr(predictor, "model")
        self.assertHasAttr(predictor, "config")
        self.assertHasAttr(predictor.model, "torch_model")
        self.assertHasAttr(predictor.model, "criterion")
        self.assertEqual(predictor.model.torch_model.feature_size,
                         feature_size)            # mandatory value
        self.assertEqual(predictor.model.torch_model.linear.in_features,
                         input_size)           # mandatory value
        self.assertEqual(predictor.model.torch_model.output_size,
                         output_size)           # mandatory value

    def test_missing_config_arg_error(self):
        with self.assertRaises(ValidationError):
            LinearConfig(feature_size=1)

    def test_missing_config_error(self):
        self.assertRaises(NotImplementedError, LinearPredictor)

    # -- INIT FROM STATE
    def test_init_with_pathname(self):
        LinearPredictor(model_path="tests/mocking/linear_model")
        LinearPredictor(model_path="tests/mocking/linear_model/trainer_checkpoint.ckpt")
        LinearPredictor(model_path="tests/mocking/linear_model/model.pb")
        with self.assertRaises(NotImplementedError) as context:
            LinearPredictor(model_path="tests/mocking/linear_model/bad_model.bin")
        self.assertTrue("Given file extention .bin is not supported" in str(context.exception))
        with self.assertRaises(FileNotFoundError) as context:
            LinearPredictor(model_path="tests/mocking/linear_model/non_existing.bin")
        self.assertTrue("No file or directory" in str(context.exception))

    def test_init_with_config_containing_path(self):
        feature_size = 1
        output_size = 10
        input_size = 100
        model_path = "tests/mocking/linear_model"
        config = LinearConfig(feature_size=feature_size,
                            output_size=output_size,
                            input_size=input_size,
                            model_path=model_path
                            )
        LinearPredictor(config=config)


class LinearFunctionalTests(CustomTestCase):

    def setUp(self):
        feature_size = 1
        output_size = 10
        config = LinearConfig(feature_size=feature_size,
                            output_size=output_size,
                            )
        self.predictor = LinearPredictor(config=config)

    # TODO More functionnal tests
