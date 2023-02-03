
from pydantic import ValidationError
from prescyent.predictor import LSTMPredictor, LSTMConfig

from tests.custom_test_case import CustomTestCase


class LSTMInitTests(CustomTestCase):

    # -- INIT FROM SCRATCH AND CONFIG
    def test_init(self):
        feature_size = 1
        output_size = 10
        hidden_size = 100
        config = LSTMConfig(feature_size=feature_size,
                            output_size=output_size,
                            hidden_size=hidden_size
                            )
        predictor = LSTMPredictor(config=config)
        self.assertHasAttr(predictor, "model")
        self.assertHasAttr(predictor, "config")
        self.assertHasAttr(predictor.model, "torch_model")
        self.assertHasAttr(predictor.model, "criterion")
        # check the model's parameters were init as expected
        self.assertEqual(predictor.model.torch_model.lstm.hidden_size,
                         hidden_size)           # overriden value
        self.assertEqual(predictor.model.torch_model.feature_size,
                         feature_size)            # mandatory value
        self.assertEqual(predictor.model.torch_model.lstm.input_size,
                         feature_size)            # mandatory value
        self.assertEqual(predictor.model.torch_model.linear.out_features,
                         feature_size)           # mandatory value
        self.assertEqual(predictor.model.torch_model.output_size,
                         output_size)           # mandatory value
        self.assertEqual(predictor.model.torch_model.lstm.num_layers,
                         config.num_layers)     # default value

    def test_missing_config_arg_error(self):
        with self.assertRaises(ValidationError):
            LSTMConfig(feature_size=1)

    def test_missing_config_error(self):
        self.assertRaises(NotImplementedError, LSTMPredictor)

    # -- INIT FROM STATE
    def test_init_with_pathname(self):
        LSTMPredictor(model_path="tests/mocking/lstm_model/lstm_baseline_ver1")
        LSTMPredictor(model_path="tests/mocking/lstm_model/lstm_baseline_ver1/trainer_checkpoint.ckpt")
        LSTMPredictor(model_path="tests/mocking/lstm_model/lstm_baseline_ver1/model.pb")
        with self.assertRaises(NotImplementedError) as context:
            LSTMPredictor(model_path="tests/mocking/lstm_model/lstm_baseline_ver1/bad_model.bin")
        self.assertTrue("Given file extention .bin is not supported" in str(context.exception))
        with self.assertRaises(FileNotFoundError) as context:
            LSTMPredictor(model_path="tests/mocking/lstm_model/lstm_baseline_ver1/non_existing.bin")
        self.assertTrue("No file or directory" in str(context.exception))

    def test_init_with_config_containing_path(self):
        feature_size = 1
        output_size = 10
        hidden_size = 100
        model_path = "tests/mocking/lstm_model/lstm_baseline_ver1"
        config = LSTMConfig(feature_size=feature_size,
                            output_size=output_size,
                            hidden_size=hidden_size,
                            model_path=model_path
                            )
        LSTMPredictor(config=config)


class LSTMFunctionalTests(CustomTestCase):

    def setUp(self):
        feature_size = 1
        output_size = 10
        config = LSTMConfig(feature_size=feature_size,
                            output_size=output_size,
                            )
        self.predictor = LSTMPredictor(config=config)

    # TODO
    # def test_train(self):
    #     self.predictor.train()
