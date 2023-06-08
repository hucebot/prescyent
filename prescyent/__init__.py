"""Description of the lib"""


from prescyent.auto_predictor import AutoPredictor
from prescyent.auto_dataset import AutoDataset
from prescyent.predictor.constant_predictor import BasePredictor
from prescyent.predictor.constant_predictor import ConstantPredictor


__version__ = "0.1.0"


def get_predictor_from_path(predictor_path: str) -> BasePredictor:
    if predictor_path:
        return AutoPredictor.load_from_config(predictor_path)
    else:
        return ConstantPredictor("ConstantPredictor")
