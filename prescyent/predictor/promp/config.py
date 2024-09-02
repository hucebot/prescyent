"""Config elements for Pytorch Lightning Modules usage"""
from prescyent.predictor.config import PredictorConfig


class PrompConfig(PredictorConfig):
    """Pydantic Basemodel for PrompPredictor configuration"""

    num_bf: float = 20
    """Lower = smoother"""
    ridge_factor: float = 1e-10
    """Regularization parameter of the ridge regression"""
