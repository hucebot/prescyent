"""Classes to handles custom exceptions"""
from prescyent.utils.errors.custom_exception import CustomException


ERROR_MSG = "Predictor file not found"
ERROR_CODE = 400


class PredictorNotFound(CustomException):
    """Raised when a predictor file is not found"""

    def __init__(self, code=ERROR_CODE, message=ERROR_MSG):
        super().__init__(code, message)
