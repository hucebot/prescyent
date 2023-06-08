"""Classes to handles custom exceptions"""
from prescyent.utils.errors.custom_exception import CustomException


ERROR_MSG = "Predictor could not be loaded"
ERROR_CODE = 400


class PredictorUnprocessable(CustomException):
    """Raised when a predictor file could not be loaded"""

    def __init__(self, code=ERROR_CODE, message=ERROR_MSG):
        super().__init__(code, message)
