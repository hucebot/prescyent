"""Classes to handles custom exceptions"""
from prescyent.utils.errors.custom_exception import CustomException


ERROR_MSG = "The provided dataset is empty. We require a non empty iterable"
ERROR_CODE = 400


class DatasetEmptyException(CustomException):
    """Raised when a loaded or given dataset is empty"""

    def __init__(self, code=ERROR_CODE, message=ERROR_MSG):
        super().__init__(code, message)
