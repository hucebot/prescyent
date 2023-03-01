"""Classes to handles custom exceptions"""
from prescyent.utils.errors.custom_exception import CustomException


ERROR_MSG = "The provided dataset is empty. We require a non empty iterable"


class EmptyDatasetException(CustomException):
    """Raised when a loaded or given dataset is empty"""
    def __init__(self):
        super().__init__(ERROR_MSG)
