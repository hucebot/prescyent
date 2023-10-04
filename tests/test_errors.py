import unittest

from prescyent.utils.errors import DatasetEmptyException
from prescyent.utils.errors.dataset.empty_dataset_exception import ERROR_MSG


class BaseExceptionTestCase(unittest.TestCase):
    def test_str_method_inheritance(self):
        self.assertEqual(str(DatasetEmptyException()), ERROR_MSG)
