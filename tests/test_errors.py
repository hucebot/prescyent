import unittest

from prescyent.utils.errors import EmptyDatasetException
from prescyent.utils.errors.empty_dataset_exception import ERROR_MSG


class BaseExceptionTestCase(unittest.TestCase):

    def test_str_method_inheritance(self):
        self.assertEqual(str(EmptyDatasetException()),
                         "EmptyDatasetException raised: %s" % ERROR_MSG)
