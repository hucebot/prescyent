import unittest
import os


os.environ.setdefault("TQDM_DISABLE", "1")


class CustomTestCase(unittest.TestCase):
    def assertHasAttr(self, obj, attr: str):
        self.assertTrue(
            hasattr(obj, attr), msg=f"Ojbect {obj}, should have an attribute {attr}"
        )
