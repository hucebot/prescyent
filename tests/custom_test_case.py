import unittest


class CustomTestCase(unittest.TestCase):
    def assertHasAttr(self, obj, attr: str):
        self.assertTrue(
            hasattr(obj, attr), msg=f"Ojbect {obj}, should have an attribute {attr}"
        )
