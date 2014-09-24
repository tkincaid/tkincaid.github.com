import unittest

from common.services.flippers import GlobalFlipper

class TestGlobalFlipper(unittest.TestCase):

    def setUp(self):
        self.flipper = GlobalFlipper()

    def test_get_when_not_defined(self):
        x = self.flipper.get('flipper_that_is_not_defined_anywhere')
        self.assertIsNone(x)

    def test_get_when_defined(self):
        x = self.flipper.get('flipper_test')
        self.assertIsNotNone(x)

    def test_direct_access_of_property(self):
        x = self.flipper.flipper_test
        self.assertTrue(x)

    def test_direct_access_of_undef_prop_should_be_none(self):
        x = self.flipper.flipper_that_is_not_defined_anywhere
        self.assertIsNone(x)
