import unittest

from common.utilities.injector import inject


def method1(param_a):
    return param_a


def method2(param_b):
    return param_b


def method3(param_a, param_b=1):
    return param_a + param_b


class TestInjector(unittest.TestCase):
    def test_inject(self):
        kwargs = inject(method1, param_a=3)
        self.assertIn('param_a', kwargs)
        self.assertEqual(kwargs['param_a'], 3)
        self.assertEqual(method1(**kwargs), 3)

    def test_inject_discard_extra(self):
        kwargs = inject(method1, param_a=3, param_b=4)
        self.assertIn('param_a', kwargs)
        self.assertEqual(kwargs['param_a'], 3)
        self.assertNotIn('param_b', kwargs)
        self.assertEqual(method1(**kwargs), 3)

    def test_inject_missing_args(self):
        kwargs = inject(method1, param_b=4)
        with self.assertRaises(TypeError):
            method1(**kwargs)
        with self.assertRaises(TypeError):
            # Because that is the same error as this
            method1()

    def test_inject_multi(self):
        kwargs = inject(method3, param_a=3)
        self.assertIn('param_a', kwargs)
        self.assertNotIn('param_b', kwargs)
        self.assertEqual(kwargs['param_a'], 3)
        self.assertEqual(method3(**kwargs), 4)

        kwargs = inject(method3, param_a=3, param_b=5)
        self.assertIn('param_a', kwargs)
        self.assertIn('param_b', kwargs)
        self.assertEqual(kwargs['param_a'], 3)
        self.assertEqual(kwargs['param_b'], 5)
        self.assertEqual(method3(**kwargs), 8)

        kwargs = inject(method3, param_b=5)
        self.assertNotIn('param_a', kwargs)
        self.assertIn('param_b', kwargs)
        self.assertEqual(kwargs['param_b'], 5)
        with self.assertRaises(TypeError):
            method3(**kwargs)
