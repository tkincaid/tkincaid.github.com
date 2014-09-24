import unittest
import numpy as np
import cPickle as pickle

from sklearn.utils import check_arrays

from ModelingMachine.engine.response import Response


class ResponseTest(unittest.TestCase):

    def test_smoke(self):
        r = Response.from_array(np.arange(10))
        self.assertEqual(r.dtype, np.int64)
        np.testing.assert_array_equal(r, np.arange(10))

    def test_arithmetic(self):
        r = Response.from_array(np.arange(10))
        self.assertEqual(r.dtype, np.int64)
        o = r * 2.0
        self.assertEqual(o.dtype, np.float64)
        np.testing.assert_array_equal(r + r, r * 2)

    def test_check(self):
        with self.assertRaises(ValueError):
            r = Response.from_array(range(10))

        with self.assertRaises(ValueError):
            r = Response.from_array((i for i in range(10)))

    def test_check_array(self):
        """Check arrays will convert Response to ndarray. """
        r = Response.from_array(np.arange(10))
        y, = check_arrays(r)
        self.assertTrue(isinstance(y, np.ndarray))

    def test_from_array_attr(self):
        r = Response.from_array(np.arange(10))
        r.ytransf = 'foobar'
        r2 = Response.from_array(r)
        self.assertIsNone(r2.ytransf)
        self.assertEqual(r.ytransf, 'foobar')

    def test_copy(self):
        r = Response.from_array(np.arange(10), ytransf='logy')
        self.assertEqual(r.ytransf, 'logy')
        r2 = r.copy()
        self.assertTrue(isinstance(r2, Response))
        self.assertEqual(r2.ytransf, r.ytransf)

    def test_pickle(self):
        r = Response.from_array(np.arange(10), ytransf='logy')
        r_ser = pickle.dumps(r)
        r_de = pickle.loads(r_ser)

        np.testing.assert_array_equal(r, r_de)
        self.assertEqual(r_de.ytransf, r.ytransf)
