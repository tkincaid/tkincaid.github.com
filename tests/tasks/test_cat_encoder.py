import unittest
import logging
import random
import pandas as pd
import numpy as np
import cPickle as pickle

from numpy.testing import assert_array_equal
from pandas import DataFrame

from ModelingMachine.engine.tasks.cat_encoders import OrdinalEncoder
from ModelingMachine.engine.tasks.cat_encoders import CategoricalToOrdinalConverter
from ModelingMachine.engine.tasks.cat_encoders import CategoricalToOrdinalConverter2
from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.container import Container


class TestCategoricalToOrdinalConverter(unittest.TestCase):
    """ Test suite for CategoricalToOrdinalConverter """
    nsamples = 100

    def test_smoke(self):
        """ smoke test """
        X = pd.DataFrame({'A': ['a', 'a', 'b', 'b']}, dtype=object)
        Y = pd.Series(np.array([0, 1, 0, 1]).astype('int'))
        Z = Partition(4, folds=2, reps=0, total_size=4)

        task = CategoricalToOrdinalConverter('ms=1')
        res = task.fit_transform(Container(X), Y, Z)
        np.testing.assert_array_equal(np.array([0, 0, 1, 1], dtype=np.float32).reshape((4, 1)),
                                      res(**Z[0]))

        # has 4 levels: a, b, missing, other
        self.assertEqual(res.coltypes(**Z[0]), [4])

    def test_nan(self):
        """ test if nans are handled properly """
        X = pd.DataFrame({'A': ['a', np.nan, 'a', None]}, dtype=object)
        Y = pd.Series(np.array([0, 1, 0, 1]).astype('int'))
        Z = Partition(4, folds=2, reps=0, total_size=4)

        task = CategoricalToOrdinalConverter('ms=1')
        res = task.fit_transform(Container(X), Y, Z)
        np.testing.assert_array_equal(np.array([0, -2, 0, -2], dtype=np.float32).reshape((4, 1)),
                                      res(**Z[0]))
        # has 3 coltypes a, missing, other
        self.assertEqual(res.coltypes(**Z[0]), [3])

    def generate_X(self):
        """ create some test data to help in the tests """
        A_pattern = ['a', 'a', 'a', 'c', 'c', 'd', 'e', 'f']
        B_pattern = ['1', '2', '2', '3']
        C_pattern = ['1', 'd', 'f', '3']
        rand = random.Random(13)
        X = pd.DataFrame({'A': [rand.sample(A_pattern, 1)[0] for i in range(self.nsamples)],
                          'B': [rand.sample(B_pattern, 1)[0] for i in range(self.nsamples)],
                          'C': [rand.sample(C_pattern, 1)[0] for i in range(self.nsamples)]},
                          dtype=object)
        return X

    def generate_Y(self):
        Y_pattern = [0, 1, 1]
        rand = random.Random(13)
        Y = pd.Series(np.array([rand.sample(Y_pattern, 1)[0]
                                for i in range(self.nsamples)]).astype('int'))
        return Y

    def generate_Z(self):
        return Partition(self.nsamples, folds=5, reps=0, total_size=self.nsamples)

    def test_transform(self):
        """ Create random data and make sure the cardinality is correct"""
        X = self.generate_X()
        Y = self.generate_Y()
        Z = self.generate_Z()

        task = CategoricalToOrdinalConverter('ms=1')
        task.fit(Container(X), Y, Z)
        res = task.transform(Container(X), Y, Z)
        # check if name
        self.assertEqual(res.colnames(), X.columns.tolist())
        # check if instance
        self.assertIsInstance(res, Container)
        # check if expected result
        for i, col in enumerate(res.colnames()):
            # must be mapped to [0, |col| - 1]
            np.testing.assert_array_equal(np.unique(res()[:, i]),
                                          np.arange(np.unique(X[col]).shape[0]))

        task = CategoricalToOrdinalConverter('ms=1;cmax=3')
        task.fit(Container(X), Y, Z)
        res = task.transform(Container(X), Y, Z)
        # check if name
        self.assertEqual(res.colnames(), ['B'])
        # check if instance
        self.assertIsInstance(res, Container)
        # check if expected result
        np.testing.assert_array_equal(np.unique(res()[:, 0]),
                                      np.arange(np.unique(X['B']).shape[0]))

        task = CategoricalToOrdinalConverter('ms=1;cmax=2')
        task.fit(Container(X), Y, Z)
        res = task.transform(Container(X), Y, Z)
        # check if name
        self.assertEqual(res.colnames(), [])
        # check if instance
        self.assertIsInstance(res, Container)

    def test_pickle(self):
        """ test that the class can be pickled. This is required! """
        X = self.generate_X()
        Y = self.generate_Y()
        Z = self.generate_Z()
        task = CategoricalToOrdinalConverter('ms=1')
        task.fit(Container(X), Y, Z)
        dump = pickle.dumps(task)
        del task
        task_2 = pickle.loads(dump)

        res = task_2.transform(Container(X), Y, Z)
        # check if name
        self.assertEqual(res.colnames(), X.columns.tolist())
        # check if instance
        self.assertIsInstance(res, Container)
        # check if expected result
        for i, col in enumerate(res.colnames()):
            # must be mapped to [0, |col| - 1]
            np.testing.assert_array_equal(np.unique(res()[:, i]),
                                          np.arange(np.unique(X[col]).shape[0]))

    def test_all_below_min_support_container(self):
        X = pd.DataFrame({'A': ['a', 'a', 'b', 'b'],'B': ['a', 'a', 'a', 'a']}, dtype=object)
        Y = pd.Series(np.array([0, 1, 0, 1]).astype('int'))
        Z = Partition(4, folds=2, reps=0, total_size=4)
        task = CategoricalToOrdinalConverter('ms=3')
        res = task.fit_transform(Container(X), Y, Z)

        assert_array_equal(res()[:, 0], -1 * np.ones(4, dtype=np.float32))
        assert_array_equal(res()[:, 1], np.zeros(4, dtype=np.float32))

class TestCategoricalToOrdinalConverter2(unittest.TestCase):
    """ Test suite for CategoricalToOrdinalConverter2 """
    nsamples = 100

    def test_smoke(self):
        """ smoke test """
        X = pd.DataFrame({'A': ['a', 'a', 'b', 'b']}, dtype=object)
        Y = pd.Series(np.array([0, 1, 0, 1]).astype('int'))
        Z = Partition(4, folds=2, reps=0, total_size=4)

        task = CategoricalToOrdinalConverter2('ms=1')
        res = task.fit_transform(Container(X), Y, Z)
        np.testing.assert_array_equal(np.array([0, 0, 1, 1], dtype=np.float32).reshape((4, 1)),
                                      res(**Z[0]))

        # has 4 levels: a, b, missing, other
        self.assertEqual(res.coltypes(**Z[0]), [4])

    def test_nan(self):
        """ test if nans are handled properly """
        X = pd.DataFrame({'A': ['a', np.nan, 'a', None]}, dtype=object)
        Y = pd.Series(np.array([0, 1, 0, 1]).astype('int'))
        Z = Partition(4, folds=2, reps=0, total_size=4)

        task = CategoricalToOrdinalConverter2('ms=1')
        res = task.fit_transform(Container(X), Y, Z)
        np.testing.assert_array_equal(np.array([0, -2, 0, -2], dtype=np.float32).reshape((4, 1)),
                                      res(**Z[0]))
        # has 3 coltypes a, missing, other
        self.assertEqual(res.coltypes(**Z[0]), [3])

    def generate_X(self):
        """ create some test data to help in the tests """
        A_pattern = ['a', 'a', 'a', 'c', 'c', 'd', 'e', 'f']
        B_pattern = ['1', '2', '2', '3']
        C_pattern = ['1', 'd', 'f', '3']
        rand = random.Random(13)
        X = pd.DataFrame({'A': [rand.sample(A_pattern, 1)[0] for i in range(self.nsamples)],
                          'B': [rand.sample(B_pattern, 1)[0] for i in range(self.nsamples)],
                          'C': [rand.sample(C_pattern, 1)[0] for i in range(self.nsamples)]},
                          dtype=object)
        return X

    def generate_Y(self):
        Y_pattern = [0, 1, 1]
        rand = random.Random(13)
        Y = pd.Series(np.array([rand.sample(Y_pattern, 1)[0]
                                for i in range(self.nsamples)]).astype('int'))
        return Y

    def generate_Z(self):
        return Partition(self.nsamples, folds=5, reps=0, total_size=self.nsamples)

    def test_transform(self):
        """ Create random data and make sure the cardinality is correct"""
        X = self.generate_X()
        Y = self.generate_Y()
        Z = self.generate_Z()

        task = CategoricalToOrdinalConverter2('ms=1')
        task.fit(Container(X), Y, Z)
        res = task.transform(Container(X), Y, Z)
        for p in Z:
            # check if name
            self.assertEqual(res.colnames(**p), X.columns.tolist())
            # check if instance
            self.assertIsInstance(res, Container)
            # check if expected result
            for i, col in enumerate(res.colnames(**p)):
                # must be mapped to [0, |col| - 1]
                np.testing.assert_array_equal(np.unique(res(**p)[:, i]),
                                              np.arange(np.unique(X[col]).shape[0]))

        task = CategoricalToOrdinalConverter2('ms=1;cmax=3')
        task.fit(Container(X), Y, Z)
        res = task.transform(Container(X), Y, Z)
        for p in Z:
            # check if name
            self.assertEqual(res.colnames(**p), ['B'])
            # check if instance
            self.assertIsInstance(res, Container)
            # check if expected result
            np.testing.assert_array_equal(np.unique(res(**p)[:, 0]),
                                          np.arange(np.unique(X['B']).shape[0]))

        task = CategoricalToOrdinalConverter2('ms=1;cmax=2')
        task.fit(Container(X), Y, Z)
        res = task.transform(Container(X), Y, Z)
        for p in Z:
            # check if name
            self.assertEqual(res.colnames(**p), [])
            # check if instance
            self.assertIsInstance(res, Container)

    def test_pickle(self):
        """ test that the class can be pickled. This is required! """
        X = self.generate_X()
        Y = self.generate_Y()
        Z = self.generate_Z()
        task = CategoricalToOrdinalConverter2('ms=1')
        task.fit(Container(X), Y, Z)
        dump = pickle.dumps(task)
        del task
        task_2 = pickle.loads(dump)

        res = task_2.transform(Container(X), Y, Z)
        for p in Z:
            # check if name
            self.assertEqual(res.colnames(**p), X.columns.tolist())
            # check if instance
            self.assertIsInstance(res, Container)
            # check if expected result
            for i, col in enumerate(res.colnames()):
                # must be mapped to [0, |col| - 1]
                np.testing.assert_array_equal(np.unique(res(**p)[:, i]),
                                              np.arange(np.unique(X[col]).shape[0]))

    def test_all_below_min_support_container(self):
        X = pd.DataFrame({'A': ['a', 'a', 'b', 'b', 'c', 'c'],'B': ['a', 'a', 'a', 'a', 'a', 'a']}, dtype=object)
        Y = pd.Series(np.array([0, 1, 0, 1, 0, 1]).astype('int'))
        Z = Partition(6, folds=2, reps=0, total_size=6)
        task = CategoricalToOrdinalConverter2('ms=3')
        res = task.fit_transform(Container(X), Y, Z)
        for p in Z:
            assert_array_equal(res(**p)[:, 0], -1 * np.ones(6, dtype=np.float32))
            assert_array_equal(res(**p)[:, 1], np.zeros(6, dtype=np.float32))

class OrdinalEncoderTest(unittest.TestCase):

    def test_ordinal_enc(self):
        """Smoke test for ordinal encoder. """

        df = DataFrame(data={'icao': ['CRJ2', 'CRJ2', 'A380', 'B737', 'B737', 'B737'],
                             'weather': ['fog', 'fog', 'oc', 'sun', 'oc', 'oc']})

        ord_enc = OrdinalEncoder(columns=['icao'], min_support=2)
        ord_enc.fit(df)
        df_prime = ord_enc.transform(df)

        self.assertEqual(df_prime.ix[0, 'icao'], 1)
        self.assertEqual(df_prime.ix[2, 'icao'], -1)  # -1 is other column
        self.assertEqual(df_prime.ix[4, 'icao'], 0)
        self.assertEqual(df_prime.ix[0, 'weather'], 'fog')

        df = DataFrame(data={'icao': ['B777'],
                             'weather': ['fog', ]})
        df = ord_enc.transform(df)
        self.assertEqual(df.ix[0, 'icao'], -1)

    def test_ordinal_enc_cols(self):
        """Test if column names are taken from dataframe. """

        df = DataFrame(data={'icao': ['CRJ2', 'CRJ2', 'A380', 'B737', 'B737', 'B737'],
                             'weather': ['fog', 'fog', 'oc', 'sun', 'oc', 'oc']})

        ord_enc = OrdinalEncoder(min_support=2)
        ord_enc.fit(df)
        df_prime = ord_enc.transform(df)

        self.assertEqual(df_prime.ix[0, 'icao'], 1)
        self.assertEqual(df_prime.ix[2, 'icao'], -1)  # -1 is other column
        self.assertEqual(df_prime.ix[4, 'icao'], 0)
        self.assertEqual(df_prime.ix[0, 'weather'], 0)

    def test_ordinal_enc_copy(self):
        """Test for ordinal encoder copy. """

        df = DataFrame(data={'icao': ['CRJ2', 'CRJ2', 'A380', 'B737', 'B737', 'B737'],
                             'weather': ['fog', 'fog', 'oc', 'sun', 'oc', 'oc']})

        ord_enc = OrdinalEncoder(columns=['icao'], min_support=2, copy=True)
        ord_enc.fit(df)
        df_prime = ord_enc.transform(df)

        self.assertEqual(df_prime.ix[0, 'icao'], 1)
        self.assertEqual(df_prime.ix[2, 'icao'], -1)  # -1 is other column
        self.assertEqual(df_prime.ix[4, 'icao'], 0)

        self.assertEqual(df.ix[0, 'icao'], 'CRJ2')

        ord_enc = OrdinalEncoder(columns=['icao'], min_support=2, copy=False)
        ord_enc.fit(df)
        df_prime = ord_enc.transform(df)

        self.assertEqual(df_prime.ix[0, 'icao'], 1)
        self.assertEqual(df.ix[0, 'icao'], 1)

    def test_ordinal_enc_nan(self):
        df = DataFrame(data={'foo': ['B777', None],
                             'bar': ['fog', np.nan]})
        ord_enc = OrdinalEncoder(columns=['foo', 'bar'], min_support=1)
        ord_enc.fit(df)
        df = ord_enc.transform(df)
        self.assertEqual(df.ix[1, 'foo'], -2)
        self.assertEqual(df.ix[1, 'bar'], -2)

    def test_ordinal_enc_nan_2(self):
        df = DataFrame(data={'foo': ['B777', None, 'A2', 'B777'],
                             'bar': ['fog', np.nan, np.nan, 'fog']})
        ord_enc = OrdinalEncoder(columns=['foo', 'bar'], min_support=2)
        ord_enc.fit(df)
        df = ord_enc.transform(df)

        self.assertEqual(df.ix[0, 'foo'], 0)
        self.assertEqual(df.ix[1, 'foo'], -2)
        self.assertEqual(df.ix[2, 'foo'], -1)
        self.assertEqual(df.ix[1, 'bar'], -2)

    def test_ordinal_enc_rand(self):
        """Test for random ordinal encoder. """

        df = DataFrame(data={'icao': ['CRJ2', 'CRJ2', 'A380', 'B737', 'B737', 'B737'],
                             'weather': ['fog', 'fog', 'oc', 'sun', 'oc', 'oc']})

        ord_enc = OrdinalEncoder(min_support=2, random_scale=True, random_state=0)
        ord_enc.fit(df)
        df_prime = ord_enc.transform(df)

        self.assertEqual(df_prime.ix[0, 'icao'], 0)
        self.assertEqual(df_prime.ix[2, 'icao'], -1)  # -1 is other column
        self.assertEqual(df_prime.ix[4, 'icao'], 1)

    def test_all_below_min_support(self):
        """Check corner case when all values are below zero. """
        df = pd.DataFrame({'A': ['a', 'a', 'b', 'b'],'B': ['a', 'a', 'a', 'a']}, dtype=object)
        ord_enc = OrdinalEncoder(min_support=3, random_scale=True, random_state=0)
        df_prime = ord_enc.fit_transform(df)

        assert_array_equal(df_prime.ix[:, 'A'], -1 * np.ones(4, dtype=np.float32))
        assert_array_equal(df_prime.ix[:, 'B'], np.zeros(4, dtype=np.float32))

    def test_ordinal_enc_freq(self):
        """ Test for frequency ordinal encoder. """

        df = DataFrame(data={'icao': ['CRJ2', 'CRJ2', 'A380', 'B737', 'B737', 'B737'],
                             'weather': ['fog', 'fog', 'oc', 'sun', 'oc', 'oc']})

        ord_enc = OrdinalEncoder(min_support=2, method='freq')
        ord_enc.fit(df)
        df_prime = ord_enc.transform(df)
        print df_prime

        self.assertEqual(df_prime.ix[0, 'icao'], 1)
        self.assertEqual(df_prime.ix[2, 'icao'], -1)  # -1 is other column
        self.assertEqual(df_prime.ix[4, 'icao'], 0)
        self.assertEqual(df_prime.ix[0, 'weather'], 1)
        self.assertEqual(df_prime.ix[2, 'weather'], 0)
        self.assertEqual(df_prime.ix[3, 'weather'], -1)

    def test_ordinal_enc_resp(self):
        """ Test for response ordinal encoder.
        """

        df = DataFrame(data={'icao': ['CRJ2', 'CRJ2', 'A380', 'B737', 'B737', 'B737'],
                             'weather': ['fog', 'fog', 'oc', 'sun', 'oc', 'oc']})

        y = np.array([-1, 1, 10, -2, 2, 0])
        ord_enc = OrdinalEncoder(min_support=2, method='resp')
        ord_enc.fit(df, y)
        df_prime = ord_enc.transform(df)
        print df_prime

        # CRJ2: 1, B737: 0 (tied, broken by lex)
        # fog: 0, oc: 1 (mean respons fog=0, oc=12)
        self.assertEqual(df_prime.ix[0, 'icao'], 1)
        self.assertEqual(df_prime.ix[2, 'icao'], -1)  # -1 is other column
        self.assertEqual(df_prime.ix[4, 'icao'], 0)
        self.assertEqual(df_prime.ix[0, 'weather'], 0)
        self.assertEqual(df_prime.ix[2, 'weather'], 1)
        self.assertEqual(df_prime.ix[3, 'weather'], -1)


if __name__ == '__main__':
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    unittest.main()
