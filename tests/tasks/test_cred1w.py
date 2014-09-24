#########################################################
#
#       Unit Test for Cred1w Task
#
#       Author: Xavier Conort
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

from __future__ import division
import numpy as np
import scipy as sp
import pandas as pd
import unittest
import random
import tempfile
import cPickle
import pprint

from ModelingMachine.engine.tasks.cred_converters import Cred1w
from ModelingMachine.engine.tasks.cred_converters import *

from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.container import Container

from common.services.flippers import FLIPPERS


class TestCred1wTasks(unittest.TestCase):
    """ Test suite for Cred1w
    """
    nsamples=100

    def test_constant(self):
        """ test it works in presence of a constant value """
        X = pd.DataFrame({  'A' : ['a','a','a','a']}, dtype=object)
        Y = pd.Series( np.array([0,1,0,1]).astype('int') )
        Z = Partition(4,folds=2,reps=0,total_size=4)

        task = Cred1w('cmin=0')
        res = task.fit_transform(Container(X),Y,Z)
        # check if equal to mean of training observations for Y
        self.assertEqual(np.all(res(**Z[0])==Y[Z.T(**Z[0])].mean()),True)

    def test_empty(self):
        """ test it works in presence of a constant value """
        X = pd.DataFrame({  'A' : ['a','a','a','a']}, dtype=object)
        Y = pd.Series( np.array([0,1,0,1]).astype('int') )
        Z = Partition(4,folds=2,reps=0,total_size=4)

        task = Cred1w('cmin=10')
        res = task.fit_transform(Container(X),Y,Z)
        # check it is empty
        self.assertEqual(np.all(res.colnames(**Z[0])==[]),True)

    def generate_X(self):
        """ create some test data to help in the tests """
        A_pattern = ['a','a','a','c','c','d','e','f']
        B_pattern = ['1','2','2','3']
        C_pattern = ['1','d','f','3']
        X=  pd.DataFrame({  'A' : [random.sample(A_pattern,1)[0] for i in range(self.nsamples)],
                    'B' : [random.sample(B_pattern,1)[0] for i in range(self.nsamples)],
                    'C' : [random.sample(C_pattern,1)[0] for i in range(self.nsamples)]},
                    dtype=object)
        return X

    def generate_Y(self):
        Y_pattern = [0,1,1]
        Y = pd.Series( np.array([random.sample(Y_pattern,1)[0] for i in range(self.nsamples)]).astype('int') )
        return Y

    def generate_Z(self):
        return Partition(self.nsamples,folds=5,reps=0,total_size=self.nsamples)

    def test_transform(self):
        """ test the transform function of the class """
        X = self.generate_X()
        Y = self.generate_Y()
        Z = self.generate_Z()

        task = Cred1w('cmin=0')
        task.fit(Container(X),Y,Z)
        res = task.transform(Container(X),Y,Z)
        # check if name
        self.assertEqual(np.all(res.colnames(**Z[3])==['cred_A','cred_B','cred_C']),True)
        # check if instance
        self.assertIsInstance(res,Container)
        # check if expected result
        self.assertEqual(np.all(res(**Z[0])>0),True)
        self.assertEqual(np.all(res(**Z[1])<1),True)

        # check with new data
        Xnew= pd.DataFrame({'A':['g','g','g'], 'B':[1,1,1], 'C':['h','h','h']},dtype=object)
        res = task.transform(Container(Xnew),Y,Z)
        # check if new category mean is equal to M
        p = Z[3]
        key = (p['r'],p['k'])
        self.assertEqual(round(res(**p).mean(),8)==round(task.cmap[key][0]['M'],8),True)

    def test_functionality_unchanged(self):
        X = pd.DataFrame({'a': ['a', 'b', 'b', 'b'],
                          'b': ['a', 'a', 'a', 'b']})
        Y = pd.Series([0.3, 0.4, 0.5, 0.7])
        Z = Partition(4, folds=5, reps=0, total_size=4)

        task = Cred1w('cmin=0')
        task.fit(Container(X), Y, Z)

        t = task.transform(Container(X), Y, Z)

        check_val = t(r=-1, k=0)
        known_val = [[0.38333333, 0.4],
                     [0.41428571, 0.4],
                     [0.41428571, 0.4],
                     [0.41428571, 0.4]]
        np.testing.assert_almost_equal(check_val, known_val)

    def test_functionality_unchanged_int(self):
        """Regression test for a bug when categoricals where encoded as ints. """
        X = pd.DataFrame({'a': [1, 2, 2, 2],
                          'b': [1, 1, 1, 2]})
        Y = pd.Series([0.3, 0.4, 0.5, 0.7])
        Z = Partition(4, folds=5, reps=0, total_size=4)

        task = Cred1w('cmin=0')
        task.fit(Container(X), Y, Z)

        t = task.transform(Container(X), Y, Z)

        check_val = t(r=-1, k=0)
        known_val = [[0.38333333, 0.4],
                     [0.41428571, 0.4],
                     [0.41428571, 0.4],
                     [0.41428571, 0.4]]
        np.testing.assert_almost_equal(check_val, known_val)

    def test_nan_calculation_identical_to_regular(self):
        X = pd.DataFrame({'a': pd.Series(['a', 'b', 'a', 'b', 'a', 'b'])},
                         index=range(6))
        X2 = pd.DataFrame({'a': pd.Series(['b', 'b', 'b'], index=[1,3,5])},
                          index=range(6))
        Y = pd.Series([1, 2, 3, 4, 5, 6])
        Z = Partition(5, folds=5, reps=0, total_size=5)

        task1 = Cred1w('cmin=0')
        task2 = Cred1w('cmin=0')

        task1.fit(Container(X), Y, Z)
        task2.fit(Container(X2), Y, Z)

        t1 = task1.transform(Container(X), Y, Z)
        t2 = task2.transform(Container(X2), Y, Z)

        ref1 = t1(r=-1, k=0)
        ref2 = t2(r=-1, k=0)
        self.assertTrue(np.all(ref1 == ref2))

    def test_pickle(self):
        """ test that the class can be pickled. This is required! """
        X = self.generate_X()
        Y = self.generate_Y()
        Z = self.generate_Z()
        task = Cred1w()
        task.fit(Container(X),Y,Z)
        with tempfile.TemporaryFile(mode='w+b') as tf:
            cPickle.dump(task, tf)

    def test_one_way(self):
        ground_truth = [{'M': 0.5830382853043281,
                         'cnt_pcat': {(0, 0): 5, (1, 1): 4, (2, 2): 11},
                         'cred_k': 10.467960697881688,
                         'meanY_pcat': {(0, 0): 0.69225523092846997,
                                        (1, 1): 0.46517327372875639,
                                        (2, 2): 0.57625422332083531}}]
        rng = np.random.RandomState(13)

        df = pd.DataFrame(data={'foo': rng.randint(0, 3, 20)})
        y = rng.rand(20)
        res = one_way_statistics(df, y, 'auto', 1)

        self.assertEqual(res[0]['cnt_pcat'], ground_truth[0]['cnt_pcat'])
        np.testing.assert_almost_equal(res[0]['M'], ground_truth[0]['M'])
        np.testing.assert_almost_equal(res[0]['cred_k'], ground_truth[0]['cred_k'])
        np.testing.assert_almost_equal(res[0]['meanY_pcat'][(0, 0)],
                                       ground_truth[0]['meanY_pcat'][(0, 0)])
        np.testing.assert_almost_equal(res[0]['meanY_pcat'][(1, 1)],
                                       ground_truth[0]['meanY_pcat'][(1, 1)])
        np.testing.assert_almost_equal(res[0]['meanY_pcat'][(2, 2)],
                                       ground_truth[0]['meanY_pcat'][(2, 2)])


class TestCredHelpers(unittest.TestCase):

    def test_single_column_statistics_keeps_nans(self):
        a = pd.Series(['a', 'b', 'a', 'b', np.nan], index=[0, 1, 2, 3, 5])
        b = pd.Series(['a', 'a', 'b', 'b', np.nan], index=[1, 2, 3, 4, 5])
        data = pd.DataFrame({'a': a, 'b': b})
        response = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        nankey = (np.nan, np.nan)

        stats = single_column_statistics(data, response, 5, 0)
        means = stats['meanY_pcat']
        self.assertEqual(means[nankey], 0.6)



if __name__ == '__main__':
    unittest.main()
