#########################################################
#
#       Unit Test for discretize
#
#       Author: Xavier Conort
#
#       Copyright DataRobot, Inc. 2014
#
########################################################

from __future__ import division
import numpy as np
import scipy as sp
import pandas
import cPickle
import tempfile
from statsmodels.distributions.empirical_distribution import ECDF

#-locals
from base_task_test import BaseTaskTest
from ModelingMachine.engine.container import Container
from ModelingMachine.engine.tasks.discretize import Discretize
from ModelingMachine.engine.tasks.discretize import buckets_create
from ModelingMachine.engine.tasks.discretize import make_reco
from ModelingMachine.engine.tasks.discretize import find_optimal_buckets


class TestDiscretize(BaseTaskTest):
    """ Test suite for Discretize
    """

    def test_buckets_create(self):
        x = pandas.DataFrame({'a': np.array([1,2,3,4,1,2,3,4])})
        task = buckets_create(ECDF(x['a']),2)
        task.fit(x['a'])
        xnew = pandas.DataFrame({'a': np.array([1,2,3,4,5])})
        res = task.transform(xnew['a'])
        self.assertEqual(np.all(res == np.array([1,1,2,2,2])),True)


    def test_reco(self):
        improvement = {0: 0.005, 0.1: 0.004, 'ridit': 0.003,
                'ridit_p2': 0.002, 'ridit_p3': 0.002, 'x_raw': 0.001}
        reco = make_reco(improvement, 0.1, 0.05, 1.2, 1.2, True)
        self.assertEqual(reco, 'ridit')
        improvement = {0: 0.005, 0.1: 0.004, 'ridit': 0.003,
                'ridit_p2': 0.003, 'ridit_p3': 0.004, 'x_raw': 0.001}
        reco = make_reco(improvement, 0.1, 0.05, 1.2, 1.2, True)
        self.assertEqual(reco, 'ridit_p3')
        improvement = {0: 0.005, 0.1: 0.004, 'ridit': 0.003,
                'ridit_p2': 0.003, 'ridit_p3': 0.003, 'x_raw': 0.001}
        reco = make_reco(improvement, 0.1, 0.01, 1.2, 1.2, True)
        self.assertEqual(reco, 0)
        improvement = {0: 0.005, 0.1: 0.004, 'ridit': 0.003,
                'ridit_p2': 0.003, 'ridit_p3': 0.003, 'x_raw': 0.001}
        reco = make_reco(improvement, 0.1, 0.01, 1.3, 1.2, True)
        self.assertEqual(reco, 0.1)
        improvement = {0: -0.005, 0.1: -0.004, 'ridit': -0.003,
                'ridit_p2': -0.003, 'ridit_p3': -0.003, 'x_raw': -0.01}
        reco = make_reco(improvement, 0.1, 0.01, 1.3, 1.2, False)
        self.assertEqual(reco, 'x_raw')
        improvement = {0: -0.005, 0.1: -0.004, 'ridit': -0.003,
                'ridit_p2': -0.003, 'ridit_p3': -0.003, 'x_raw': -0.01}
        reco = make_reco(improvement, 0.1, 0.01, 1.3, 1.2, True)
        self.assertEqual(reco, 'drop')


    def test_find_optimal_buckets(self):
        X,Y,Z = self.create_bin_data()
        W = np.ones(len(Y))
        for penalty in ['l1','l2']:
            res = find_optimal_buckets(
                    X.dataframe.iloc[:,0], Y, W, 0.5, 1234,
                    [0, 0.1, 0.3], True,
                    'Binary', [0.01,0.1,1], [0.1,1,10], penalty, 0.01)
            self.assertEqual(sorted(res['improvement'].keys()), [0.1, 0.3])
            res = find_optimal_buckets(
                    X.dataframe.iloc[:,3], Y, W, 0.5, 1234,
                    [0, 0.1, 0.3], True,
                    'Binary', [0.01,0.1,1], [0.1,1,10], penalty, 0.01)
            self.assertEqual(sorted(res['improvement'].keys()), [0, 0.1, 0.3])
            res = find_optimal_buckets(
                    X.dataframe.iloc[:,3], Y, W, 0.5, 1234,
                    [0, 0.1, 0.3], False,
                    'Binary', [0.01,0.1,1], [0.1,1,10], penalty, 0.01)
            self.assertEqual(sorted(res['improvement'].keys()),
                    [0, 0.1, 0.3, 'ridit', 'riditAbuckets',
                    'ridit_p2', 'ridit_p3', 'x_raw'])
            self.assertEqual(sorted(res['signif_buckets'].keys()),
                    [0, 0.1, 0.3, 'riditAbuckets'])

    def test_transform(self):
        """ test the transform function of the class"""
        X,Y,Z = self.create_bin_data()
        X.initialize({'weight': pandas.Series(np.ones(len(Y)))})
        task = Discretize('rgl_p=0')
        task.fit(X, Y, Z)
        res = task.transform(X, Y, Z)
        # check if instance
        self.assertIsInstance(res,Container)
        # check if expected result
        p = {'k': -1, 'r': 0}
        self.assertEqual(res(**p).shape, (200, 43))

        X,Y,Z = self.create_bin_data()
        task = Discretize('rgl_p=1')
        task.fit(X, Y, Z)
        res = task.transform(X, Y, Z)
        # check if instance
        self.assertIsInstance(res,Container)
        # check if expected result
        p = {'k': -1, 'r': 0}
        self.assertEqual(res(**p).shape, (200, 89))

        X,Y,Z = self.create_reg_data()
        task = Discretize('rgl_p=0;dscr=1')
        task.fit(X, Y, Z)
        res = task.transform(X, Y, Z)
        # check if instance
        self.assertIsInstance(res,Container)
        # check if expected result
        p = {'k': -1, 'r': 0}
        self.assertEqual(res(**p).shape, (200, 120))
        self.assertEqual(sp.sparse.issparse(res(**p)), True)

        X,Y,Z = self.create_reg_data()
        task = Discretize('rgl_p=1;dscr=1')
        task.fit(X, Y, Z)
        res = task.transform(X, Y, Z)
        # check if instance
        self.assertIsInstance(res,Container)
        # check if expected result
        p = {'k': -1, 'r': 0}
        self.assertEqual(res(**p).shape, (200, 65))
        self.assertEqual(sp.sparse.issparse(res(**p)), True)


    def test_case_with_many_NAs(self):
        """ test the transform function of the class"""
        X,Y,Z = self.create_bin_data()
        X.dataframe.iloc[5:,0] = np.NAN
        task = Discretize('rgl_p=0')
        task.fit(X, Y, Z)
        res = task.transform(X, Y, Z)
        # check if instance
        self.assertIsInstance(res,Container)
        # check if expected result
        p = {'k': -1, 'r': 0}
        self.assertEqual(res(**p).shape, (200, 42))


    def test_pickle(self):
        """ test that the class can be pickled. This is required! """
        X,Y,Z = self.create_bin_data()
        task = Discretize()
        task.fit(X, Y, Z)
        with tempfile.TemporaryFile(mode='w+b') as tf:
            cPickle.dump(task, tf)

if __name__ == '__main__':
    unittest.main()
