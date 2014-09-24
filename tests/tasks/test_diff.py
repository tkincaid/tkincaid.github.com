#########################################################
#
#       Unit Test for s2w_transformer
#
#       Author: Xavier Conort
#
#       Copyright DataRobot, Inc. 2014
#
########################################################

from __future__ import division
import numpy as np
import pandas as pd
import unittest
import pytest
import random
import tempfile
import cPickle
import copy

#-locals
from ModelingMachine.engine.container import Container
from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.tasks.s2w_transformer import mmDIFF


class TestmmDIFF(unittest.TestCase):
    """ Test suite for mmDIFF
    """

    def generate_data(self, nrows=5000, ncols=4, seed=56):
        colnames = ['X'+str(i) for i in xrange(ncols)]
        np.random.seed(seed)
        x = abs(np.random.randn(nrows, ncols))
        x[:, 1] = x[:, 0]*1.5 + x[:, 1]
        X = Container()
        X.initialize({'weight': pd.Series(np.ones(nrows))})
        X.add(x, colnames=colnames)
        Z = Partition(size=nrows, folds=1, reps=1, total_size=nrows)
        Z.set(max_reps=1, max_folds=0)
        Y = 3 * (x[:, 1] - x[:, 0]) + 0.2 * x[:, 3]
        return X, Y, Z

    def test_fit(self):
        """ test the fit function of the class """
        X, Y, Z = self.generate_data()
        p = {'k': -1, 'r': 0}
        key = (p['r'], p['k'])
        task = mmDIFF()
        fit_result = task.fit(X, Y, Z)
        self.assertEqual(fit_result.new_diff[key] == ['X0_XX_X1'], True)

    def test_unique_col(self):
        """ test the fit function of the class """
        X, Y, Z = self.generate_data(nrows=10)
        x = abs(np.random.randn(10, 1))
        X = Container()
        X.add(x)
        p = {'k': -1, 'r': 0}
        key = (p['r'], p['k'])
        task = mmDIFF()
        fit_result = task.fit(X, Y, Z)
        self.assertEqual(fit_result.new_diff[key] == [], True)

    def test_transform(self):
        """ test the transform function of the class """
        X, Y, Z = self.generate_data()
        p = {'k': -1, 'r': 0}
        #task = mmDIFF('ivr=0.00;t_f=0.3;rs=12')
        task = mmDIFF('ivr=0.01;rs=3252')
        task.fit(X, Y, Z)
        res = task.transform(X, Y, Z)
        # check if Instance
        self.assertIsInstance(res, Container)
        # check if expected response
        self.assertEqual(res.colnames(**p) == ['X0_minus_X1'], True)
        self.assertEqual(np.all(res(**p)[:, 0] == X()[:, 0] - X()[:, 1]), True)

        filter = Y > Y.mean()
        Y[filter] = 1
        Y[filter == False] = 0
        task.fit(X, Y, Z)
        res = task.transform(X, Y, Z)
        # check if Instance
        self.assertIsInstance(res, Container)
        # check if expected response
        self.assertEqual(res.colnames(**p) == ['X0_minus_X1'], True)
        self.assertEqual(np.all(res(**p)[:, 0] == X()[:, 0] - X()[:, 1]), True)

    def test_pickle(self):
        """ test that the class can be pickled. This is required! """
        X, Y, Z = self.generate_data(nrows=200)
        task = mmDIFF()
        task.fit(X, Y, Z)
        with tempfile.TemporaryFile(mode='w+b') as tf:
            cPickle.dump(task, tf)

if __name__ == '__main__':
    unittest.main()
