#########################################################
#
#       Unit Test for best_transformer
#
#       Author: Xavier Conort
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

from __future__ import division
import numpy as np
import pandas
import unittest
import pytest
import random
import tempfile
import cPickle
import copy

#-locals
from ModelingMachine.engine.container import Container
from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.tasks.transformers import mmUNIF
from ModelingMachine.engine.tasks.best_transformer import mmBTRANSF


class TestmmBTRANSF(unittest.TestCase):
    """ Test suite for mmBTRANSF
    """

    def generate_data(self, nrows=5000, ncols=4, seed=56):
        colnames = ['X'+str(i) for i in xrange(ncols)]
        np.random.seed(seed)
        x = np.random.randn(nrows, ncols)
        x[:, 1] = abs(x[:, 1])
        x[:, 3] = 1/abs(x[:, 3])
        #create 2 NA flags
        x = np.column_stack((x, np.zeros(nrows)))
        colnames.append('X0-mi')
        x = np.column_stack((x, np.zeros(nrows)))
        colnames.append('X3-mi')
        # create a case where names containing -mi is not a flag
        colnames[1] = 'X1-mi'
        X = Container()
        X.initialize({'weight': pandas.Series(np.ones(nrows))})
        X.add(x, colnames=colnames)
        Z = Partition(size=nrows, folds=1, reps=1, total_size=nrows)
        Z.set(max_reps=1, max_folds=0)
        task = mmUNIF()
        Y = x[:, 1]
        X_unif = task.fit_transform(X, Y, Z)
        Y = 50*x[:, 1] + 20*X_unif()[:, 3] + 100*(X_unif()[:, 3]) ** 2

        return X, X_unif, Y, Z

    @pytest.mark.dscomp
    def generate_randomY(self, nrows=5000, seed=56):
        np.random.seed(seed)
        Y = np.random.randn(nrows, 1)[:, 0]
        return Y

    def generate_data_logX(self, nrows=5000, ncols=4, seed=85):
        colnames = ['X'+str(i) for i in xrange(ncols)]
        np.random.seed(seed)
        x = np.random.normal(0.0, 1.0, size=(nrows, ncols))
        x[:, 1] = (x[:, 1])
        x[:, 3] = np.exp(x[:, 3] ** 2)
        #create 2 NA flags
        x = np.column_stack((x, np.zeros(nrows)))
        colnames.append('X0-mi')
        x = np.column_stack((x, np.zeros(nrows)))
        colnames.append('X3-mi')
        # create a case where names containing -mi is not a flag
        colnames[1] = 'X1-mi'
        X = Container()
        X.initialize({'weight': pandas.Series(np.ones(nrows))})
        X.add(x, colnames=colnames)
        Z = Partition(size=nrows, folds=1, reps=1, total_size=nrows)
        Z.set(max_reps=1, max_folds=0)
        Y = x[:, 1]
        Y = 120*x[:, 1] + 100*np.log(x[:, 3])
        return X, Y, Z

    @pytest.mark.dscomp
    def test_fit(self):
        """ test the fit function of the class """
        X, X_unif, Y, Z = self.generate_data()
        p = {'k': -1, 'r': 0}
        key = (p['r'], p['k'])
        task = mmBTRANSF('t=0')
        fit_result = task.fit(X, Y, Z)
        self.assertEqual(fit_result.best_transfo[key].shape[0] == 4, True)

    @pytest.mark.dscomp
    def test_poisson(self):
        """ test the fit function of the class """
        X, X_unif, Y, Z = self.generate_data()
        Y[Y<0] = 0
        p = {'k': -1, 'r': 0}
        key = (p['r'], p['k'])
        task = mmBTRANSF('dist=1')
        fit_result = task.fit(X, Y, Z)
        self.assertEqual(fit_result.best_transfo[key].shape[0] == 4, True)

    @pytest.mark.dscomp
    def test_gamma(self):
        """ test the fit function of the class """
        X, X_unif, Y, Z = self.generate_data()
        Y[Y<=1] = 1
        p = {'k': -1, 'r': 0}
        key = (p['r'], p['k'])
        task = mmBTRANSF('dist=3')
        fit_result = task.fit(X, Y, Z)
        self.assertEqual(fit_result.best_transfo[key].shape[0] == 4, True)

    @pytest.mark.dscomp
    def test_tweedie(self):
        """ test the fit function of the class """
        X, X_unif, Y, Z = self.generate_data()
        Y[Y<0] = 0
        p = {'k': -1, 'r': 0}
        key = (p['r'], p['k'])
        task = mmBTRANSF('dist=4;im=1')
        fit_result = task.fit(X, Y, Z)
        self.assertEqual(fit_result.best_transfo[key].shape[0] == 4, True)

    @pytest.mark.dscomp
    def test_transform(self):
        """ test the transform function of the class """
        X, X_unif, Y, Z = self.generate_data()
        p = {'k': -1, 'r': 0}
        task = mmBTRANSF('t=2;im=sm')
        task.fit(X, Y, Z)
        res = task.transform(X, Y, Z)
        # check if Instance
        self.assertIsInstance(res, Container)
        # check if expected response
        self.assertEqual(res.colnames(**p) == ['X1-mi', 'unif_X3', 'unif_X3_p2', 'X3-mi'], True)
        self.assertEqual(np.all(res(**p)[:, 0] == X()[:, 1]), True)
        self.assertEqual(np.all(res(**p)[:, 2] - X_unif()[:, 3] ** 2 == 0), True)

    def test_randomY(self):
        """ test the transform function of the class """
        X, X_unif, Y, Z = self.generate_data()
        Y = self.generate_randomY(seed=1568)
        p = {'k': -1, 'r': 0}
        task = mmBTRANSF('d=1;p=0.5;t=2;im=C')
        task.fit(X, Y, Z)
        res = task.transform(X, Y, Z)
        # check if Instance
        self.assertIsInstance(res, Container)
        # check if expected response
        self.assertEqual(res.colnames(**p) == ['zeros___'], True)
        self.assertEqual(np.all(res(**p)[:, 0] == 0), True)

    @pytest.mark.dscomp
    def test_newdata(self):
        """ test the transform function of the class """
        X, X_unif, Y, Z = self.generate_data()
        p = {'k': -1, 'r': 0}
        task = mmBTRANSF('t=2;im=sm')
        task.fit(X, Y, Z)

        X, X_unif, Y, Z = self.generate_data(nrows=50, ncols=4, seed=35)
        res = task.transform(X, Y, Z)
        # check if Instance
        self.assertIsInstance(res, Container)
        # check if expected response
        self.assertEqual(res.colnames(**p) == ['X1-mi', 'unif_X3', 'unif_X3_p2', 'X3-mi'], True)

    @pytest.mark.dscomp
    def test_transform_logX(self):
        """ test the transform function of the class """
        X, Y, Z = self.generate_data_logX()
        p = {'k': -1, 'r': 0}
        task = mmBTRANSF('im=C;p=0.001')
        task.fit(X, Y, Z)
        res = task.transform(X, Y, Z)
        # check if Instance
        self.assertIsInstance(res, Container)
        # check if expected response
        self.assertEqual(res.colnames(**p) ==
                ['X1-mi', 'log_X3', 'X3-mi'], True)
        np.testing.assert_allclose(res(**p)[:, 1], np.log(X()[:, 3]))

    @pytest.mark.dscomp
    def test_transform_logX_with_new_negative_data(self):
        """ test the transform function of the class """
        X, Y, Z = self.generate_data_logX()
        p = {'k': -1, 'r': 0}
        task = mmBTRANSF('im=C')
        task.fit(X, Y, Z)

        # add negative values
        X_new = Container()
        x = X(**p)
        x[0,3] = -2
        X_new.add(x, colnames=X.colnames(**p))

        res = task.transform(X_new, Y, Z)
        # check if Instance
        self.assertIsInstance(res, Container)
        # check if expected response
        self.assertEqual(res.colnames(**p) ==
                ['X1-mi', 'log_X3', 'X3-mi'], True)

    @pytest.mark.dscomp
    def test_pickle(self):
        """ test that the class can be pickled. This is required! """
        X, X_unif, Y, Z = self.generate_data(nrows=200)
        task = mmBTRANSF()
        task.fit(X, Y, Z)
        with tempfile.TemporaryFile(mode='w+b') as tf:
            cPickle.dump(task, tf)

if __name__ == '__main__':
    unittest.main()
