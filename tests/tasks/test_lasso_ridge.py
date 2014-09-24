#########################################################
#
#       Unit Test for lasso_ridge.py
#
#       Author: Xavier Conort
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

import unittest
import pytest
import numpy as np
import cPickle as pickle

from sklearn.linear_model import Lasso, Ridge
from sklearn.datasets import make_regression

from base_task_test import BaseTaskTest
from ModelingMachine.engine.tasks.lasso_ridge import RegL1,RegL2, RegL1BlockCD, _CDRegressor

class TestLassoRidge(BaseTaskTest):

    @pytest.mark.dscomp
    def test_arguments(self):
        xt = np.array([[1,2,3],[4,5,6],[7,8,9]])
        yt = np.array([0,1,3])
        self.check_arguments(RegL1,Lasso,xt,yt)
        self.check_arguments(RegL1BlockCD, _CDRegressor,xt,yt)
        self.check_arguments(RegL2, Ridge, xt, yt)

    def test_defaults(self):
        self.check_default_arguments(RegL1, Lasso,['alpha'])
        self.check_default_arguments(RegL2, Ridge,['alpha'])
        self.check_default_arguments(RegL1BlockCD, _CDRegressor, ['alpha', 'penalty', 'random_state'])

    @pytest.mark.dscomp
    def test_lasso(self):
        X,Y,Z = self.create_bin_data()
        task = self.check_task('LASSO',X,Y,Z,transform=True)

    @pytest.mark.dscomp
    def test_lasso_cdregressor(self):
        X,Y,Z = self.create_bin_data()
        task = self.check_task('LASSO2',X,Y,Z,transform=True)

    @pytest.mark.dscomp
    def test_ridge(self):
        X,Y,Z = self.create_bin_data()
        t = self.check_task('RIDGE',X,Y,Z,transform=False)

    def test_lasso_cdregressor_coef(self):
        X, y = make_regression()
        est = _CDRegressor(fit_intercept=True)
        est.fit(X, y)
        self.assertEqual(est.coef_.flatten().shape[0], X.shape[0])
        self.assertNotEqual(est.intercept_, 0.0)

        est = _CDRegressor(fit_intercept=False)
        est.fit(X, y)
        self.assertEqual(est.coef_.flatten().shape[0], X.shape[0])
        self.assertEqual(est.intercept_, 0.0)

    def test_lasso_cdregressor_pickle(self):
        X, y = make_regression()
        est = _CDRegressor(fit_intercept=True)
        est.fit(X, y)

        buf = pickle.dumps(est)
        est2 = pickle.loads(buf)
        np.testing.assert_array_equal(est.coef_, est2.coef_)


if __name__ == '__main__':
    unittest.main()
