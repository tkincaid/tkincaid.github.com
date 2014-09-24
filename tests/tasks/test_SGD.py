#########################################################
#
#       Unit Test for tasks/sgd.py
#
#       Author: Tom de Godoy, Peter Prettenhofer
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

import unittest
import pytest
import numpy as np

from mock import patch
from scipy import sparse as sp
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.linear_model.tests.test_sgd import DenseSGDRegressorTestCase
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

from base_task_test import BaseTaskTest
from ModelingMachine.engine.tasks.sgd import AutoSGDRegressor
from ModelingMachine.engine.tasks.sgd import SGDC
from ModelingMachine.engine.tasks.sgd import SGDR
from ModelingMachine.engine.tasks.sgd import SGDRA
from ModelingMachine.engine.tasks.sgd import n_iter_heuristic
from common.exceptions import NumericalInstabilityError


DEFAULT_CEIL = 100


class TestSGD(BaseTaskTest):

    def test_arguments(self):
        xt = np.array([[1,2,3], [4,5,6], [7,8,9]])
        yt = np.array([0,1,0])
        self.check_arguments_min(SGDC, SGDClassifier, xt, yt)
        self.check_arguments_min(SGDR, SGDRegressor, xt, yt)
        self.check_arguments_min(SGDRA, AutoSGDRegressor, xt, yt)

    @pytest.mark.dscomp
    def test_01(self):
        X,Y,Z = self.create_bin_data()
        self.check_task('SGDC',X,Y,Z,transform=True)

    def test_02(self):
        X, Y, Z = self.create_reg_data()
        self.check_task('SGDR', X, Y, Z, transform=True)

    def test_03(self):
        X, Y, Z = self.create_reg_data()
        self.check_task('SGDRS', X, Y, Z, transform=True)

    @pytest.mark.dscomp
    def test_sgdra(self):
        X, Y, Z = self.create_reg_data()
        self.check_task('SGDRA', X, Y, Z, transform=True)

    @pytest.mark.dscomp
    def test_n_iter_auto_regression(self):
        X, Y, Z = self.create_reg_data()
        for vertex in 'SGDRA,SGDR,SGDRS'.split(','):
            task = self.check_task('%s ni=auto' % vertex, X, Y, Z, transform=True)
            self.assertEqual(task.parameters['n_iter'], DEFAULT_CEIL)

    @pytest.mark.dscomp
    def test_n_iter_auto_classification(self):
        X, Y, Z = self.create_bin_data()
        for vertex in 'SGDC'.split(','):
            task = self.check_task('%s ni=auto' % vertex, X, Y, Z, transform=True)
            self.assertEqual(task.parameters['n_iter'], DEFAULT_CEIL)


def test_n_iter_heuristic():
    """Checks correct output of n_iter heuristic on sparse and dense matrices. """
    X = np.random.rand(100, 10)
    ceil = 100
    n_iter = n_iter_heuristic(X, ceil=ceil)
    assert n_iter == min(ceil, int(np.ceil(10.0 ** 7.0 / (100 * 10))))

    X = sp.csr_matrix(X)
    n_iter = n_iter_heuristic(X, ceil=ceil)
    assert n_iter == min(ceil, int(np.ceil(10.0 ** 6.0 / 100)))

    X = np.empty((10 ** 8, 1))
    assert n_iter_heuristic(X) == 1


class SparseAutoSGDRegressor(AutoSGDRegressor):
    """Test factory for sklearn SGDRegressor test case - convert input data to sparse format. """

    def fit(self, X, y, *args, **kw):
        X = sp.csr_matrix(X)
        return super(SparseAutoSGDRegressor, self).fit(X, y, *args, **kw)

    def partial_fit(self, X, y, *args, **kw):
        X = sp.csr_matrix(X)
        return super(SparseAutoSGDRegressor, self).partial_fit(X, y, *args, **kw)

    def decision_function(self, X, *args, **kw):
        X = sp.csr_matrix(X)
        return super(SparseAutoSGDRegressor, self).decision_function(X, *args, **kw)


class DenseAutoSGDRegressorTestCase(DenseSGDRegressorTestCase):
    """Test case for AutoSGDRegressor based on sklearn's test case.

    Need to skip warm start tests since they wont work.
    """
    factory = AutoSGDRegressor

    @pytest.mark.Skip
    def test_warm_start_optimal(self):
        pass

    @pytest.mark.Skip
    def test_warm_start_invscaling(self):
        pass

    @pytest.mark.Skip
    def test_warm_start_constant(self):
        pass

    def test_best_i(self):
        X, y = make_regression()
        # n_probe_updates > len(X)
        est = self.factory(eta0s=[10.0, 0.001], n_probe_updates=1000)
        est.fit(X, y)
        self.assertEqual(est.best_eta0_, 0.001)

        # n_probe_updates < len(X)
        est = self.factory(eta0s=[10.0, 0.001], n_probe_updates=100)
        est.fit(X, y)
        self.assertEqual(est.best_eta0_, 0.001)

    def test_init_shuffle(self):
        X, y = make_regression()
        # n_probe_updates > len(X)
        est = self.factory(eta0s=[10.0, 0.001], n_probe_updates=1000, init_shuffle=True)
        est.fit(X, y)
        self.assertEqual(est.best_eta0_, 0.001)

    def test_normalize(self):
        """Test if normalize works properly - and scaler gets only initialized once. """
        X, y = make_regression(random_state=0)
        # n_probe_updates > len(X)
        est = self.factory(eta0s=[10.0, 0.001], n_probe_updates=1000, normalize=True, random_state=1)
        est.fit(X, y)
        self.assertEqual(est.best_eta0_, 0.001)
        pred = est.predict(X)
        rmse_internal_normalize = np.sqrt(np.mean((y - pred) ** 2.0))

        # check if we are in the sparse or dense case - for sparse no mean normalization
        with_mean = True
        if self.factory is SparseAutoSGDRegressor:
            with_mean = False
        scaler = StandardScaler(with_mean=with_mean)
        X = scaler.fit_transform(X)

        est = self.factory(eta0s=[10.0, 0.001], n_probe_updates=1000, random_state=1)
        est.fit(X, y)
        self.assertEqual(est.best_eta0_, 0.001)
        pred = est.predict(X)
        rmse_external_normalize = np.sqrt(np.mean((y - pred) ** 2.0))
        np.testing.assert_almost_equal(rmse_internal_normalize, rmse_external_normalize)

    def test_partial_fit_equal_fit_optimal(self):
        """Deactivate this test because learning rate optimal is not supported here. """
        pass


class SparseAutoSGDRegressorTestCase(DenseAutoSGDRegressorTestCase):
    factory = SparseAutoSGDRegressor


class AutoSGDRegressorTest(unittest.TestCase):

    @patch('ModelingMachine.engine.tasks.sgd.AutoSGDRegressor._probe' )
    def test_numerical_instability_user_error(self, mock_probe):
        """Test if ASGD raises NumericalInstabilityError. """
        X, y = make_regression()
        mock_probe.side_effect = ValueError('floating-point')
        est = AutoSGDRegressor(eta0s=[10.0, 0.001], n_probe_updates=1000)
        self.assertRaises(NumericalInstabilityError, est.fit, X, y)


if __name__ == '__main__':
    unittest.main()
