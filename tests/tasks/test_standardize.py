#########################################################
#
#       Unit Test for tasks/standardize.py
#
#       Author: Tom de Godoy
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

import unittest
import numpy as np
import scipy.sparse as sp
import pytest

from base_task_test import BaseTaskTest
from ModelingMachine.engine.tasks.standardize import Standardize
from ModelingMachine.engine.vertex import Vertex
from ModelingMachine.engine.container import Container
from ModelingMachine.engine.partition import Partition


class TestStd(BaseTaskTest):
    """Unit tests to test the ST "standardize" task
    """

    def test_01(self):
        """Test that ST produces mean zero, variance unity on binary data
        """
        X, Y, Z = self.create_bin_data()
        tasks = ['NI', 'ST']
        vertex = Vertex(tasks, 'id')
        out = vertex.fit_transform(X, Y, Z)

        m = out().mean(0)
        v = out().var(0)
        np.testing.assert_array_almost_equal(m, np.zeros(len(m)))
        np.testing.assert_allclose(v, np.ones(len(m)))

    @pytest.mark.unit
    def test_ST_friedman_data_check_task(self):
        """Test that it is successfully works as a task on Friedman data
        """
        X, Y, Z = self.create_reg_syn_data()
        self.check_task('ST', X, Y, Z, predict=False)

    @pytest.mark.unit
    def test_ST_friedman_data_check_fit_transform_mean_zero(self):
        """Test that all the means are zeros
        """
        X, Y, Z = self.create_reg_syn_data()
        model = Standardize()
        x = X.dataframe.values
        X = Container()
        X.add(x)
        model.fit(X, Y, Z)
        out = model.transform(X, Y, Z)
        means = out().mean(axis=0)
        self.assertEqual(len(means), X().shape[1])
        np.testing.assert_array_almost_equal(means, np.zeros(len(means)))

    @pytest.mark.unit
    def test_ST_friedman_data_check_fit_transform_var_one(self):
        """Test that all the variances are unity
        """
        X, Y, Z = self.create_reg_syn_data()
        model = Standardize()
        x = X.dataframe.values
        X = Container()
        X.add(x)
        Z.set
        model.fit(X, Y, Z)
        out = model.transform(X, Y, Z)
        var = out().var(axis=0)
        self.assertEqual(len(var), X().shape[1])
        np.testing.assert_allclose(var, np.ones(len(var)))

    @pytest.mark.unit
    def test_SparseStandardize_CSR(self):
        """Test that it returns sparse for sparse data in CSR format
        """
        x = sp.rand(500, 500, format='csr')
        X = Container()
        X.add(x)
        self.assertTrue(sp.issparse(X()))
        y = np.random.randint(0, 1, 50)
        Z = Partition(size=x.shape[0], folds=5, reps=5,total_size=x.shape[0])
        Z.set(max_reps=5, max_folds=0)
        model = Standardize()
        res = model.fit(X, y, Z)
        out = res.transform(X, y, Z)
        self.assertTrue(sp.issparse(out()))
        self.assertTrue(sp.issparse(out(r=1, k=0)))

    @pytest.mark.unit
    def test_SparseStandardize_coo(self):
        """Test that it returns sparse for sparse data in COO format
        """
        x = sp.rand(500, 500, format='coo')
        X = Container()
        X.add(x)
        self.assertTrue(sp.issparse(X()))
        y = np.random.randint(0, 1, 50)
        Z = Partition(size=x.shape[0], folds=5, reps=5,total_size=x.shape[0])
        Z.set(max_reps=5, max_folds=0)
        model = Standardize()
        res = model.fit(X, y, Z)
        out = res.transform(X, y, Z)
        self.assertTrue(sp.issparse(out()))
        self.assertTrue(sp.issparse(out(r=1, k=0)))

    @pytest.mark.unit
    def test_SparseStandardize_CSR_force_mean(self):
        """Test that it returns sparse for sparse data in CSR format
        """
        x = sp.rand(500, 500, format='csr')
        X = Container()
        X.add(x)
        self.assertTrue(sp.issparse(X()))
        y = np.random.randint(0, 1, 50)
        Z = Partition(size=x.shape[0], folds=5, reps=5,total_size=x.shape[0])
        Z.set(max_reps=5, max_folds=0)
        model = Standardize('fm=1')
        res = model.fit(X, y, Z)
        out = res.transform(X, y, Z)
        self.assertFalse(sp.issparse(out()))
        self.assertFalse(sp.issparse(out(r=1, k=0)))

    @pytest.mark.unit
    def test_SparseStandardize_CSR_variance_unity_or_zero(self):
        """Test that it returns sparse for sparse data in CSR format
        """
        x = sp.rand(500, 500, format='csr')
        X = Container()
        X.add(x)
        self.assertTrue(sp.issparse(X()))
        y = np.random.randint(0, 1, 50)
        Z = Partition(size=x.shape[0], folds=5, reps=5,total_size=x.shape[0])
        Z.set(max_reps=5, max_folds=0)
        model = Standardize()
        res = model.fit(X, y, Z)
        out = res.transform(X, y, Z)
        var = out().toarray().var(axis=0)
        self.assertEqual(len(var), X().shape[1])
        np.testing.assert_array_almost_equal(var[var > 0], np.ones(len(var[np.abs(var) > 0])))

    @pytest.mark.unit
    def test_SparseStandardize_CSR_force_mean_mean_is_zero(self):
        """Test that it returns sparse for sparse data in CSR format
        """
        x = sp.rand(500, 500, format='csr')
        X = Container()
        X.add(x)
        self.assertTrue(sp.issparse(X()))
        y = np.random.randint(0, 1, 50)
        Z = Partition(size=x.shape[0], folds=5, reps=5,total_size=x.shape[0])
        Z.set(max_reps=5, max_folds=0)
        model = Standardize('fm=1')
        res = model.fit(X, y, Z)
        out = res.transform(X, y, Z)
        means = out().mean(axis=0)
        self.assertEqual(len(means), X().shape[1])
        np.testing.assert_array_almost_equal(means, np.zeros(len(means)))

if __name__ == '__main__':
    unittest.main()
