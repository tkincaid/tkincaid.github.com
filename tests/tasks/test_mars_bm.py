#########################################################
#
#       Unit Test for tasks/rmars_bm.py
#
#       Author: Tom de Godoy/Glen Koundry/Mark Steadman
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

import unittest
import logging
import numpy as np
import pytest

from sklearn.datasets import make_friedman1
from sklearn.datasets import make_hastie_10_2
from sklearn.cross_validation import train_test_split

from base_task_test import BaseTaskTest
from ModelingMachine.engine.tasks.rmars_bm import MARSR
from ModelingMachine.engine.tasks.rmars_bm import MARSC
from ModelingMachine.engine.tasks.rmars_bm import MarsEstimatorC
from ModelingMachine.engine.tasks.rmars_bm import MarsEstimatorR
from ModelingMachine.engine.tasks.rmars_bm import MarsTransformer
from ModelingMachine.engine.tasks.rmars_bm import create_term_names
from ModelingMachine.engine.tasks.mmpyearth import PyEarthTransformer
from ModelingMachine.engine.vertex import Vertex
from ModelingMachine.engine.container import Container

class TestMARS(BaseTaskTest):
    """Unit tests for the tasks in rmars_bm.py

    This implements the Multiplicative Adaptive Regression Splines
    using the earth package in R
    """
    @pytest.mark.unit
    def test_arguments(self):
        """Test that the MARS tasks pass check_arguments"""
        xt_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        yt_data = np.array([0, 1, 0])
        self.check_arguments(MARSR, MarsEstimatorR, xt_data, yt_data)
        self.check_arguments(MARSC, MarsEstimatorC, xt_data, yt_data)

    @pytest.mark.unit
    def test_check_mars_classifier(self):
        """ Smoke test for MARS classifier"""
        X, Y, Z = self.create_bin_data()
        self.check_task('MARSC2', X, Y, Z, transform=False, standardize=True)

    @pytest.mark.unit
    def test_check_mars_regressor_gaussian(self):
        """ Smoke test for base MARS regressor"""
        X, Y, Z = self.create_reg_data()
        self.check_task('MARSR2', X, Y, Z, transform=False, standardize=True)

    @pytest.mark.unit
    def test_check_mars_regressor_poission(self):
        """ Smoke test for base MARS regressor poisson loss function"""
        X, Y, Z = self.create_reg_data()
        self.check_task('MARSP2', X, Y, Z, transform=False, standardize=True)

    @pytest.mark.unit
    def test_check_mars_regressor_gamma(self):
        """ Smoke test for MARS regressor gamma loss function"""
        X, Y, Z = self.create_reg_data()
        self.check_task('MARSA2', X, Y, Z, transform=False, standardize=True)

    @pytest.mark.dscomp
    def test_mars_classification_accuracy(self):
        """Test classification.
        Check if predict_proba generates probablities for classification.
        """
        hastieX, hastieY = make_hastie_10_2(n_samples=1000)
        X_train, X_test, y_train, y_test = train_test_split(hastieX,
                                                            hastieY,
                                                            test_size=0.1,
                                                            random_state=13)

        n_samples, n_features = X_train.shape
        mrs = MarsEstimatorC(**{'degree': 10, 'random_state': 1234})
        mrs.set_params(**{'degree': 2})
        parms = mrs.get_params(deep=False)
        self.assertEqual(parms['degree'], 2)
        parmsDeep = mrs.get_params(deep=True)
        self.assertEqual(parms['degree'], 2)
        mrs._getSeed()
        mrs.fit(X_train, y_train)
        probs = mrs.predict_proba(X_test)
        self.assertEqual(probs.shape[1], 2)
        probs = probs[:, 1]
        pred = mrs.predict(X_test)
        np.testing.assert_array_equal(np.unique(y_test), np.unique(pred))
        acc = np.mean(y_test == pred)
        self.assertGreater(acc, 0.8)
        self.assertTrue(np.all(probs <= 1.00) and np.all(probs >= 0.00))
        self.assertEqual(probs.shape[0], X_test.shape[0])
        pred = (2 * (probs >= 0.5).astype(np.int) - 1).astype(np.float)
        acc = np.mean(y_test == pred.flatten())
        self.assertGreater(acc, 0.8)

    @pytest.mark.unit
    def test_mars_regression_accuracy(self):
        """Test regression"""
        X, y = make_friedman1(n_samples=1000)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=13)
        est = MarsEstimatorR(**{'degree': 2, 'random_state': 1234})
        est.fit(X_train, y_train)
        pred = est.predict(X_test)
        mse = np.mean((pred - y_test) ** 2.0)
        self.assertLess(mse, 0.2)

    @pytest.mark.unit
    def test_MARS_transformer_smoke_test(self):
        """Test that the MARS transformer works
        """
        X, Y, Z, = self.create_reg_syn_data()
        X = X.dataframe
        Xcont = Container()
        Xcont.add(X.values)
        est = MarsTransformer('dg=1;nk=21')
        est.fit(Xcont, Y, Z)
        out = est.transform(Xcont, Z=Z)
        # Test that the output nrows match input
        # And that the number of output columns is reasonable
        self.assertLess(out().shape[1], 30)
        self.assertEqual(out().shape[0], X.shape[0])
        for p in Z:
            self.assertEqual(out(**p).shape[0], X.shape[0])
            self.assertLess(out(**p).shape[1], 30)
    
    @pytest.mark.unit
    def test_MARS_transformer_regression_data_check_task(self):
        """Test that the mars transfomer passes check_task
        """
        X, Y, Z = self.create_reg_data()
        self.check_transformer('MARST', X, Y, Z)

    @pytest.mark.unit
    def test_MARS_transformer_classification_data_check_task(self):
        """Test that the mars transfomer passes check_task
        """
        X, Y, Z = self.create_bin_data()
        self.check_transformer('MARST dg=2;nk=6', X, Y, Z)

    @pytest.mark.unit
    def test_mars_transformer_make_mars(self):
        """Test that the mars transformer make_mars task is behaving correctly
        """
        X, Y, Z = self.create_bin_data()
        X = X.dataframe
        xarray = X.values
        xarray[np.isnan(xarray)] = 0
        est = MarsTransformer('dg=2;tr=5;nk=21')
        est._modify_parameters(X, Y)
        outmatrix, outcolnames, outearth = est.make_mars(xarray, Y, X.columns)
        self.assertIsInstance(outmatrix, np.ndarray)
        self.assertIsInstance(outcolnames, list)
        self.assertEqual(outmatrix.shape[1], len(outcolnames))
    
    def test_mars_transformer_create_term_names(self):
        """ Test that the mars transformer create_term_names is able to
        create a list of term names based on a list of cuts and dirs
        """
        cut_list = [ [], [1, 5], [3, 3], [6, 6, 4]]
        dir_list = [ [], [-1, 2], [2, -1], [2, 3, 2]]
        colnames = ["col0", "col1", "col2", "col3", "col4"]
        out = create_term_names(cut_list, dir_list, colnames)
        desired = ["ASR-Intercept", "ASR-h(1 - col1)h(col2 - 5)", "ASR-h(col2 - 3)h(3 - col1)",
                   "ASR-h(col2 - 6)h(col3 - 6)h(col2 - 4)"]
        self.assertEqual(out, desired)

    def check_transformer(self, taskname, X, Y, Z):
        """Helper method to check transformer task
        """
        tasks = ['NI','ST']
        vertex = Vertex(tasks, 'id')
        X = vertex.fit_transform(X, Y, Z)
        vertex = Vertex([taskname], 'id')
        #fit and predict
        vertex.fit(X, Y, Z)
        task, xfunc, yfunc = vertex.steps[-1]
        #transform
        logging.debug(' ----- testing %s  transform -----'%taskname)
        out= vertex.transform(X, Y, Z)
        self.check_transform(out, X)
        return task

    @pytest.mark.unit
    def test_pyearth_transformer_smoke_test(self):
        """Test that the pyearth transformer works
        """
        X, Y, Z, = self.create_reg_syn_data()
        X = X.dataframe
        Xcont = Container()
        Xcont.add(X.values)
        est = PyEarthTransformer('md=1;mt=21')
        est.fit(Xcont, Y, Z)
        out = est.transform(Xcont, Z=Z)
        # Test that the output nrows match input
        # And that the number of output columns is reasonable
        self.assertLess(out().shape[1], 30)
        self.assertEqual(out().shape[0], X.shape[0])
        for p in Z:
            self.assertEqual(out(**p).shape[0], X.shape[0])
            self.assertLess(out(**p).shape[1], 30)

if __name__ == '__main__':
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    unittest.main()
