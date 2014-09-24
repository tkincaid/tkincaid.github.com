#########################################################
#
#       Unit Test for tasks/cart.py
#
#       Author: Peter Prettenhofer
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

import unittest
import pytest
import logging
import numpy as np
from tesla.tree import DecisionTreeClassifier, DecisionTreeRegressor

from base_task_test import BaseTaskTest
from ModelingMachine.engine.tasks.cart import CARTClassifier
from ModelingMachine.engine.tasks.cart import CARTRegressor


class TestCART(BaseTaskTest):

    def test_cart_arguments(self):
        """Test if arguments are passed to wrapped estimator"""
        xt = np.arange(63).reshape(21,3)
        yt = np.arange(21) % 2
        self.check_arguments(CARTClassifier, DecisionTreeClassifier, xt, yt)
        self.check_arguments(CARTRegressor, DecisionTreeRegressor, xt, yt)

    def test_defaults(self):
        self.check_default_arguments(CARTClassifier, DecisionTreeClassifier,['random_state','min_samples_split'])
        self.check_default_arguments(CARTRegressor, DecisionTreeRegressor,['random_state','min_samples_split'])

    @pytest.mark.dscomp
    def test_cart_clf(self):
        """Smoke test for classification. """
        X, Y, Z = self.create_bin_data()
        self.check_task("DTC md=[8,100];t_f=0.2;t_m=AUC",
                        X, Y, Z, transform=True, standardize=False)

    @pytest.mark.dscomp
    def test_cart_reg(self):
        """Smoke test for regression. """
        X, Y, Z = self.create_reg_data()
        self.check_task("DTR md=[8,100];t_f=0.2;t_m=RMSE",
                        X, Y, Z, transform=True, standardize=False)

    @pytest.mark.dscomp
    def test_cart_reg_mss_tune(self):
        """Test _modify_paramters. """
        X, Y, Z = self.create_reg_data()
        t = self.check_task("DTR md=8;ss=auto",
                            X, Y, Z, transform=True, standardize=False)
        ssgrid = [10, 20, 40, 60, 80, 100, 125, 150, 175, 200, 250, 300]
        n_samples = X.dataframe.shape[0] * 0.8  # Somewhere, the 200 gets trimmed down
                                      # probably for CV
        self.assertEqual(t.parameters['min_samples_split'],
                         [mss for mss in ssgrid if mss < 0.50 * n_samples])


if __name__ == '__main__':
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    unittest.main()
