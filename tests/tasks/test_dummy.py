#########################################################
#
#       Unit Test for tasks/dummy.py
#
#       Author: Peter Prettenhofer
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

import unittest
import logging
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.dummy import DummyRegressor
from sklearn import datasets

from base_task_test import BaseTaskTest
from ModelingMachine.engine.tasks.dummy import RandomRegressor
from ModelingMachine.engine.tasks.dummy import RandomClassifier
from ModelingMachine.engine.tasks.dummy import _DummyClassifier


class TestDummy(BaseTaskTest):

    def test_dummy_arguments(self):
        """Test if arguments are passed to wrapped estimator"""
        xt = np.array([[1,2,3],[4,5,6],[7,8,9]])
        yt = np.array([0,1,0])
        self.check_arguments(RandomClassifier, DummyClassifier, xt, yt)
        self.check_arguments(RandomRegressor, DummyRegressor, xt, yt)

    def test_defaults(self):
        # we use a different default strategy for classification
        self.check_default_arguments(RandomClassifier, DummyClassifier,['random_state', 'strategy'])
        self.check_default_arguments(RandomRegressor, DummyRegressor)

    def test_dummy_clf(self):
        """Smoke test for classification. """
        X, Y, Z = self.create_bin_data()
        self.check_task("RC s=most_frequent",
                        X, Y, Z, transform=False, standardize=False)

    def test_dummy_reg(self):
        """Smoke test for regression. """
        X, Y, Z = self.create_reg_data()
        self.check_task("RR",
                        X, Y, Z, transform=False, standardize=False)

    def test_most_freq_clf_proba(self):
        X, y = datasets.make_hastie_10_2(random_state=13, n_samples=100)
        prior_pos = (y == 1).mean()
        clf = _DummyClassifier(strategy='most_frequent').fit(X, y)
        proba = clf.predict_proba(X)
        np.testing.assert_array_equal(proba[:, 1], np.ones(X.shape[0]) * prior_pos)
        np.testing.assert_array_equal(proba[:, 0], np.ones(X.shape[0]) * (1 - prior_pos))


if __name__ == '__main__':
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    unittest.main()
