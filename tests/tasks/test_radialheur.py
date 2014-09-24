#########################################################
#
#       Unit Test for tasks/svc.py, get_gamma_heuristic
#
#       Author: Xavier Conort
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

import unittest
import pytest
import logging
import numpy as np
import pandas as pd

from scipy import sparse as sp

from base_task_test import BaseTaskTest
from ModelingMachine.engine.tasks.svc import _gamma_heuristic as get_gamma_heuristic_python


def get_gamma_heuristic_r(xt, yt, frac=0.5, random_state=13):
    """Heuristic to compute good default for gamma for RBF kernel.

    Uses the ``sigest`` function in R's kernlab package.
    This will create two samples of ``xt`` and compute the reciprocal of the
    median sq. distances of a sample of ``xt``.
    """
    from rpy2 import robjects
    from ModelingMachine.engine.data_connector import convert_to_r_dataframe

    robjects.globalenv['xt'] = convert_to_r_dataframe(pd.DataFrame(xt))
    robjects.globalenv['yt'] = robjects.FloatVector(yt.tolist())
    yvals = pd.Series(yt).unique()
    classification = len(yvals) == 2 and set(yvals) == set([0,1])
    if classification:
        rcode = """
        set.seed(%d)
        library(kernlab); out=sigest(y~., data=data.frame(y=as.factor(yt),xt), scale=FALSE);
        """ % int(random_state)
    else:
        rcode = """
        set.seed(%d)
        library(kernlab); out=sigest(y~., data=data.frame(y=yt,xt), scale=FALSE);
        """ % int(random_state)
    robjects.r(rcode)
    return robjects.globalenv['out'][1]


class TestRBFHeur(BaseTaskTest):
    """A test case for the RBF gamma heuristic in ``SVMC`` and ``SVMR``. """

    def test_01(self):
        X, Y, Z = self.create_bin_data()
        X = X.dataframe
        res = get_gamma_heuristic_python(X.values, Y)
        self.assertEqual(res > 0, True)

    def test_02(self):
        X,Y,Z = self.create_reg_data()
        X = X.dataframe
        res = get_gamma_heuristic_python(X.values, Y)
        self.assertEqual(res > 0,True)

    @pytest.mark.dscomp
    def test_get_gamma_py(self):
        X,Y,Z = self.create_reg_data()
        X = X.dataframe
        for i in range(5):
            res_r = get_gamma_heuristic_r(X.values, Y, random_state=i)
            res_py = get_gamma_heuristic_python(X.values, Y, random_state=i)
            np.testing.assert_almost_equal(res_py, res_r, decimal=1)

    def test_rbf_heuristic_sparse(self):
        X, Y, Z = self.create_reg_data()
        X = X.dataframe.values
        for i in range(5):
            res_dense = get_gamma_heuristic_python(X, Y, random_state=i)
            res_sparse = get_gamma_heuristic_python(sp.csr_matrix(X), Y, random_state=i)
            np.testing.assert_almost_equal(res_dense, res_sparse)

            res_sparse = get_gamma_heuristic_python(sp.csc_matrix(X), Y, random_state=i)
            np.testing.assert_almost_equal(res_dense, res_sparse)

            res_sparse = get_gamma_heuristic_python(sp.coo_matrix(X), Y, random_state=i)
            np.testing.assert_almost_equal(res_dense, res_sparse)

    def test_rbf_heuristic_inf(self):
        X = np.array([[0.5, 0.25],
                      [0.7, 0.35],
                      [0.8, 0.45]])
        Y = np.array([1, 1, -1])
        res = get_gamma_heuristic_python(X, Y, random_state=0)

        X_inf = np.vstack((X, np.array([[np.inf, 0.35]])))
        Y_inf = np.array([1, 1, -1, 1])
        res_inf = get_gamma_heuristic_python(X, Y, random_state=0)
        self.assertEqual(res, res_inf)

        X_inf = np.vstack((X, np.array([[np.nan, 0.35]])))
        Y_inf = np.array([1, 1, -1, 1])
        res_inf = get_gamma_heuristic_python(X, Y, random_state=0)
        self.assertEqual(res, res_inf)

        self.assertRaises(ValueError, get_gamma_heuristic_python, sp.csr_matrix(X_inf), Y_inf,
                          random_state=0)


if __name__ == '__main__':
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    unittest.main()
