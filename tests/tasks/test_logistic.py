#########################################################
#
#       Unit Test for tasks/logistic.py
#
#       Author: Tom de Godoy
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

import unittest
import pytest
import copy
import os
import pandas
import numpy as np

from sklearn.linear_model import LogisticRegression

from base_task_test import BaseTaskTest
from ModelingMachine.engine.vertex import Vertex
from ModelingMachine.engine.tasks.logistic import LogRegL1, LR1S

class TestLogistic(BaseTaskTest):

    def test_arguments(self):
        xt = np.array([[1,2,3],[4,5,6],[7,8,9]])
        yt = np.array([0,1,0])
        self.check_arguments(LogRegL1,LogisticRegression,xt,yt)
        self.check_arguments(LR1S,LogisticRegression,xt,yt)

    def test_defaults(self):
        self.check_default_arguments(LogRegL1, LogisticRegression,['C'])
        self.check_default_arguments(LR1S, LogisticRegression,['C'])

    @pytest.mark.dscomp
    def test_01(self):
        X,Y,Z = self.create_bin_data()
        task = self.check_task('LR1',X,Y,Z,transform=True)

    @pytest.mark.dscomp
    def test_02(self):
        X,Y,Z = self.create_bin_data()
        self.check_task('LR1S',X,Y,Z,transform=False)

    @pytest.mark.dscomp
    def test_modify_params_l1(self):
        X,Y,Z = self.create_bin_data()
        task = self.check_task('LR1 p=l1', X, Y, Z, transform=True)
        cgrid = task.parameters['C']
        expected_cgrid = np.logspace( 0, 4, num=50, base=10 )
        # actual cgrid is scaled by cinit - must be equal for all of them
        cinits = cgrid / expected_cgrid
        cinit = cinits[0]
        np.testing.assert_array_almost_equal(np.repeat(cinit, cinits.shape[0]), cinits)
        np.testing.assert_array_almost_equal(cgrid / cinit, expected_cgrid)
        self.assertEqual(task.parameters['dual'], False)

    def test_LR1_can_use_rate_at_10_pct(self):
        '''This is actually a test for many - ALL the modelers need to be able
        to use Rate@Top10%
        '''
        X, Y, Z = self.create_bin_data()
        task = self.check_task('LR1 t_m=Rate@Top10%', X, Y, Z, transform=True)


    def test_modify_params_l1_nonauto(self):
        X,Y,Z = self.create_bin_data()
        task = self.check_task('LR1 p=l1;C=[1.0, 0.1, 0.01]', X, Y, Z, transform=True)
        cgrid = task.parameters['C']
        expected_cgrid = np.array(sorted([1.0, 0.1, 0.01]), dtype=np.float64)
        np.testing.assert_array_almost_equal(cgrid, expected_cgrid)
        self.assertEqual(task.parameters['dual'], False)

    @pytest.mark.dscomp
    def test_modify_params_l2(self):
        X,Y,Z = self.create_bin_data()
        task = self.check_task('LR1 p=l2', X, Y, Z, transform=True)
        cgrid = task.parameters['C']
        expected_cgrid = np.logspace(-2, 2, num=50, base=10 )
        np.testing.assert_array_almost_equal(cgrid, expected_cgrid)

    def test_modify_params_l2_nonauto(self):
        X,Y,Z = self.create_bin_data()
        task = self.check_task('LR1 p=l2;C=[1.0, 0.1, 0.01]', X, Y, Z, transform=True)
        cgrid = task.parameters['C']
        expected_cgrid = np.array(sorted([1.0, 0.1, 0.01]), dtype=np.float64)
        np.testing.assert_array_almost_equal(cgrid, expected_cgrid)


if __name__ == '__main__':
    unittest.main()
