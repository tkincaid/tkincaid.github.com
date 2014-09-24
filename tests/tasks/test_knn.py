#########################################################
#
#       Unit Test for tasks/knn.py
#
#       Author: Xavier Conort
#
#       Copyright DataRobot, Inc. 2014
#
########################################################

import unittest
import pytest
import logging
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from base_task_test import BaseTaskTest
from ModelingMachine.engine.tasks.knn import KNNC, KNNR

class TestKNN(BaseTaskTest):

    @pytest.mark.dscomp
    def test_arguments(self):
        xt = np.array([[1,2,3],[4,5,6],[7,8,9]])
        yt = np.array([0,1,0])
        self.check_arguments(KNNC, KNeighborsClassifier, xt,yt)
        self.check_arguments(KNNR, KNeighborsRegressor, xt,yt)

    def test_defaults(self):
        self.check_default_arguments(KNNC, KNeighborsClassifier,['n_neighbors'])
        self.check_default_arguments(KNNR, KNeighborsRegressor,['n_neighbors'])

    @pytest.mark.dscomp
    def test_01(self):
        X,Y,Z = self.create_bin_data()
        self.check_task('KNNC',X,Y,Z,transform=False,standardize=False)

    @pytest.mark.dscomp
    def test_02(self):
        X,Y,Z = self.create_reg_data()
        self.check_task('KNNR',X,Y,Z,transform=False,standardize=False)

    def test_03(self):
        X,Y,Z = self.create_reg_data()
        self.check_task('KNNR k=[500,600]',X,Y,Z,transform=False,standardize=False)

    def test_03b(self):
        X,Y,Z = self.create_reg_data()
        self.check_task('KNNR k=[50]',X,Y,Z,transform=False,standardize=False)

    @pytest.mark.dscomp
    def test_04(self):
        X,Y,Z = self.create_reg_data()
        self.check_task('KNNR m=0',X,Y,Z,transform=False,standardize=False)

    @pytest.mark.dscomp
    def test_05(self):
        X,Y,Z = self.create_reg_data()
        self.check_task('KNNR m=1',X,Y,Z,transform=False,standardize=False)

    @pytest.mark.dscomp
    def test_06(self):
        X,Y,Z = self.create_reg_data()
        self.check_task('KNNR m=2',X,Y,Z,transform=False,standardize=False)

    @pytest.mark.dscomp
    def test_07(self):
        X,Y,Z = self.create_reg_data()
        t = self.check_task('KNNR k=2;m=minkowski',X, Y, Z, transform=False, standardize=False)
        self.assertEqual(t.parameters['algorithm'], 'ball_tree')

if __name__ == '__main__':
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    unittest.main()
