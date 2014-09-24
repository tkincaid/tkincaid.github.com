#########################################################
#
#       Unit Test for tws.py
#
#       Author: Xavier Conort
#
#       Copyright DataRobot, Inc. 2014
#
########################################################

import unittest
import pytest
import copy
import os
import pandas
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge

from base_task_test import BaseTaskTest
from ModelingMachine.engine.vertex import Vertex
from ModelingMachine.engine.task_map import task_map
from ModelingMachine.engine.tasks.tws import TWSL2

class TestTWSL2(BaseTaskTest):

    def check_default_arguments(
            self, Task, args1_name, args2_name, Estimator1, Estimator2,
            ignore_params1=None, ignore_params2=None):
        task = Task()
        task._parse_parameters('')
        tuneparms, fixedparms = task._separate_parameters('')
        task_est = task._create_estimator(fixedparms)
        task_est_params = task_est.get_params(deep=False)

        task_est1_params = {}
        task_est2_params = {}
        for k in task_est_params:
            if args1_name in k:
                task_est1_params[k.split(': ')[1]] = task_est_params[k]
            if args2_name in k:
                task_est2_params[k.split(': ')[1]] = task_est_params[k]

        est1 = Estimator1()
        est1_params = est1.get_params()
        est2 = Estimator2()
        est2_params = est2.get_params()

        if ignore_params1 is None:
            ignore_params1 = []
        for param in ignore_params1:
            est1_params.pop(param, None)
            task_est1_params.pop(param, None)

        self.assertDictEqual(est1_params, task_est1_params)

        if ignore_params2 is None:
            ignore_params2 = []
        for param in ignore_params2:
            est2_params.pop(param, None)
            task_est2_params.pop(param, None)

        self.assertDictEqual(est2_params, task_est2_params)

    def test_defaults(self):
        self.check_default_arguments(TWSL2, 'Logistic', 'Ridge',
            LogisticRegression, Ridge, ['C','class_weight','random_state'], ['alpha','class_weight'])

    def test_arguments(self):
        xt = np.array([[1,2,3],[4,5,6],[7,8,9]])
        yt = np.array([0,1,0])
        self.check_arguments_min(TWSL2,None,xt,yt)

    @pytest.mark.skip('Somehow, is no longer reproducible')
    @pytest.mark.dscomp
    def test_reproducible(self):
        X, Y, Z = self.create_reg_data(rows=50)
        reference = [2.018756, 98.6337123, 132.04603282]
        task = self.check_task('TWSL2', X, Y, Z, transform=False,
                               reference=reference)

if __name__ == '__main__':
    unittest.main()
