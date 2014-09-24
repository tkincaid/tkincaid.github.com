#########################################################
#
#       Unit Test for enet_reg_bm.py
#
#       Author: Xavier Conort
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

from sklearn.linear_model import ElasticNet

from base_task_test import BaseTaskTest
from ModelingMachine.engine.vertex import Vertex
from ModelingMachine.engine.task_map import task_map
from ModelingMachine.engine.tasks.enet_reg_bm import Enet1

class TestEnet1(BaseTaskTest):

    @pytest.mark.dscomp
    def test_arguments(self):
        xt = np.array([[1,2,3],[4,5,6],[7,8,9]])
        yt = np.array([0,1,3])
        self.check_arguments(Enet1,ElasticNet,xt,yt)

    def test_defaults(self):
        self.check_default_arguments(Enet1, ElasticNet,['alpha'])

    @pytest.mark.dscomp
    def test_01(self):
        X,Y,Z = self.create_bin_data()
        task = self.check_task('ENR2',X,Y,Z,transform=True)

if __name__ == '__main__':
    unittest.main()
