#########################################################
#
#       Unit Test for tasks/filters.py
#
#       Author: Tom de Godoy
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

import unittest
import pytest
import numpy as np

from base_task_test import BaseTaskTest
from ModelingMachine.engine.tasks.rf import RFC, RFR
from ModelingMachine.engine.tasks.sgd import SGDC, SGDR

class TestFSC(BaseTaskTest):

    @pytest.mark.dscomp
    def test_01(self):
        X,Y,Z = self.create_bin_data()
        task = self.check_task('FSC',X,Y,Z,transform=True)
        self.assertIsInstance(task,RFC)

    @pytest.mark.dscomp
    def test_02(self):
        X,Y,Z = self.create_bin_data()
        task = self.check_task('FSC fs_n=6',X,Y,Z,transform=True,standardize=True)
        self.assertIsInstance(task,SGDC)
    
    @pytest.mark.dscomp
    def test_03(self):
        X,Y,Z = self.create_bin_data()
        task = self.check_task('FSC fs_s=1000',X,Y,Z,transform=True,standardize=True)
        self.assertIsInstance(task,SGDC)
        



if __name__ == '__main__':
    unittest.main()
