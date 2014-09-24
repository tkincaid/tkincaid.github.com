#########################################################
#
#       Unit Test for Gaussian Transform
#
#       Author: Glen Koundry
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

import unittest
import numpy as np
import os
import sys
tests_dir = os.path.dirname(os.path.abspath(__file__) )
modeling_machine_dir = os.path.join(tests_dir, '../..')
sys.path.append(modeling_machine_dir)
import ModelingMachine.engine.tasks.transformers
from ModelingMachine.engine.container import Container

class TestGaussianTransformer(unittest.TestCase):
    def setUp(self):
        self.data = Container()
        self.data.add( np.array([ 1, 2, 3, 4, 5]))
        self.data.add( np.array([ 32, 16, 8, 4, 2]))

    def test_predict(self):
        gt = ModelingMachine.engine.tasks.transformers.mmGAUS()
        X = gt.fit_transform(self.data,None,None)
        # make sure the order of all columns remains the same
        self.assertTrue(np.all(np.argsort(X(),axis=0)==np.argsort(self.data(),axis=0)))

if __name__ == '__main__':
    unittest.main()
