#########################################################
#
#       Unit Test for Numeric_impute_predict Task
#
#       Author: Glen Koundry
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

import unittest
import pandas
import numpy as np
import os
import sys
tests_dir = os.path.dirname(os.path.abspath(__file__) )
modeling_machine_dir = os.path.join(tests_dir, '../..')
sys.path.append(modeling_machine_dir)
import ModelingMachine.engine.tasks.transformers
from ModelingMachine.engine.container import Container

class TestNumericImputePredict(unittest.TestCase):
    def setUp(self):
        # c = 1 + 2 * a + 4 * b
        # d = 2 + 2 * a + 4 * b
        self.c = Container()
        self.c.add(
            np.array([
                [ 1, 2, 11,  float('NaN') ],
                [ 2, 3, 17,  18  ],
                [ 3, 2, float('NaN'), 16 ],
                [ 4, 1, 13,  14 ],
                [ 20, 1, 45, 46 ]
            ]), colnames=['a','b','c','d'])
        self.correct = np.array([
            [ 0, 1, 1, 2, 11, 12 ],
            [ 0, 0, 2, 3, 17, 18 ],
            [ 1, 0, 3, 2, 15, 16 ],
            [ 0, 0, 4, 1, 13, 14 ],
            [ 0, 0, 20, 1, 45, 46 ]
        ])

    def test_predict(self):
        nip = ModelingMachine.engine.tasks.transformers.Numeric_impute_predict()
        X = nip.fit_transform(self.c,None,None)
        self.assertTrue(np.all(X() - self.correct < 0.0001))
        self.assertListEqual(X.colnames(),['c-pi','d-pi','a','b','c','d'])

if __name__ == '__main__':
    unittest.main()
