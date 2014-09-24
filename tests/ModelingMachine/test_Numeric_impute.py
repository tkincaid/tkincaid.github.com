#########################################################
#
#       Unit Test for Numeric_impute_arbitrary Task
#
#       Author: Sergey Yurgenson
#
#       Copyright DataRobot, Inc. 2014
#
########################################################

import unittest
import pandas as pd
import numpy as np
import os
import sys
tests_dir = os.path.dirname(os.path.abspath(__file__) )
modeling_machine_dir = os.path.join(tests_dir, '../..')
sys.path.append(modeling_machine_dir)
from  ModelingMachine.engine.tasks.converters import Numeric_impute_arbitrary
from ModelingMachine.engine.tasks.converters import Numeric_impute
from ModelingMachine.engine.container import Container

class TestNumericImputeArbitrary(unittest.TestCase):
    def setUp(self):
        self.c = pd.DataFrame(data=np.array([
                [ 1, 2, 11,  float('NaN') ],
                [ 2, 3, 17,  18  ],
                [ 3, 2, float('NaN'), 16 ],
                [ 4, 1, 13,  14 ],
                [ 20, 1, 45, 46 ]
            ]),columns=['a','b','c','d'])
        self.correct1 = np.array([
            [ 1, 2, 11, -9999 ],
            [ 2, 3, 17, 18 ],
            [ 3, 2, -9999, 16 ],
            [ 4, 1, 13, 14 ],
            [ 20, 1, 45, 46 ]
        ])
        self.correct2 = np.array([
            [ 1, 2, 11, 100 ],
            [ 2, 3, 17, 18 ],
            [ 3, 2, 100, 16 ],
            [ 4, 1, 13, 14 ],
            [ 20, 1, 45, 46 ]
        ])

    def test_transform_arbitrary_imputation(self):
        nip = Numeric_impute_arbitrary()
        X = nip.fit_transform(Container(self.c))
        print 'Result is \n{}'.format(X())
        self.assertTrue(np.all(X() - self.correct1 < 0.0001))

        nip = Numeric_impute_arbitrary('m=100')
        X = nip.fit_transform(Container(self.c))
        print 'Result is \n{}'.format(X())
        self.assertTrue(np.all(X() - self.correct2 < 0.0001))

class TestNumericImpute(unittest.TestCase):

    def test_transform_imputation_with_object_type(self):
        self.c = pd.DataFrame(data=np.array([
                [ 1, 2, 11,  float('NaN') ],
                [ 2, 3, 17,  18  ],
                [ 3, 2, float('NaN'), 16 ],
                [ 4, 1, 13,  14 ],
                [ 20, 1, 45, 46 ]
            ]).astype('object'),columns=['a','b','c','d'])
        nip = Numeric_impute()
        X = nip.fit_transform(Container(self.c))
        # passes if no error, should probably assert something

    def test_transform_missing_test_only(self):
        train = pd.DataFrame(data=np.array([
                [ 1, 2, 11,  11 ],
                [ 2, 3, 17,  18  ],
                [ 3, 2, 14, 16 ],
                [ 4, 1, 13,  14 ],
                [ 20, 1, 45, 46 ]], dtype=np.float))
        test = pd.DataFrame(data=np.array([
                [ 1, 2, 11,  np.nan ],
                [ 2, 3, 17,  np.nan  ],
                [ 3, 2, np.nan, np.nan ],
                [ 4, 1, 13,  np.nan ],
                [ 20, 1, 45, np.nan ]], dtype=np.float))
        print(test)
        nip = Numeric_impute()
        nip.fit(Container(train))
        np.testing.assert_array_equal(nip.nan_count_, np.zeros(4))
        out = nip.transform(Container(test))
        print(out())
        self.assertTrue(np.all(np.isfinite(out())))

if __name__ == '__main__':
    unittest.main()
