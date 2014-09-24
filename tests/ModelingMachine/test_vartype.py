#########################################################
#
#       Unit Test for var types
#
#       Author: Tom de Godoy
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

import unittest
import numpy as np
import pandas
import os
import copy
import random

from ModelingMachine.engine.vartypes import check_var_type, identify_binary_column_indices

class TestUtilities(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        here = os.path.dirname(os.path.abspath(__file__))

        self.ds1 = pandas.read_csv(os.path.join(here,'../testdata/credit-sample-200.csv'))
        self.ds2 = pandas.read_csv(os.path.join(here,'../testdata/allstate-nonzero-200.csv'))
        self.ds3 = pandas.read_csv(os.path.join(here,'../testdata/credit-train-small.csv'))
        self.ds4 = pandas.read_csv(os.path.join(here,'../testdata/amazon-sample-1000.csv'))
        self.ds5 = pandas.read_csv(os.path.join(here,'../testdata/amazon-sample-9999.csv'))

        self.target1 = 'SeriousDlqin2yrs'
        self.target2 = 'Claim_Amount'
        self.target3 = 'SeriousDlqin2yrs'
        self.target4 = 'ACTION'
        self.target5 = 'ACTION'

    def test_check_type(self):
        out = check_var_type( self.ds5, self.target5)
        print self.ds5
        print out
        self.assertEqual(out, 'NCCNCCCCCC')
        out = check_var_type( self.ds3, self.target3)
        print self.ds3
        print out
        self.assertEqual(out, 'NNNNNNNNNNNX')

    def test_identify_binary_cols_smoke(self):
        df = pandas.DataFrame({'target':np.random.randn(10),
                               'bin':[random.choice([0,1]) for i in xrange(10)],
                               'random':np.arange(10)}, columns=['bin','target','random'])
        ixs = identify_binary_column_indices(df, 'target')
        self.assertEqual(ixs, [0])

    def test_identify_binary_cols_smoke_other_order(self):
        df = pandas.DataFrame({'target':np.random.randn(10),
                               'bin':[random.choice([0,1]) for i in xrange(10)],
                               'random':np.arange(10)}, columns=['random','target','bin'])
        ixs = identify_binary_column_indices(df, 'target')
        self.assertEqual(ixs, [2])

    def test_identify_binary_cols_not_take_strings(self):
        df = pandas.DataFrame({'target':np.random.randn(10),
                               'bin':[random.choice(['A','B']) for i in xrange(10)],
                               'random':np.arange(10)}, columns=['random','target','bin'])
        ixs = identify_binary_column_indices(df, 'target')
        self.assertEqual(ixs, [])

    def test_identify_binary_cols_must_be_0_1(self):
        df = pandas.DataFrame({'target':np.random.randn(10),
                               'bin':[random.choice([1,2]) for i in xrange(10)],
                               'random':np.arange(10)}, columns=['random','target','bin'])
        ixs = identify_binary_column_indices(df, 'target')
        self.assertEqual(ixs, [])


if __name__ == '__main__':
    unittest.main()
