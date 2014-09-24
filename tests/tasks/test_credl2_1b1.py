#########################################################
#
#       Unit Test for CredL2_1b1 Task
#
#       Author: # Mark Steadman
#
#       Copyright DataRobot, Inc. 2014
#
########################################################

from __future__ import division
import numpy as np
import scipy as sp
import pandas as pd
import unittest
import pytest
import random
import tempfile
import cPickle

from ModelingMachine.engine.tasks.cred_converters import *

from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.container import Container

class TestCredL2_1b1(unittest.TestCase):
    """ Unit testing for the credibility converter using L2 regularized regression
    """

    def test_CredL2_1b1_transform_smoketest_transformer_stack_binary(self):
        """ Test that the transform function works for binary data
        """
        np.random.seed(2525)
        Xdata = np.random.choice(np.array(['A', 'B', 'C', 'D']), size=(100, 5),
                p=[0.5, 0.3, 0.1, 0.1])
        orig_columns = ['ColA', 'ColB', 'ColC', 'ColD', 'ColE']
        Xdf = pd.DataFrame(Xdata, columns=orig_columns)
        Container_Xdf = Container(Xdf)
        np.random.seed(2525)
        yData = np.random.choice([0, 1], size=(100, 1), p=[0.9, 0.1]).ravel()
        conv = CredL2_1b1('rgl_C=[0.01,0.1];cmin=4;i2w=1')
        Z = Partition(size=100, folds=5, reps=5)
        Z.set(max_reps=1, max_folds=0)
        for weight in [False, True]:
            if weight:
                Container_Xdf.initialize({'weight': pandas.Series(np.ones(100))})
            out = conv.fit(Container_Xdf, yData, Z)
            self.assertIsInstance(out, CredL2_1b1)

            out = conv.transform(Container_Xdf, yData, Z)
            self.assertIsInstance(out, Container)
            p = {'k': -1, 'r': 0}
            self.assertEqual(out(**p).shape[0], Xdf.shape[0])
            self.assertEqual(out(**p).shape[1], 15)
            self.assertEqual(out.colnames(**p), ['DR_cred_' + i for i in orig_columns] +
                [ 'DR_cred_ColA_XX_ColB', 'DR_cred_ColA_XX_ColC', 'DR_cred_ColA_XX_ColD', 'DR_cred_ColA_XX_ColE', 'DR_cred_ColB_XX_ColC', 'DR_cred_ColB_XX_ColD', 'DR_cred_ColB_XX_ColE', 'DR_cred_ColC_XX_ColD', 'DR_cred_ColC_XX_ColE', 'DR_cred_ColD_XX_ColE'])

            out = conv.transformer_stack(Container_Xdf, yData, Z)
            self.assertIsInstance(out, Container)
            for p in Z:
                self.assertEqual(out(**p).shape[0], Xdf.shape[0])
                self.assertEqual(out(**p).shape[1], 15)
                self.assertEqual(out.colnames(**p), ['DR_cred_' + i for i in orig_columns] +
                [ 'DR_cred_ColA_XX_ColB', 'DR_cred_ColA_XX_ColC', 'DR_cred_ColA_XX_ColD', 'DR_cred_ColA_XX_ColE', 'DR_cred_ColB_XX_ColC', 'DR_cred_ColB_XX_ColD', 'DR_cred_ColB_XX_ColE', 'DR_cred_ColC_XX_ColD', 'DR_cred_ColC_XX_ColE', 'DR_cred_ColD_XX_ColE'])
            if weight:
                self.assertEqual(np.all(out.get('weight') == np.ones(100)), True)


    def test_CredL2_1b1_smoketest_transformer_stack_regression(self):
        """ Test that the tranform functions functions for regression data
        """
        np.random.seed(2525)
        Xdata = np.random.choice(np.array(['A', 'B', 'C', 'D']), size=(100, 5),
                p=[0.5, 0.3, 0.1, 0.1])
        orig_columns = ['ColA', 'ColB', 'ColC', 'ColD', 'ColE']
        Xdf = pd.DataFrame(Xdata, columns=orig_columns)
        Container_Xdf = Container(Xdf)
        np.random.seed(2525)
        yData = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], size=100, p
                =[0.7/2.22, 0.5/2.22, 0.35/2.22, 0.25/2.22, 0.17/2.22, 0.12/2.22, 0.08/2.22, 0.05/2.22])
        conv = CredL2_1b1('rgl_a=[0.01,0.1];cmin=4')
        Z = Partition(size=100, folds=5, reps=5)
        Z.set(max_reps=1, max_folds=0)
        for weight in [False, True]:
            if weight:
                Container_Xdf.initialize({'weight': pandas.Series(np.ones(100))})
            Container_Xdf.initialize({'weight': pandas.Series(np.ones(100))})
            out = conv.fit(Container_Xdf, yData, Z)
            self.assertIsInstance(out, CredL2_1b1)

            out = conv.transform(Container_Xdf, yData, Z)
            self.assertIsInstance(out, Container)
            for p in Z:
                self.assertEqual(out(**p).shape[0], Xdf.shape[0])
                self.assertEqual(out(**p).shape[1], len(orig_columns))
                self.assertEqual(out.colnames(**p), ['DR_cred_' + i for i in orig_columns])

            out = conv.transformer_stack(Container_Xdf, yData, Z)
            self.assertIsInstance(out, Container)
            for p in Z:
                self.assertEqual(out(**p).shape[0], Xdf.shape[0])
                self.assertEqual(out(**p).shape[1], len(orig_columns))
                self.assertEqual(out.colnames(**p), ['DR_cred_' + i for i in orig_columns])
            if weight:
                self.assertEqual(np.all(out.get('weight') == np.ones(100)), True)

if __name__ == '__main__':
    unittest.main()
