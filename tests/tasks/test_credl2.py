#########################################################
#
#       Unit Test for Cred1wl2 Task
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

class TestCredL2(unittest.TestCase):
    """ Unit testing for the credibility converter using L2 regularized regression
    """

    def test_credl2_fit_smoketest_binary(self):
        """ Test that the fit function works
        """
        np.random.RandomState(2525)
        Xdata = np.random.choice(np.array(['A', 'B', 'C', 'D']), size=(100, 5),
                p=[0.5, 0.3, 0.1, 0.1])
        orig_columns = ['ColA', 'ColB', 'ColC', 'ColD', 'ColE']
        Xdf = pd.DataFrame(Xdata, columns=orig_columns)
        yData = np.random.choice([0, 1], size=100, p=[0.9, 0.1])
        conv = CredL2('cmin=4;k=2')
        Z = Partition(size=100, folds=5, reps=5)
        Z.set(max_reps=1, max_folds=0)
        for Wdata in [pandas.Series(np.ones(100)), None]:
            Container_Xdf = Container(Xdf)
            if Wdata is not None:
                Container_Xdf.initialize({'weight': Wdata})
            out = conv.fit(Container_Xdf, yData, Z)
            self.assertIsInstance(out, CredL2)


    def test_credl2_fit_smoketest_regression(self):
        """ Test that the transform function functions
        """
        Xdata = np.random.choice(np.array(['A', 'B', 'C', 'D']), size=(100, 5),
                p=[0.5, 0.3, 0.1, 0.1])
        orig_columns = ['ColA', 'ColB', 'ColC', 'ColD', 'ColE']
        Xdf = pd.DataFrame(Xdata, columns=orig_columns)
        yData = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], size=100,
                p=[0.7/2.22, 0.5/2.22, 0.35/2.22, 0.25/2.22, 0.17/2.22, 0.12/2.22, 0.08/2.22, 0.05/2.22])
        conv = CredL2('cmin=4')
        Z = Partition(size=100, folds=5, reps=5)
        Z.set(max_reps=1, max_folds=0)
        for Wdata in [pandas.Series(np.ones(100)), None]:
            Container_Xdf = Container(Xdf)
            if Wdata is not None:
                Container_Xdf.initialize({'weight': Wdata})
            out = conv.fit(Container_Xdf, yData, Z)
            self.assertIsInstance(out, CredL2)

    def test_credl2_transform_smoketest_transform_binary(self):
        """ Test that the transform function works for binary data
        """
        Xdata = np.random.choice(np.array(['A', 'B', 'C', 'D']), size=(100, 5),
                p=[0.5, 0.3, 0.1, 0.1])
        orig_columns = ['ColA', 'ColB', 'ColC', 'ColD', 'ColE']
        Xdf = pd.DataFrame(Xdata, columns=orig_columns)
        yData = np.random.choice([0, 1], size=(100, 1), p=[0.9, 0.1]).ravel()
        print yData.shape
        conv = CredL2('cmin=4')
        Z = Partition(size=100, folds=5, reps=5)
        Z.set(max_reps=1, max_folds=0)
        for Wdata in [pandas.Series(np.ones(100)), None]:
            Container_Xdf = Container(Xdf)
            if Wdata is not None:
                Container_Xdf.initialize({'weight': Wdata})
            conv.fit(Container_Xdf, yData, Z)
            out = conv.transformer_stack(Container_Xdf, yData, Z)
            self.assertIsInstance(out, Container)
            for p in Z:
                self.assertEqual(out(**p).shape[0], Xdf.shape[0])
                self.assertEqual(out(**p).shape[1], len(orig_columns))
                self.assertEqual(out.colnames(**p), ['DR_cred_' + i for i in orig_columns])

    def test_credl2_smoketest_transform_regression(self):
        """ Test that the tranform functions functions for regression data
        """
        Xdata = np.random.choice(np.array(['A', 'B', 'C', 'D']), size=(100, 5),
                p=[0.5, 0.3, 0.1, 0.1])
        orig_columns = ['ColA', 'ColB', 'ColC', 'ColD', 'ColE']
        Xdf = pd.DataFrame(Xdata, columns=orig_columns)
        yData = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], size=100,
                p=[0.7/2.22, 0.5/2.22, 0.35/2.22, 0.25/2.22, 0.17/2.22, 0.12/2.22, 0.08/2.22, 0.05/2.22])
        conv = CredL2('cmin=4')
        Z = Partition(size=100, folds=5, reps=5)
        Z.set(max_reps=1, max_folds=0)
        for Wdata in [pandas.Series(np.ones(100)), None]:
            Container_Xdf = Container(Xdf)
            if Wdata is not None:
                Container_Xdf.initialize({'weight': Wdata})
            conv.fit(Container_Xdf, yData, Z)
            out = conv.transformer_stack(Container_Xdf, yData, Z)
            self.assertIsInstance(out, Container)
            for p in Z:
                self.assertEqual(out(**p).shape[0], Xdf.shape[0])
                self.assertEqual(out(**p).shape[1], len(orig_columns))
                self.assertEqual(out.colnames(**p), ['DR_cred_' + i for i in orig_columns])

    def test_credl2_transform_smoketest_transformer_stack_binary(self):
        """ Test that the transform function works for binary data
        """
        Xdata = np.random.choice(np.array(['A', 'B', 'C', 'D']), size=(100, 5),
                p=[0.5, 0.3, 0.1, 0.1])
        orig_columns = ['ColA', 'ColB', 'ColC', 'ColD', 'ColE']
        Xdf = pd.DataFrame(Xdata, columns=orig_columns)
        yData = np.random.choice([0, 1], size=(100, 1), p=[0.9, 0.1]).ravel()
        conv = CredL2('cmin=4')
        Z = Partition(size=100, folds=5, reps=5)
        Z.set(max_reps=1, max_folds=0)
        for Wdata in [pandas.Series(np.ones(100)), None]:
            Container_Xdf = Container(Xdf)
            if Wdata is not None:
                Container_Xdf.initialize({'weight': Wdata})
            conv.fit(Container_Xdf, yData, Z)
            out = conv.transformer_stack(Container_Xdf, yData, Z)
            self.assertIsInstance(out, Container)
            for p in Z:
                self.assertEqual(out(**p).shape[0], Xdf.shape[0])
                self.assertEqual(out(**p).shape[1], len(orig_columns))
                self.assertEqual(out.colnames(**p), ['DR_cred_' + i for i in orig_columns])

    def test_credl2_smoketest_transformer_stack_regression(self):
        """ Test that the tranform functions functions for regression data
        """
        Xdata = np.random.choice(np.array(['A', 'B', 'C', 'D']), size=(100, 5),
                p=[0.5, 0.3, 0.1, 0.1])
        orig_columns = ['ColA', 'ColB', 'ColC', 'ColD', 'ColE']
        Xdf = pd.DataFrame(Xdata, columns=orig_columns)
        yData = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], size=100, p
                =[0.7/2.22, 0.5/2.22, 0.35/2.22, 0.25/2.22, 0.17/2.22, 0.12/2.22, 0.08/2.22, 0.05/2.22])
        conv = CredL2('cmin=4')
        Z = Partition(size=100, folds=5, reps=5)
        Z.set(max_reps=1, max_folds=0)
        for Wdata in [pandas.Series(np.ones(100)), None]:
            Container_Xdf = Container(Xdf)
            if Wdata is not None:
                Container_Xdf.initialize({'weight': Wdata})
            conv.fit(Container_Xdf, yData, Z)
            out = conv.transformer_stack(Container_Xdf, yData, Z)
            self.assertIsInstance(out, Container)
            for p in Z:
                self.assertEqual(out(**p).shape[0], Xdf.shape[0])
                self.assertEqual(out(**p).shape[1], len(orig_columns))
                self.assertEqual(out.colnames(**p), ['DR_cred_' + i for i in orig_columns])

    def test_credl2_fit_smoketest_binary_pairwise(self):
        """ Test that the fit function works
        """
        Xdata = np.random.choice(np.array(['A', 'B', 'C', 'D']), size=(100, 5),
                p=[0.5, 0.3, 0.1, 0.1])
        orig_columns = ['ColA', 'ColB', 'ColC', 'ColD', 'ColE']
        Xdf = pd.DataFrame(Xdata, columns=orig_columns)
        yData = np.random.choice([0, 1], size=100, p=[0.9, 0.1])
        conv = CredL2('cmin=4;all=0')
        Z = Partition(size=100, folds=5, reps=5)
        Z.set(max_reps=1, max_folds=0)
        for Wdata in [pandas.Series(np.ones(100)), None]:
            Container_Xdf = Container(Xdf)
            if Wdata is not None:
                Container_Xdf.initialize({'weight': Wdata})
            out = conv.fit(Container_Xdf, yData, Z)
            self.assertIsInstance(out, CredL2)


    def test_credl2_fit_smoketest_regression_pairwise(self):
        """ Test that the transform function functions
        """
        Xdata = np.random.choice(np.array(['A', 'B', 'C', 'D']), size=(100, 5),
                p=[0.5, 0.3, 0.1, 0.1])
        orig_columns = ['ColA', 'ColB', 'ColC', 'ColD', 'ColE']
        Xdf = pd.DataFrame(Xdata, columns=orig_columns)
        yData = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], size=100,
                p=[0.7/2.22, 0.5/2.22, 0.35/2.22, 0.25/2.22, 0.17/2.22, 0.12/2.22, 0.08/2.22, 0.05/2.22])
        conv = CredL2('cmin=4;all=0')
        Z = Partition(size=100, folds=5, reps=5)
        Z.set(max_reps=1, max_folds=0)
        for Wdata in [pandas.Series(np.ones(100)), None]:
            Container_Xdf = Container(Xdf)
            if Wdata is not None:
                Container_Xdf.initialize({'weight': Wdata})
            out = conv.fit(Container_Xdf, yData, Z)
            self.assertIsInstance(out, CredL2)


    def test_credl2_transform_smoketest_transform_binary_pairwise(self):
        """ Test that the transform function works for binary data
        """
        Xdata = np.random.choice(np.array(['A', 'B', 'C', 'D']), size=(100, 5),
                p=[0.5, 0.3, 0.1, 0.1])
        orig_columns = ['ColA', 'ColB', 'ColC', 'ColD', 'ColE']
        Xdf = pd.DataFrame(Xdata, columns=orig_columns)
        yData = np.random.choice([0, 1], size=100, p=[0.9, 0.1])
        conv = CredL2('cmin=4;all=0')
        Z = Partition(size=100, folds=5, reps=5)
        Z.set(max_reps=1, max_folds=0)
        for Wdata in [pandas.Series(np.ones(100)), None]:
            Container_Xdf = Container(Xdf)
            if Wdata is not None:
                Container_Xdf.initialize({'weight': Wdata})
            conv.fit(Container_Xdf, yData, Z)
            out = conv.transform(Container_Xdf, yData, Z)
            self.assertIsInstance(out, Container)
            for p in Z:
                self.assertEqual(out(**p).shape[0], Xdf.shape[0])
                self.assertEqual(out(**p).shape[1], len(orig_columns))
                self.assertEqual(out.colnames(**p), ['DR_cred_' + i for i in orig_columns])

    def test_credl2_smoketest_transform_regression_pairwise(self):
        """ Test that the tranform functions functions for regression data
        """
        Xdata = np.random.choice(np.array(['A', 'B', 'C', 'D']), size=(100, 5),
                p=[0.5, 0.3, 0.1, 0.1])
        orig_columns = ['ColA', 'ColB', 'ColC', 'ColD', 'ColE']
        Xdf = pd.DataFrame(Xdata, columns=orig_columns)
        yData = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], size=100,
                p=[0.7/2.22, 0.5/2.22, 0.35/2.22, 0.25/2.22, 0.17/2.22, 0.12/2.22, 0.08/2.22, 0.05/2.22])
        conv = CredL2('cmin=4;all=0')
        Z = Partition(size=100, folds=5, reps=5)
        Z.set(max_reps=1, max_folds=0)
        for Wdata in [pandas.Series(np.ones(100)), None]:
            Container_Xdf = Container(Xdf)
            if Wdata is not None:
                Container_Xdf.initialize({'weight': Wdata})
            conv.fit(Container_Xdf, yData, Z)
            out = conv.transform(Container_Xdf, yData, Z)
            self.assertIsInstance(out, Container)
            for p in Z:
                self.assertEqual(out(**p).shape[0], Xdf.shape[0])
                self.assertEqual(out(**p).shape[1], len(orig_columns))
                self.assertEqual(out.colnames(**p), ['DR_cred_' + i for i in orig_columns])
            self.assertIsInstance(out, Container)

    def test_calculate_l2_credibility_all_once(self):
        """ Unit test that the calculate_l2_credibility_all_at_once behaves correctly
        """
        levels = ['A', 'B', 'C', 'D']
        Xdata = np.random.choice(np.array(levels), size=(100, 5), p=[0.5, 0.3, 0.1, 0.1])
        orig_columns = ['ColA', 'ColB', 'ColC', 'ColD', 'ColE']
        Xdf = pd.DataFrame(Xdata, columns=orig_columns)
        for Wdata in [pandas.Series(np.ones(100)), None]:
            designm, colm = calculate_l2_credibility_designmatrix(Xdf, Wdata)
            Ydata = np.random.random(size=(100,1)).ravel()
            alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
            params = {'card_max': 1000, 'small_count': 5, 'num_folds': 5, 'fit_intercept': False, 'intercept_scaling': True}
            stack_coeffs, stack_coeffs_colnames, coeffs = calculate_l2_credibility_all_at_once(Xdf, Ydata, Wdata, alpha, params)
            stack_coeffs = pd.DataFrame(create_stacked_coefficients_from_rows(stack_coeffs[0], stack_coeffs[1]),
                    columns=stack_coeffs_colnames)
            self.assertIsInstance(coeffs, dict)
            self.assertIsInstance(stack_coeffs, pd.DataFrame)
            self.assertEqual(len(coeffs), len(orig_columns))
            self.assertEqual(stack_coeffs.shape[1], designm().shape[1])

            for key in coeffs:
                self.assertTrue(key in orig_columns)
                col = coeffs[key]
                self.assertIsInstance(col, pd.DataFrame)
                self.assertTrue("coeff" in col.columns and "lev" in col.columns)
                for val in np.unique(col['lev']):
                    self.assertTrue(val in levels or val == 'small_count')

    def test_calculate_l2_credibility_pairwise(self):
        """ Unit test that the calculate_l2_credibility_pairwise behaves correctly
        """
        levels = ['A', 'B', 'C', 'D']
        Xdata = np.random.choice(np.array(levels), size=(100, 5), p=[0.5, 0.3, 0.1, 0.1])
        orig_columns = ['ColA', 'ColB', 'ColC', 'ColD', 'ColE']
        Xdf = pd.DataFrame(Xdata, columns=orig_columns)
        for Wdata in [pandas.Series(np.ones(100)), None]:
            designm, colm = calculate_l2_credibility_designmatrix(Xdf, Wdata)
            Ydata = np.random.random(size=(100,1)).ravel()
            alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
            params = {'card_max': 1000, 'small_count': 5, 'num_folds': 5, 'fit_intercept': True, 'intercept_scaling': True}
            stack_coeffs, coeffs = calculate_l2_credibility_pairwise(Xdf, Ydata, Wdata, alpha, params)
            self.assertIsInstance(coeffs, dict)
            self.assertIsInstance(stack_coeffs, pd.DataFrame)
            self.assertEqual(len(coeffs), len(orig_columns))
            self.assertEqual(stack_coeffs.shape[1], designm().shape[1])

            for key in coeffs:
                self.assertTrue(key in orig_columns)
                col = coeffs[key]
                self.assertIsInstance(col, pd.DataFrame)
                self.assertTrue("coeff" in col.columns and "lev" in col.columns)
                for val in np.unique(col['lev']):
                    self.assertTrue(val in levels or val == 'small_count')

    def test_cred_apply_l2_cat(self):
        """ Unit test that the cred apply correctly applies credibility
        """
        cmap = {}
        rstate = np.random.RandomState(2014)
        cmap['ColA'] = pd.DataFrame({"coeff": [1, 2, 3, 4], "lev": ['a', 'b', 'c', 'd']})
        cmap['ColB'] = pd.DataFrame({"coeff": [4, 3, 2, 1], "lev": ['a', 'b', 'c', 'small_count']})
        cmap['ColC'] = pd.DataFrame({"coeff": [1, 0, -1], "lev": ['a', 'b', 'c']})

        Xdata = rstate.choice(np.array(['a', 'b', 'c', 'd']), size=(100, 3),
                              p=[0.5, 0.3, 0.1, 0.1])
        orig_columns = ['ColA', 'ColB', 'ColC']
        Xdf = pd.DataFrame(Xdata, columns=orig_columns)
        out = cred_apply_l2_cat(Xdf, cmap)
        self.assertEqual(out.shape[0], Xdata.shape[0])
        self.assertEqual(out.shape[1], len(orig_columns))
        self.assertEqual(sorted(out.columns), sorted(["".join(["DR_cred_", s]) for s in orig_columns]))
        self.assertEqual(out, out.dropna())


    def test_calculate_l2_credibility_designmatrix(self):
        """ Unittest to see that calculate l2 credibility designmatrix works correctly
        """
        Xdata = np.random.choice(np.array(['A', 'B', 'C', 'D']), size=(100, 5), p=[0.5, 0.3, 0.1, 0.1])
        orig_columns = ['ColA', 'ColB', 'ColC', 'ColD', 'ColE']
        Xdf = pd.DataFrame(Xdata, columns=orig_columns)
        for Wdata in [pandas.Series(np.ones(100)), None]:
            dm, colmap = calculate_l2_credibility_designmatrix(Xdf, Wdata)
            new_columns = dm.colnames()
            self.assertIsInstance(dm, Container)
            self.assertIsInstance(colmap, dict)
            self.assertEqual(sorted(colmap.keys()), sorted(orig_columns))
            self.assertEqual(Xdata.shape[0], dm().shape[0])
            self.assertEqual(dm().shape[1], len(new_columns))
            for col in colmap:
                for ncol in colmap[col]:
                    self.assertTrue(ncol in new_columns)

    def test_calculate_coefficientMap(self):
        """ Unitest to check that it properly creates a coefficient map based on the ridge fit.
        """
        desired = {}
        coeffs = [['ColA-a', 1], ['ColA-b', 2], ['ColA-c', 3]]
        colmap =  {'ColA': ['ColA-a', 'ColA-b', 'ColA-c']}
        out = calculate_coefficientMap(coeffs, colmap)
        self.assertIsInstance(out, dict)
        self.assertTrue('ColA' in out)
        desired['ColA'] = pd.DataFrame({"coeff": [1, 2, 3], "lev": ['a', 'b', 'c']})
        self.assertEqual(out['ColA'], desired['ColA'])

if __name__=='__main__':
    unittest.main()
