#########################################################
#
#       Unit Test for CountCatw Task
#
#       Author: Xavier Conort
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

from __future__ import division
import numpy as np
import scipy as sp
import pandas as pd
import unittest
import random
import tempfile
import cPickle

from ModelingMachine.engine.tasks.converters import CountCatw
from ModelingMachine.engine.container import Container

class TestCountCatwTasks(unittest.TestCase):
    """ Test suite for CountCatw
    """
    def create_testdata(self):
        """ create some test data to help in the tests """
        # create a dataframe
        x=  pd.DataFrame({'A' : ['a', 'a', 'a', 'c'], 'B' : ['0', '1', '1', '4'], 'C' : ['0', '0', '0', '2'] }, dtype=object)
        xnew=  pd.DataFrame({'A' : ['a', 'a'], 'B' : ['1', '2'], 'C' : ['0', '0'] }, dtype=object)
        return x, xnew

    def test_transform(self):
        """ test the transform function of the class """
        x,xnew = self.create_testdata()
        task = CountCatw('dm=1')
        task.fit(Container(x))

        res = task.transform(Container(x))
        # check if instance
        self.assertIsInstance(res,Container)
        # check if expected result
        self.assertEqual(np.all(res()==np.array([[3,1],[3,2],[3,2],[1,1]])),True)
        # check if name
        self.assertEqual(np.all(res.colnames()==['count_A','count_B']),True)

        res = task.transform(Container(xnew))
        # check if expected result
        print res()
        self.assertEqual(np.all(res()==np.array([[3,2],[3,1]])),True)
        # check if name

        task = CountCatw('dm=2')
        task.fit(Container(x))
        res = task.transform(Container(xnew))
        # check if expected result
        self.assertEqual(np.all(res()==np.array([[3,2],[3,1]])),True)
        # check if name
        self.assertEqual(np.all(res.colnames()==['count_A','count_B']),True)

    def test_one_cat_only(self):
        x=  pd.DataFrame({  'A' : ['a', 'a', 'a', 'c']},dtype=object)
        task = CountCatw('dm=2')
        task.fit(Container(x))
        res = task.transform(Container(x))
        # check if name
        self.assertEqual(res().shape[1]==1,True)

    def test_univariate_keeps_nans_as_categories(self):
        features = pd.DataFrame({'x': pd.Series(['a', 'a', 'b', np.nan])})
        task = CountCatw('dm=1')
        counted = task.fit(Container(features))
        out = task.transform(Container(features))
        check = out().flatten()
        reference = np.asarray([2, 2, 1, 1])
        np.testing.assert_equal(check, reference)

    def test_bivariate_keeps_nans_as_categories(self):
        feature1 = pd.Series(['a', 'a', 'a', np.nan])
        feature2 = pd.Series(['b', np.nan, 'b', np.nan])
        features = pd.DataFrame({'x1': feature1, 'x2': feature2})
        task = CountCatw('dm=2')
        task.fit(Container(features))
        out = task.transform(Container(features))
        check = out()
        reference = np.asarray([[3, 2, 2],
                                [3, 2, 1],
                                [3, 2, 2],
                                [1, 2, 1]])
        np.testing.assert_equal(check, reference)

    def test_pickle(self):
        """ test that the class can be pickled. This is required! """
        x,xnew = self.create_testdata()
        task = CountCatw()
        task.fit(Container(x))
        with tempfile.TemporaryFile(mode='w+b') as tf:
            cPickle.dump(task, tf)

if __name__=='__main__':
    unittest.main()
