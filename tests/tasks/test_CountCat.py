#########################################################
#
#       Unit Test for CountCat Task
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

from ModelingMachine.engine.tasks.converters import CountCat
from ModelingMachine.engine.container import Container

class TestCountCatTasks(unittest.TestCase):
    """ Test suite for CountCat
    """
    def create_testdata(self):
        """ create some test data to help in the tests """
        # create a dataframe
        x=  pd.DataFrame({'A' : ['a', 'a', 'a', 'c'], 'B' : ['0', '1', '1', '4'] }, dtype=object)
        xnew=  pd.DataFrame({'A' : ['a', 'a', 'd', 'd'], 'B' : ['1', '2', '2', '4'] }, dtype=object)
        return x, xnew

    def test_transform(self):
        """ test the transform function of the class """
           #1 assumes that fit has been run and the parameters are stored in class attributes
           #2 must return a Container
        x,xnew = self.create_testdata()
        task = CountCat('cmin=3')
        task.fit(Container(x))

        res = task.transform(Container(x))
        # check if instance
        self.assertIsInstance(res,Container)
        # check if expected result
        self.assertEqual(np.all(res()==np.array([[1],[2],[2],[1]])),True)
        # check if name
        self.assertEqual(np.all(res.colnames()==['B_count']),True)

        res = task.transform(Container(xnew))
        # check if expected result
        self.assertEqual(np.all(res()==np.array([[2],[1],[1],[1]])),True)
        # check if name
        self.assertEqual(np.all(res.colnames()==['B_count']),True)

        # check if works when cmin too high
        task = CountCat('cmin=5')
        task.fit(Container(x))
        res = task.transform(Container(x))
        # check if instance
        self.assertIsInstance(res,Container)

    def test_count_cat_keeps_nans(self):
        feature = pd.Series(['a', 'a', 'b', np.nan])
        task = CountCat()
        counted = task.count_cat(feature)
        self.assertIn(np.nan, counted.keys())

    def test_pickle(self):
        """ test that the class can be pickled. This is required! """
        x,xnew = self.create_testdata()
        task = CountCat()
        task.fit(Container(x))
        with tempfile.TemporaryFile(mode='w+b') as tf:
            cPickle.dump(task, tf)

if __name__ == '__main__':
    unittest.main()
