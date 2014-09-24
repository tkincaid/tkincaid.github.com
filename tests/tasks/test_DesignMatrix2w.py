#########################################################
#
#       Unit Test for DesignMatrix2w Task
#
#       Author: Xavier Conort
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

from __future__ import division
import numpy as np
import pandas as pd
import unittest
import tempfile
import cPickle

from ModelingMachine.engine.tasks.converters import DesignMatrix2w
from ModelingMachine.engine.tasks.converters import GroupCat

from ModelingMachine.engine.container import Container

class TestDesignMatrix2wTasks(unittest.TestCase):
    """ Test suite for DesignMatrix2w
    """
    def create_testdata(self):
        """ create some test data to help in the tests """
        # create a dataframe
        x=  pd.DataFrame({'A' : ['a', 'a', 'a', 'c', 'e', 'a', 'a',],
            'B' : ['0', '0', '1', '4', '1', '1', '4'],
            'C' : ['0', '0', '1', '4', '1', '1', '4'] }, dtype=object)
        xnew=  pd.DataFrame({'A' : ['a'], 'B' : ['1'], 'C' : ['0'] }, dtype=object)
        return x, xnew

    def test_transform(self):
        """ test the transform function of the class """
        x,xnew = self.create_testdata()
        task = DesignMatrix2w('sc=3;mdf=2')

        task.fit(Container(x))
        res = task.transform(Container(x))
        # check if instance
        self.assertIsInstance(res,Container)
        # check if expected result
        self.assertEqual(res.colnames()[0]=='A_A-a-a',True)
        # check if correct dimension
        self.assertEqual(res().shape[1]==9,True)

        res = task.transform(Container(xnew))
        # check if expected result
        self.assertEqual(res().shape==(1,9),True)
        self.assertEqual(np.all(res().todense()==np.array([[1,0,1,0,1,0,1,1,1]])),True)

    def test_does_not_crash_with_IDs(self):
        '''When every single entry in a column was different (think ID fields)
        it was causing problems by generating empty containers'''
        x = pd.DataFrame({'A' : [str(i) for i in xrange(25)]})
        task = DesignMatrix2w('sc=2;mdf=1')
        xtrans = task.fit_transform(Container(x))
        self.assertEqual(xtrans().shape==(25,1),True)

    def test_pickle(self):
        """ test that the class can be pickled. This is required! """
        x,xnew = self.create_testdata()
        task = DesignMatrix2w('sc=2;mdf=1')
        task.fit(Container(x))
        with tempfile.TemporaryFile(mode='w+b') as tf:
            cPickle.dump(task, tf)

if __name__ == '__main__':
    unittest.main()
