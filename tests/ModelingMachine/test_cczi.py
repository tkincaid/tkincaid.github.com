#########################################################
#
#       Unit Test for CCZI transformer
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

from ModelingMachine.engine.tasks.transformers import mmCCZI

from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.container import Container

class TestmmCCZI(unittest.TestCase):
    """ Test suite for mmCCZI
    """
    def create_testdata(self):
        """ create some test data to help in the tests """
        # create a numpy array (matrix)
        a= np.zeros(100)
        a[2]=1
        a[5]=2
        a[10]=2
        b= np.ones(100)
        b[3]=0
        c = range(100)
        x = np.vstack((a,b,c)).T
        # create a scipy sparse matrix
        # the syntax is (data,(rows, cols)), shape=(nrows,ncols)
        s= sp.sparse.coo_matrix( ([3,2],([0,2],[1,2])),shape=(100,3))
        return x, s

    def test_fit(self):
        """ test the fit function of the class """
        x,s = self.create_testdata()

        X = Container()
        X.add(x)
        task = mmCCZI()
        task.fit(X)
        self.assertEqual(np.all(task.static_feats==[2]),True)

        task = mmCCZI('nif=2')
        task.fit(X)
        self.assertEqual(np.all(task.static_feats==[0,2]),True)

        X = Container()
        X.add(s)
        task = mmCCZI()
        task.fit(X)
        self.assertEqual(np.all(task.static_sp_feats==[]),True)

    def test_transform(self):
        """ test the transform function of the class """
        x,s = self.create_testdata()

        X = Container()
        X.add(x,colnames=['a','b','c'])
        task = mmCCZI()
        task.fit(X)
        res = task.transform(X)
        # check if Instance
        self.assertIsInstance(res,Container)
        # check if names
        self.assertEqual(res.colnames()==['c'],True)

        # check with new data
        x2=x
        x2[1,2]=100
        X = Container()
        X.add(x,colnames=['a','b','c'])
        res = task.transform(X)
        self.assertEqual(res.colnames()==['c'],True)
        self.assertEqual(res()[1,0]==100,True)


    def test_pickle(self):
        """ test that the class can be pickled. This is required! """
        x,s = self.create_testdata()
        X = Container()
        X.add(x)
        task = mmCCZI()
        task.fit(X)
        with tempfile.TemporaryFile(mode='w+b') as tf:
            cPickle.dump(task, tf)

if __name__=='__main__':
    unittest.main()
