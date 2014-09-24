#########################################################
#
#       Unit Test for mmGAUS2 Task
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
import logging

from ModelingMachine.engine.tasks.transformers import mmRDT2,mmGAUS2
from scipy.stats import shapiro, norm


from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.container import Container

class TestmmGAUS2(unittest.TestCase):
    """ Test suite for mmRDT2
    """
    def create_testdata(self):
        """ create some test data to help in the tests """
        # create a numpy array (matrix)
        x= np.array([[1,2,3],[4,5,6],[7,8,9]])
        x_rand= np.random.randn(100,5)*100000
        # create a scipy sparse matrix
        # the syntax is (data,(rows, cols)), shape=(nrows,ncols)
        s= sp.sparse.coo_matrix( ([3,2],([0,2],[1,6])),shape=(3,10))
        return x, x_rand, s

    def generate_X(self, nrows=500, ncols=10, seed=None):
        if seed:
            np.random.seed(seed)
        colnames = [str(i) for i in xrange(ncols)]
        x = np.random.randn(nrows, ncols)
        X = Container()
        X.add(x, colnames=colnames)
        return X

    def test_transform_after_RIDIT(self):
        """ test the transform function of the class """
        X = self.generate_X()
        task = mmRDT2()
        task.fit(X)
        res = task.transform(X)
        task2 = mmGAUS2('ri=1')
        res2 = task2.transform(res)
        # check if Instance
        self.assertIsInstance(res2,Container)
        # check if names
        self.assertEqual(np.all(res2.colnames()==[str(i) for i in xrange(len(res.colnames()))]),True)
        # check if values as within the range expected
        for i in range(len(res2.colnames())):
            self.assertEqual(round(res2()[:,i].mean(),8),0)

        #test unfriendly case
        xnew=np.array([[-9999999,-9999999,-9999999,-9999999,-9999999,
                        -9999999,-9999999,-9999999,-9999999,-9999999]])

        Xnew = Container()
        Xnew.add(xnew)
        res = task.transform(Xnew)
        res2 = task2.transform(res)
        self.assertEqual(np.all(res2()<norm.ppf(0.001)),True)


    def test_transform_implementation(self):
        '''Should be the same whether we use parallelism or not'''
        X = self.generate_X(seed=1)
        task = mmGAUS2()
        task.fit(X)
        res = task.transform(X)

        # Here we compare against some hard-coded values
        v = res()

        self.assertAlmostEqual(v[0,0], norm.ppf((0.914+1)/2), places=5)
        self.assertAlmostEqual(v[0,-1], norm.ppf((-0.222+1)/2), places=5)
        self.assertAlmostEqual(v[-1,0], norm.ppf((-0.43+1)/2), places=5)
        self.assertAlmostEqual(v[-1,-1], norm.ppf((0.974+1)/2), places=5)

    def test_pickle(self):
        """ test that the class can be pickled. This is required! """
        X = self.generate_X()
        task = mmGAUS2()
        task.fit(X,Y=None,Z=None)
        with tempfile.TemporaryFile(mode='w+b') as tf:
            cPickle.dump(task, tf)

if __name__ == '__main__':
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    unittest.main()
