#########################################################
#
#       Unit Test for new ridit transformer
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

from ModelingMachine.engine.tasks.Rridit import mmRDTR

from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.container import Container

class TestmmRDTR(unittest.TestCase):
    """ Test suite for mmRDTR
    """
    def create_testdata(self):
        """ create some test data to help in the tests """
        # create a numpy array (matrix)
        x= np.array([[1,2,3],[4,5,6],[7,8,9]])
        x_rand= np.random.randn(100,5)
        # create a scipy sparse matrix
        # the syntax is (data,(rows, cols)), shape=(nrows,ncols)
        s= sp.sparse.coo_matrix( ([3,2],([0,2],[1,6])),shape=(3,10))
        return x, x_rand, s

    def generate_X(self, nrows=500, ncols=10):
        colnames = [str(i) for i in xrange(ncols)]
        x = np.random.randn(nrows, ncols)
        X = Container()
        X.add(x, colnames=colnames)
        return X

    def test_check_sparse(self):
        """ test the check_sparse helper function of the class """
        x, x_rand, s = self.create_testdata()
        task = mmRDTR()
        #check that a dense array x is passed thru unchanged
        check = task.check_sparse(x)
        self.assertEqual(np.all(check==x),True)
        #check that a sparse matrix s is converted to a numpy array
        check = task.check_sparse(s)
        self.assertIsInstance(check,np.ndarray)
        self.assertEqual(np.all(check==s.todense()),True)

    def test_apply_ridit(self):
        """ test the apply_ridit helper function of the class """
        x, x_rand, s = self.create_testdata()
        task = mmRDTR()
        s = task.check_sparse(s)

        for data in [x, x_rand, s]:
            #get ecdf for data
            Fn=task.make_ECDF(data)
            out=task.apply_ridit( data, Fn )
            # check same dimension
            self.assertEqual(np.all(data.shape==out.shape),True)
            # check mean almost equal to 0
            for i in range(data.shape[1]):
                self.assertEqual(round(out[:,i].mean(),8),0)
            # check if works with newdata
            newdata=data[:2]
            out=task.apply_ridit( newdata, Fn )
            # check same dimension
            self.assertEqual(np.all(newdata.shape==out.shape),True)

    def test_fit(self):
        """ test the fit function of the class """
        X = self.generate_X()
        task = mmRDTR()
        fit_result = task.fit(X)

    def test_transform(self):
        """ test the transform function of the class """
        X = self.generate_X()
        task = mmRDTR()
        task.fit(X)
        res = task.transform(X)
        # check if Instance
        self.assertIsInstance(res,Container)
        # check if names
        self.assertEqual(np.all(res.colnames()==[str(i) for i in xrange(len(res.colnames()))]),True)
        # check if values as within the range expected
        self.assertEqual(np.all(res().min()>=-1),True)
        self.assertEqual(np.all(res().max()<=1),True)
        for i in range(len(res.colnames())):
            self.assertEqual(round(res()[:,i].mean(),8),0)
        # check with new data
        Y = self.generate_X()
        res = task.transform(Y)
        self.assertEqual(np.all(res.colnames()==[str(i) for i in xrange(len(res.colnames()))]),True)
        self.assertEqual(np.all(res().min()>=-1),True)
        self.assertEqual(np.all(res().max()<=1),True)


    def test_pickle(self):
        """ test that the class can be pickled. This is required! """
        X = self.generate_X()
        task = mmRDTR()
        task.fit(X)
        with tempfile.TemporaryFile(mode='w+b') as tf:
            cPickle.dump(task, tf)
