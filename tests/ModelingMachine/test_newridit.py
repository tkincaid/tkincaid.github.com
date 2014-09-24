
from __future__ import division
import numpy as np
import scipy as sp
import pandas as pd
import unittest
import random
import tempfile
import cPickle

from ModelingMachine.engine.tasks.transformers import mmRDT2, multi_apply_ridit_one


from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.container import Container

class TestmmRDT2(unittest.TestCase):
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

    def test_check_sparse(self):
        """ test the check_sparse helper function of the class """
        x, x_rand, s = self.create_testdata()
        task = mmRDT2()
        #check that a dense array x is passed thru unchanged
        check = task._check_sparse(x)
        self.assertEqual(np.all(check==x),True)
        #check that a sparse matrix s is converted to a numpy array
        check = task._check_sparse(s)
        self.assertIsInstance(check,np.ndarray)
        self.assertEqual(np.all(check==s.todense()),True)

    def test_apply_ridit(self):
        """ test the apply_ridit helper function of the class """
        x, x_rand, s = self.create_testdata()
        task = mmRDT2()
        s = task._check_sparse(s)

        for data in [x, x_rand, s]:
            #get ecdf for data
            Fn=task.make_ECDF(data)
            col_args = [(ecdf,) for ecdf in Fn]
            # n_jobs=-2 says all but one core
            out=task.apply_in_parallel(data, multi_apply_ridit_one, col_args=col_args, n_jobs=-2)
            # check same dimension
            self.assertEqual(np.all(data.shape==out.shape),True)
            # check mean almost equal to 0
            for i in range(data.shape[1]):
                self.assertEqual(round(out[:,i].mean(),8),0)
            # check if works with newdata
            newdata=data[:2]
            out=task.apply_in_parallel(newdata, multi_apply_ridit_one, col_args=col_args, n_jobs=-2)
            # check same dimension
            self.assertEqual(np.all(newdata.shape==out.shape),True)

    def test_fit(self):
        """ test the fit function of the class """
        X = self.generate_X()
        task = mmRDT2()
        fit_result = task.fit(X)

    def test_transform(self):
        """ test the transform function of the class """
        X = self.generate_X()
        task = mmRDT2()
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

    def test_extreme_case(self):

        X = self.generate_X()
        task = mmRDT2()
        task.fit(X)
        res = task.transform(X)
        #test unfriendly case
        xnew=np.array([[-9999999,-9999999,-9999999,-9999999,-9999999,
                        -9999999,-9999999,-9999999,-9999999,-9999999]])
        Xnew = Container()
        Xnew.add(xnew)
        res = task.transform(Xnew)
        print res()
        self.assertEqual(np.all(res().max()<-0.99),True)


    def test_transform_implementation(self):
        '''Should be the same whether we use parallelism or not'''
        X = self.generate_X(seed=1)
        task = mmRDT2()
        task.fit(X)
        res = task.transform(X)

        # Here we compare against some hard-coded values
        v = res()

        self.assertAlmostEqual(v[0,0], 0.914)
        self.assertAlmostEqual(v[0,-1], -0.222)
        self.assertAlmostEqual(v[-1,0], -0.43)
        self.assertAlmostEqual(v[-1,-1], 0.974)

    def test_pickle(self):
        """ test that the class can be pickled. This is required! """
        X = self.generate_X()
        task = mmRDT2()
        task.fit(X)
        with tempfile.TemporaryFile(mode='w+b') as tf:
            cPickle.dump(task, tf)


if __name__=='__main__':
    unittest.main()
