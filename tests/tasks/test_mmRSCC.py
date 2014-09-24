#########################################################
#
#       Unit Test for mmRSCC transformer
#
#       Author: Sergey Yurgenson
#
#       Copyright DataRobot, Inc. 2013
#
########################################################
import scipy as sp
import numpy as np
import unittest
import tempfile
import cPickle

from ModelingMachine.engine.tasks.transformers import mmRSCC
from ModelingMachine.engine.container import Container

from base_task_test import BaseTaskTest


class TestmmRCCS(BaseTaskTest):
    """ Test suite for mmRSCC
    """
    def create_testdata(self):
        """ create some test data to help in the tests """
        colnames = [str(i) for i in xrange(3)]
        row=np.array([0,0,1,2]);
        col=np.array([0,2,1,1]);
        da=np.array([1,1,2,3]);
        x=sp.sparse.csc_matrix((da,(row,col)),shape=(3,3))
        X = Container()
        X.add(x, colnames=colnames)

        # one more sparse
        colnames = [str(i) for i in xrange(3)]
        row=np.array([0,0,1,2,2,2]);
        col=np.array([0,2,1,1,0,2]);
        da=np.array([1,1,2,3,4,6]);
        x=sp.sparse.csc_matrix((da,(row,col)),shape=(3,3))
        X2 = Container()
        X2.add(x, colnames=colnames)

        # one more sparse
        x=sp.sparse.csc_matrix([[1,1,1],[0,0,1]])
        Xnew = Container()
        Xnew.add(x, colnames=colnames)

        # dense
        colnames1 = [str(i) for i in xrange(5)]
	xd=np.array([[6,1,0,20,1],[7,0,2,10,0],[8,0,3,10,0]])
        Xdense = Container()
        Xdense.add(xd, colnames=colnames1)

        return X, Xnew, Xdense, X2


    def test_does_not_crash_with_sparse_data(self):
        """ test it does not crash with sparse data """
        x,xnew,xd,x2 = self.create_testdata()
        task = mmRSCC('tsh=0.99;method=spearman')
        xtrans = task.fit_transform(x)
        self.assertEqual(xtrans().shape, (3,2))

    def test_removing_zero_var(self):
        """ test it removes zero variance columns, sparse data """
        x,xnew,xd,x2 = self.create_testdata()
        task = mmRSCC('tsh=0.99')
        xtrans = task.fit_transform(xnew)
        self.assertEqual(xtrans().shape, (2,1))

    def test_transform_1(self):
        """ test the transform function of the class , dense data"""
        x,xnew,xd,x2 = self.create_testdata()
        task = mmRSCC('tsh=0.9999;method=pearson')
        task.fit(xd)
        res = task.transform(xd)
        # check if instance
        self.assertIsInstance(res,Container)
        res = task.transform(xd)
        # check if expected result
        self.assertEqual(res().shape, (3,3))
        print 'Result is \n{}'.format(res() )
        self.assertTrue(np.all(res()==np.array([[6,0,1],[7,2,0],[8,3,0]])) )

    def test_transform_2(self):
        """ test the transform function of the class , sparse data"""
        x,xnew,xd,x2 = self.create_testdata()
        task = mmRSCC('tsh=0.9999;method=pearson;absval=1')
        task.fit(x)
        res = task.transform(x)
        # check if instance
        self.assertIsInstance(res,Container)
        res = task.transform(x)
        # check if expected result
        self.assertEqual(res().shape, (3,2) )
        print 'Result is \n{}'.format(res())
        self.assertTrue(np.all(res()==np.array([[0,1],[2,0],[3,0]])) )

    def test_transform_3(self):
        """ test the transform function of the class , spase data, spase flag"""
        x,xnew,xd,x2 = self.create_testdata()
        task = mmRSCC('tsh=0.9999;method=pearson;sparse=1')
        task.fit(x)
        res = task.transform(x)
        # check if instance
        self.assertIsInstance(res,Container)
        res = task.transform(x)
        # check if expected result
        self.assertEqual(res().shape, (3,2) )
        print 'Result is \n{}'.format(res())
        self.assertTrue(np.all(res()==np.array([[0,1],[2,0],[3,0]])) )


    def test_transform_4(self):
        """ test the transform function of the class , spase data, spase flag, spearman"""
        x,xnew,xd,x2 = self.create_testdata()
        task = mmRSCC('tsh=0.9999;method=spearman;sparse=1')
        task.fit(x)
        res = task.transform(x)
        # check if instance
        self.assertIsInstance(res,Container)
        res = task.transform(x)
        # check if expected result
        self.assertEqual(res().shape, (3,2) )
        print 'Result is \n{}'.format(res())
        self.assertTrue(np.all(res()==np.array([[0,1],[2,0],[3,0]])) )

    def test_transform_5(self):
        """ test the transform function of the class , spase data, spase flag, spearman"""
        x,xnew,xd,x2 = self.create_testdata()
        task = mmRSCC('tsh=0.9999;method=spearman;sparse=1')
        task.fit(x2)
        res = task.transform(x2)
        # check if instance
        self.assertIsInstance(res,Container)
        res = task.transform(x2)
        # check if expected result
        self.assertEqual(res().shape, (3,2) )
        print 'Result is \n{}'.format(res())
        self.assertTrue(np.all(res()==np.array([[0,1],[2,0],[3,6]])) )


    def test_fit_transform_1(self):
        np.random.seed(0)
        x_rand= np.random.randn(100,5)
        x_rand= np.column_stack((x_rand, x_rand[:,4]**3))
        X_rand = Container()
        colnames = [str(i) for i in xrange(6)]
        X_rand.add(x_rand, colnames=colnames)
        task = mmRSCC('tsh=0.99;method=spearman')
        xtrans = task.fit_transform(X_rand)
        self.assertEqual(xtrans().shape, (100,5))

    def test_fit_transform_2(self):
        x_rand= np.random.randn(100,4)
        x_rand= np.column_stack((x_rand, x_rand[:,3]**3, -x_rand[:,3]**2))
        X_rand = Container()
        colnames = [str(i) for i in xrange(6)]
        X_rand.add(x_rand, colnames=colnames)
        task = mmRSCC('tsh=0.99;method=spearman;absval=1')
        xtrans = task.fit_transform(X_rand)
        self.assertEqual(xtrans().shape, (100,4) )
        task = mmRSCC('tsh=0.99;method=spearman;absval=0')
        xtrans = task.fit_transform(X_rand)
        self.assertEqual(xtrans().shape, (100,5) )

    def test_fit_transform_dyn_dense(self):
        X, Y, Z = self.create_bin_data()
        X = X.dataframe
        p = next(iter(Z))
        c = Container()
        colnames = X.columns.tolist()
        X = X.values
        X[:, 0] = 0.0  # make one col zero - this will be removed
        c.add(X, colnames=colnames, **p)
        task = mmRSCC('tsh=0.99;method=spearman')
        c_ = task.fit_transform(c)
        self.assertEqual(c_(**p).shape, (X.shape[0], X.shape[1] - 1))

    def test_fit_transform_dyn_sparse(self):
        X, Y, Z = self.create_bin_data()
        X = X.dataframe
        p = next(iter(Z))
        c = Container()
        colnames = X.columns.tolist()
        X = X.values
        X[:, 0] = 0.0  # make one col zero - this will be removed
        c.add(sp.sparse.csr_matrix(X), colnames=colnames, **p)
        task = mmRSCC('tsh=0.99;method=spearman;sparse=1')
        c_ = task.fit_transform(c)
        self.assertEqual(c_(**p).shape, (X.shape[0], X.shape[1] - 1))

    def test_fit_transform_one_col(self):
        x_rand= np.random.randn(100,1)
        X_rand = Container()
        colnames = [str(i) for i in xrange(1)]
        X_rand.add(x_rand, colnames=colnames)
        task = mmRSCC('tsh=0.99;method=spearman')
        xtrans = task.fit_transform(X_rand)
        self.assertEqual(xtrans().shape, (100,1) )


    def test_pickle(self):
        """ test that the class can be pickled. This is required! """
        x,xnew,xd,x2 = self.create_testdata()
        task = mmRSCC('tsh=0.5;method=spearman')
        task.fit(x)
        with tempfile.SpooledTemporaryFile(mode='w+b') as tf:
            cPickle.dump(task, tf)

if __name__ == '__main__':
    unittest.main()


