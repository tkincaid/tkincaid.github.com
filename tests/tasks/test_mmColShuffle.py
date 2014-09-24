#########################################################
#
#       Unit Test for mmColShuffle transformer
#
#       Author: Sergey Yurgenson
#
#       Copyright DataRobot, Inc. 2014
#
########################################################
import scipy as sp
import numpy as np
import pandas as pd
import unittest
import tempfile
import cPickle

from ModelingMachine.engine.tasks.transformers import mmColShuffle

from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.container import Container

from base_task_test import BaseTaskTest

class TestColShuffle(BaseTaskTest):
    """ Test suite for RankBlend
    """
    def create_sparse_and_dense_testdata(self):
        """ create some sparse and dense test data to help in the tests """
        colnames = [str(i) for i in xrange(3)]

	xd=np.array([[1,0,1],[0,2,0],[0,3,0],[4,0,1],[0,2,5],[6,3,0],[1,7,1],[0,2,8],[9,3,0]])
        Xdense = Container()
        Xdense.add(xd, colnames=colnames)

        return Xdense

    def test_transform(self):
        """ test the transform function of the class """

        X, Y, Z = self.create_bin_data()
        X = X.dataframe
        p = next(iter(Z))
        c = Container()
        colnames = X.columns.tolist()
        X = X.values
        c.add(X, colnames=colnames, **p)
        task = mmColShuffle()
        c_ = task.fit_transform(c)
        self.assertIsInstance(c_,Container)
        self.assertEqual(c_(**p).shape, X.shape)


    def test_transform_2(self):
        xd = self.create_sparse_and_dense_testdata()
        shuf = mmColShuffle()
        res = shuf.transform(xd,None,None)
        # check if instance
        self.assertIsInstance(res,Container)
        self.assertEqual(res().shape,xd().shape)

    def test_random_seed(self):
        xd = self.create_sparse_and_dense_testdata()
        shuf1 = mmColShuffle('s=20')
        res1 = shuf1.transform(xd,None,None)
        shuf2 = mmColShuffle('s=20')
        res2 = shuf2.transform(xd,None,None)
        shuf3 = mmColShuffle('s=200')
        res3 = shuf3.transform(xd,None,None)
        self.assertTrue(np.all(res1()==res2()))
        self.assertFalse(np.all(res1()==res3()))



    def test_pickle(self):
        """ test that the class can be pickled. This is required! """
        xd = self.create_sparse_and_dense_testdata()
        Z = Partition(size=xd().shape[0],folds=3,reps=1,total_size=xd.nrow)
        task = mmColShuffle()
        task.transform(xd,xd,Z)
        with tempfile.SpooledTemporaryFile(mode='w+b') as tf:
            cPickle.dump(task, tf)

if __name__ == '__main__':
    unittest.main()


