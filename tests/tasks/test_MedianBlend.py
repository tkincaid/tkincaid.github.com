#########################################################
#
#       Unit Test for mmRSSC transformer
#
#       Author: Sergey Yurgenson
#
#       Copyright DataRobot, Inc. 2013
#
########################################################
import scipy as sp
import numpy as np
import pandas as pd
import unittest
import tempfile
import cPickle

from ModelingMachine.engine.tasks.blend import MedianBlend

from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.container import Container
from ModelingMachine.engine.response import Response

class TestMedianBlend(unittest.TestCase):
    """ Test suite for MedianBlend
    """
    def create_sparse_and_dense_testdata(self):
        """ create some sparse and dense test data to help in the tests """
        colnames = [str(i) for i in xrange(3)]
        row=np.array([0,0,1,2]);
        col=np.array([0,2,1,1]);
        da=np.array([1,1,2,3]);
        x=sp.sparse.csc_matrix((da,(row,col)),shape=(10,3))
        X = Container()
        X.add(x, colnames=colnames)

        x=sp.sparse.csc_matrix([[1,1,1],[0,0,1]])
        Xnew = Container()
        Xnew.add(x, colnames=colnames)

        xd=np.array([[1,0,1],[0,2,0],[0,3,0],[1,3,4],[12,3,4],[1,3,2],[13,1,2],[3,2,1],[1,3,2]])
        Xdense = Container()
        Xdense.add(xd, colnames=colnames)

        return X, Xnew, Xdense

    def test_predict(self):
        """ test the predict function of the class """
        x,xnew,xd = self.create_sparse_and_dense_testdata()
        blend = MedianBlend()
        Z = Partition(size=xd.shape[0],folds=3,reps=1,total_size=xd.nrow)
        Y = Response.from_array(np.arange(xd.shape[0]))
        blend.fit(xd, Y, Z)#second xd is just dummy variable
        res = blend.predict(xd, None, Z)#second xd is just dummy variable
        # check if instance
        self.assertIsInstance(res,Container)
        # check if expected result
        for p in Z:
            key = (p['r'],p['k'])
            self.assertEqual(res(**p).shape,(9,1))
            self.assertEqual(np.all(res(**p)==np.array([[1],[0],[0],[3],[4],[2],[2],[2],[2]])),True)

    def test_pickle(self):
        """ test that the class can be pickled. This is required! """
        x,xnew,xd = self.create_sparse_and_dense_testdata()
        Z = Partition(size=xd.shape[0],folds=3,reps=1,total_size=xd.nrow)
        Y = Response.from_array(np.arange(xd.shape[0]))
        blend = MedianBlend()
        blend.fit(xd, Y , Z)
        blend.predict(xd, None, Z)
        with tempfile.SpooledTemporaryFile(mode='w+b') as tf:
            cPickle.dump(blend, tf)

if __name__ == '__main__':
    unittest.main()


