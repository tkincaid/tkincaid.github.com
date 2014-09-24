#########################################################
#
#       Unit Test for PCA transformer
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

from ModelingMachine.engine.tasks.transformers import mmPCA

from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.container import Container

class TestmmPCA(unittest.TestCase):
    """ Test suite for mmPCA
    """

    def generate_X(self, nrows=150, ncols=100):
        colnames = [str(i) for i in xrange(ncols)]
        x = np.random.randn(nrows, ncols)
        X = Container()
        X.add(x, colnames=colnames)
        s= sp.sparse.coo_matrix( x)
        S = Container()
        S.add(s, colnames=colnames)
        return X,S

    def test_transform(self):
        """ test the transform function of the class """
        X,S = self.generate_X()
        task = mmPCA('pv=0.98')
        res = task.fit_transform(X)
        # check if Instance
        self.assertIsInstance(res,Container)

        # check if lower pctvar gives smaller matrix
        task = mmPCA('pv=0.90')
        task.fit(X)
        res2 = task.fit_transform(X)
        self.assertEqual(res().shape[1]>res2().shape[1],True)

        # check if works with sparse matrix
        task = mmPCA()
        res = task.fit_transform(S)
        # check if Instance
        self.assertIsInstance(res,Container)


    def test_pickle(self):
        """ test that the class can be pickled. This is required! """
        X,S = self.generate_X()
        task = mmPCA()
        task.fit(X)
        with tempfile.TemporaryFile(mode='w+b') as tf:
            cPickle.dump(task, tf)

if __name__ == '__main__':
    unittest.main()

