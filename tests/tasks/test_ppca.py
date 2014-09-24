#########################################################
#
#       Unit Test for partial PCA transformer
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

from ModelingMachine.engine.tasks.transformers import mmPPCA

from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.container import Container

class TestmmPPCA(unittest.TestCase):
    """ Test suite for mmPPCA
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

    def test_fit(self):
        """ test the fit function of the class """
        X,S = self.generate_X()
        task = mmPPCA('k=3000')
        fit_result = task.fit(X)
        print(fit_result.vt.shape[0])
        self.assertEqual(fit_result.vt.shape[1]==80,True)
        task = mmPPCA('k=5')
        fit_result = task.fit(S)
        self.assertEqual(fit_result.vt.shape[1]==5,True)
        task = mmPPCA('km=1000')
        fit_result = task.fit(S)
        self.assertEqual(fit_result.vt.shape[1]==10,True)

    def test_transform(self):
        """ test the transform function of the class """
        X,S = self.generate_X()
        task = mmPPCA('k=auto')
        task.fit(X)
        res = task.transform(X)
        # check if Instance
        self.assertIsInstance(res,Container)
        # check if values as within the range expected
        self.assertEqual(res().shape[1]==10,True)

    def test_pickle(self):
        """ test that the class can be pickled. This is required! """
        X,S = self.generate_X()
        task = mmPPCA()
        task.fit(X)
        with tempfile.TemporaryFile(mode='w+b') as tf:
            cPickle.dump(task, tf)
