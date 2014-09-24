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

from ModelingMachine.engine.tasks.transformers import mmSRDT

from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.container import Container

class TestmmSRDT(unittest.TestCase):
    """ Test suite for mmSRDT
    """

    def generate_X(self):
        colnames = [str(i) for i in xrange(3)]
        x=sp.sparse.csc_matrix( [[3,2,1],[0,0,1],[0,0,0],[4,0,0]])
        X = Container()
        X.add(x, colnames=colnames)

        x=sp.sparse.csc_matrix( [[1,1,1],[0,0,1]])
        Xnew = Container()
        Xnew.add(x, colnames=colnames)
        return X, Xnew

    def test_fit(self):
        """ test the fit function of the class """
        X,Xnew = self.generate_X()
        task = mmSRDT()
        fit_result = task.fit(X)

    def test_transform(self):
        """ test the transform function of the class """
        X,Xnew = self.generate_X()
        task = mmSRDT()
        task.fit(X)
        res = task.transform(X)
        # check if Instance
        self.assertIsInstance(res,Container)
        # check if sparse
        self.assertEqual(sp.sparse.issparse(res()),True)
        # check if names
        self.assertEqual(np.all(res.colnames()==[str(i) for i in xrange(len(res.colnames()))]),True)
        # check if values as within the range expected
        self.assertEqual(np.all(res().todense().min()>=-1),True)
        self.assertEqual(np.all(res().todense().max()<=1),True)
        self.assertEqual(res()[:,1].todense().max()==1,True)
        self.assertEqual(round(res().todense()[:,0].mean(),8),0)
        # check with new data
        res = task.transform(Xnew)
        self.assertEqual(np.all(res.colnames()==[str(i) for i in xrange(len(res.colnames()))]),True)
        self.assertEqual(np.all(res().todense().min()>=-1),True)
        self.assertEqual(np.all(res().todense().max()<=1),True)
        self.assertEqual(res()[:,1].todense().max()==1,True)


    def test_pickle(self):
        """ test that the class can be pickled. This is required! """
        X,Xnew = self.generate_X()
        task = mmSRDT()
        task.fit(X)
        with tempfile.TemporaryFile(mode='w+b') as tf:
            cPickle.dump(task, tf)
