#########################################################
#
#       Unit Test for mmPOLY Task
#
#       Author: Xavier Conort
#
#       Copyright DataRobot, Inc. 2013
#
########################################################


from __future__ import division
import numpy as np
import scipy as sp
import pandas
import unittest
import random
import tempfile
import cPickle
import logging

from ModelingMachine.engine.tasks.transformers import mmPOLY


from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.container import Container

class TestmmPOLY(unittest.TestCase):
    """ Test suite for mmPOLY
    """

    def generate_X(self, nrows=500, ncols=10, seed=None):
        if seed:
            np.random.seed(seed)
        colnames = [str(i) for i in xrange(ncols)]
        x = np.random.randn(nrows, ncols)
        x[x[:,9]>0,9]=1
        x[x[:,9]<=0,9]=0
        X = Container()
        X.add(x, colnames=colnames)
        return X

    def test_transform_default(self):
        X = self.generate_X(seed=1)

        task = mmPOLY()
        task.fit(X)
        res = task.transform(X)
        # check if right dimension
        self.assertEqual(res().shape[1]==19,True)
        # check if right value
        self.assertEqual(np.all((res()[:,:9])**2==res()[:,10:19]),True)

        # check if right name
        self.assertEqual(res.colnames()[-1]=='8_order_2',True)

    def test_transform_degree4_and_scales(self):
        X = self.generate_X(seed=1)
        task = mmPOLY('dg=4')
        task.fit(X)
        res = task.transform(X)
        # check if right dimension
        self.assertEqual(res().shape[1]==4*9+1,True)

        task = mmPOLY('dg=4;sc1=2;sc2=2;sc3=2;sc4=2')
        task.fit(X)
        res2 = task.transform(X)
        # check if right value
        self.assertEqual(np.all(res2()==2*res()),True)

    def test_pickle(self):
        """ test that the class can be pickled. This is required! """
        X = self.generate_X()
        task = mmPOLY()
        task.fit(X,Y=None,Z=None)
        with tempfile.TemporaryFile(mode='w+b') as tf:
            cPickle.dump(task, tf)

if __name__ == '__main__':
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    unittest.main()
