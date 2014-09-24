#########################################################
#
#       Unit Test for search transformers
#
#       Author: Xavier Conort
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

from __future__ import division
import numpy as np
import unittest
import pytest
import random
import tempfile
import cPickle
import copy

#-locals
from ModelingMachine.engine.container import Container
from ModelingMachine.engine.partition import Partition

from ModelingMachine.engine.tasks.transformers import mmSCH2W,mmSCHPOLY

class TestmmSCH(unittest.TestCase):
    """ Test suite for mmSCHPOLY and mmSCH2W
    """

    def generate_data(self,nrows=5000,ncols=4,seed=56):
        colnames = [str(i) for i in xrange(ncols)]
        np.random.seed(seed)
        x = np.random.randn(nrows, ncols)
        X = Container()
        X.add(x, colnames=colnames)
        Y=x[:,0]+x[:,1]**2+x[:,2]*x[:,1]
        Z = Partition(size=nrows,folds=1,reps=1,total_size=nrows)
        Z.set(max_reps=1,max_folds=0)
        return X,Y,Z

    @pytest.mark.dscomp
    def test_fit(self):
        """ test the fit function of the class """
        X,Y,Z = self.generate_data()

        p={'k':-1,'r':0}
        key = (p['r'],p['k'])

        task = mmSCHPOLY()
        fit_result = task.fit(X,Y,Z)

        self.assertEqual(fit_result.best_poly_df[key].shape[0]==4,True)

        task2 = mmSCH2W()
        fit_result = task2.fit(X,Y,Z)

        self.assertEqual(fit_result.best_inter_df[key].shape[0]==6,True)

    @pytest.mark.dscomp
    def test_transform(self):
        """ test the transform function of the class """
        X,Y,Z = self.generate_data()

        p={'k':-1,'r':0}

        task = mmSCHPOLY('sc2=0.5')
        res= task.fit_transform(X,Y,Z)
        # check if Instance
        self.assertIsInstance(res,Container)
        # check if expected response
        print res.colnames(**p)
        self.assertEqual(res.colnames(**p)==['0','Pol1stTerm_1','Pol2ndTerm_1'],True)
        self.assertEqual(np.all(res(**p)[:,0]-X()[:,0]==0),True)
        self.assertEqual(np.all(res(**p)[:,1]-X()[:,1]==0),True)
        self.assertEqual(np.all(res(**p)[:,2]-0.5*X()[:,1]**2==0),True)

        task = mmSCH2W('sc=0.1')
        res= task.fit_transform(X,Y,Z)
        # check if Instance
        self.assertIsInstance(res,Container)
        # check if expected response
        print res.colnames(**p)
        self.assertEqual(res.colnames(**p)==['Product_1_2'],True)
        self.assertEqual(np.all(res(**p)[:,0]-0.1*X()[:,1]*X()[:,2]==0),True)

    def test_pickle(self):
        """ test that the class can be pickled. This is required! """
        X,Y,Z = self.generate_data(nrows=200)
        task = mmSCHPOLY()
        task.fit(X,Y,Z)
        with tempfile.TemporaryFile(mode='w+b') as tf:
            cPickle.dump(task, tf)

        X,Y,Z = self.generate_data(nrows=200)
        task = mmSCH2W()
        task.fit(X,Y,Z)
        with tempfile.TemporaryFile(mode='w+b') as tf:
            cPickle.dump(task, tf)
