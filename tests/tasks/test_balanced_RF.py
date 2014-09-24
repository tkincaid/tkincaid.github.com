#########################################################
#
#       Unit Test for Balanced RF Task
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

from ModelingMachine.engine.tasks.rrf import RRFCB

from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.container import Container

class TestRRFCB(unittest.TestCase):
    '''A suite of tests to run on the RRFCB model exercise.'''

    @classmethod
    def setUpClass(cls):
        '''Any setup that should be done once for all tests within
        this suite can be done here.  Setup which should be done
        for each test can be taken care of in the
        ``setUp`` method

        '''
        random.seed(0)
        cls.model = RRFCB()
        cls.X = cls.generate_X()
        cls.Y = cls.generate_Y()
        cls.Z = cls.generate_Z()
        cls.model.fit(cls.X, cls.Y, cls.Z)

    @classmethod
    def tearDownClass(self):
        '''Any preparation that took place in setUpClass that should
        be cleaned up before exiting the test suite can be done here
        '''
        pass

    def setUp(self):
        '''Any setup that should be done before each test can be done here'''
        pass

    def tearDown(self):
        '''Any cleanup that should be done after every test case can
        be done inside this method'''
        pass

    @classmethod
    def generate_X(cls, nsamples=100, ncols=10):
        '''Generate a reasonable set of classification data'''
        colnames = [str(i) for i in xrange(ncols)]
        values = [i for i in xrange(100)]
        data = np.array([random.choice(values) for i in xrange(nsamples*ncols)]).reshape(nsamples, ncols)
        X = Container()
        X.add(data, colnames=colnames,coltypes=0)
        return X

    @classmethod
    def generate_Y(cls, nsamples=100):
        '''Generate some target variables'''
        Y = pd.Series( (np.random.randn(nsamples) > 0).astype('float') )
        Y[Y != 1] = 0
        return Y.astype('int')

    @classmethod
    def generate_Z(cls, nsamples=100):
        return Partition(nsamples, folds=5, reps=1, total_size=nsamples)

    def test_make_sampsize(self):
        '''Compute sampsize
        '''
        rows = np.ones(round(len(self.Y)/3))
        sampsize = self.model.make_sampsize(self.Y,rows,0.3)
        # check correct size
        self.assertEqual(len(sampsize)==2,True)

    def test_predict_returns_single_column(self):
        '''Predict must be a single column
        '''
        preds = self.model.predict(self.X, self.Y, self.Z)
        for part in self.Z:
            self.assertEqual(preds(**part).shape[1], 1,
                    'Prediction needs to return a column vector')
            self.assertEqual(len(preds(**part)), len(self.X(**part)),
                    'Need a prediction for each input row')
            pred_data = preds(**part)
            self.assertFalse(np.all(pred_data == 0),
                    'The predictions are currently all 0')

    def test_can_pickle(self):
        '''All models need to be serializable

        Our models need to be able to be written to disk.  In reality
        this imposes very little constraint on how you take care of your model,
        but it is good and important to check this just in case

        '''
        import cPickle
        with tempfile.SpooledTemporaryFile(1024*1024*10, mode='w+b') as tf:
            cPickle.dump(self.model, tf, cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    unittest.main()
