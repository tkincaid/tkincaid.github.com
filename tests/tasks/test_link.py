#########################################################
#
#       Unit Test for mmLINK Task
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
import logging

from ModelingMachine.engine.tasks.transformers import mmLINK


from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.container import Container


class TestmmLINK(unittest.TestCase):
    """ Test suite for mmLINK
    """

    def generate_XYZ(self, dynamic_only=False):
        """ create some test data to help in the tests """
        nsamples=10
        A_pattern = [0.1,0.2,0.3,0.5]
        B_pattern = [0.1,0.2,0.8]
        x=  pd.DataFrame({
            'A' : [random.sample(A_pattern,1)[0] for i in range(nsamples)],
            'B' : [random.sample(B_pattern,1)[0] for i in range(nsamples)]},
            dtype=float)

        colnames = [i for i in x.columns]
        Z = Partition(nsamples,folds=2,reps=0,total_size=nsamples)
        X = Container()
        if dynamic_only:
            for p in Z:
                X.add(x.values, colnames=colnames, **p)
        else:
            X.add(x.values, colnames=colnames)
        Y_pattern = [0,1,1]
        Y = pd.Series( np.array([random.sample(Y_pattern,1)[0] for i in range(nsamples)]).astype('int') )

        return X,Y,Z

    def generate_sparse(self):
        """ create some test data to help in the tests """
        X, Y, Z = self.generate_XYZ()
        new_X = Container()
        static = X()
        new_X.add(sp.sparse.csc_matrix(static), colnames=X.colnames())
        return new_X, Y, Z

    def test_transform_smoke(self):
        """ test the transform function of the class """

        # identity
        X,Y,Z = self.generate_XYZ()
        task = mmLINK()
        task.fit(X,Y,Z)
        res = task.transform(X,Y,Z)

        # check if Instance
        self.assertIsInstance(res,Container)
        # check if names
        self.assertEqual(np.all(res.colnames()==['A','B']),True)
        # check if values as within the range expected
        self.assertEqual(np.all(np.around(res(**Z[0]),decimals=3)==np.around(X(**Z[0]),decimals=3)),True)

    def test_transform_logit(self):
        # logit
        X,Y,Z = self.generate_XYZ()
        task = mmLINK('l=1')
        task.fit(X,Y,Z)
        res = task.transform(X,Y,Z)

        # check if Instance
        self.assertIsInstance(res,Container)
        # check if names
        self.assertEqual(np.all(res.colnames()==['logit_A','logit_B']),True)
        # check if values as within the range expected
        self.assertEqual(np.all(np.around(res(**Z[0]),decimals=3)==np.around(np.log(X(**Z[0]))-np.log(1-X(**Z[0])),decimals=3)),True)

    def test_transform_log(self):
        # log
        X,Y,Z = self.generate_XYZ()
        task = mmLINK('l=2')
        task.fit(X,Y,Z)
        res = task.transform(X,Y,Z)

        # check if Instance
        self.assertIsInstance(res,Container)
        # check if names
        self.assertEqual(np.all(res.colnames()==['log_A','log_B']),True)
        # check if values as within the range expected
        self.assertEqual(np.all(np.around(res(**Z[0]),decimals=3)==np.around(np.log(X(**Z[0])),decimals=3)),True)

    def test_transform_logp1(self):
        # log
        X,Y,Z = self.generate_XYZ()
        task = mmLINK('l=3')
        task.fit(X,Y,Z)
        res = task.transform(X,Y,Z)

        # check if Instance
        self.assertIsInstance(res,Container)
        # check if names
        self.assertEqual(np.all(res.colnames()==['logp1_A','logp1_B']),True)
        # check if values as within the range expected
        self.assertEqual(np.all(np.around(res(**Z[0]),decimals=3)==np.around(np.log(1+X(**Z[0])),decimals=3)),True)

    def test_transform_log_scale(self):
        # scale
        X,Y,Z = self.generate_XYZ()
        task = mmLINK('l=2;sc=1')
        task.fit(X,Y,Z)
        res = task.transform(X,Y,Z)
        # check if Instance
        self.assertIsInstance(res,Container)
        # check if names
        self.assertEqual(np.all(res.colnames()==['std_log_A','std_log_B']),True)
        # check if values as within the range expected
        self.assertEqual(np.all(np.around(res(**Z[0]).var(axis=0),decimals=3)==1),True)

    def test_transform_scale2(self):
        # scale
        X,Y,Z = self.generate_XYZ()
        task = mmLINK('sc=2')
        task.fit(X,Y,Z)
        res = task.transform(X,Y,Z)
        # check if values as within the range expected
        self.assertEqual(np.all(np.around(res(**Z[0]).var(axis=0)-1/3,decimals=3)==0),True)

    def test_transform_sparse_scale(self):
        # try with sparse
        X, Y, Z = self.generate_sparse()
        task = mmLINK('sc=1')
        task.fit(X, Y, Z)
        res = task.transform(X, Y, Z)
        # check if Instance
        self.assertIsInstance(res, Container)
        # check if values as within the range expected
        self.assertEqual(np.all(np.around(res(**Z[0]).var(axis=0)-1,decimals=3)==0),True)

    def test_transform_log_scale_dynamic(self):
        # scale
        X, Y, Z = self.generate_XYZ(dynamic_only=True)
        task = mmLINK('l=2;sc=1')
        task.fit(X,Y,Z)
        res = task.transform(X,Y,Z)
        # check if Instance
        self.assertIsInstance(res, Container)
        # check if names
        p = next(iter(res))
        self.assertEqual(res.colnames(dynamic_only=True, **p), ['std_log_A','std_log_B'])
        # check if values as within the range expected
        res_var = res(**Z[0]).var(axis=0)
        np.testing.assert_array_almost_equal(res_var, np.ones_like(res_var))

    def test_pickle(self):
        """ test that the class can be pickled. This is required! """
        X,Y,Z = self.generate_XYZ()
        task = mmLINK('l=1;sc=0')
        task.fit(X,Y=None,Z=None)
        with tempfile.TemporaryFile(mode='w+b') as tf:
            cPickle.dump(task, tf)

if __name__ == '__main__':
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    unittest.main()
