#########################################################
#
#       Unit Test for Cred2w Task
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

from ModelingMachine.engine.tasks.cred_converters import Cred2w
from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.container import Container

class TestCred2wTasks(unittest.TestCase):
    """ Test suite for Cred2w
    """
    nsamples=100

    def test_constant(self):
        """ test it works in presence of a constant value """
        X = pd.DataFrame({  'A' : ['a','a','a','a'],'B' : ['a','a','a','a']}, dtype=object)
        Y = pd.Series( np.array([0,1,0,1]).astype('int') )
        Z = Partition(4,folds=2,reps=0,total_size=4)

        task = Cred2w('cmin=0')
        res = task.fit_transform(Container(X),Y,Z)
        # check if equal to mean of training observations for Y
        self.assertEqual(np.all(res(**Z[0])==Y[Z.T(**Z[0])].mean()),True)

    def test_empty(self):
        """ test it works in presence of a constant value """
        X = pd.DataFrame({  'A' : ['a','a','a','a'], 'B' : ['a','a','a','a']}, dtype=object)
        Y = pd.Series( np.array([0,1,0,1]).astype('int') )
        Z = Partition(4,folds=2,reps=0,total_size=4)

        task = Cred2w('cmin=10')
        res = task.fit_transform(Container(X),Y,Z)
        # check it is empty
        self.assertEqual(np.all(res.colnames(**Z[0])==[]),True)

        X = pd.DataFrame({  'A' : ['a','a','a','a']}, dtype=object)
        Y = pd.Series( np.array([0,1,0,1]).astype('int') )
        Z = Partition(4,folds=2,reps=0,total_size=4)

        task = Cred2w('cmin=0')
        res = task.fit_transform(Container(X),Y,Z)
        # check it is empty
        self.assertEqual(np.all(res.colnames(**Z[0])==[]),True)

    def generate_X(self):
        """ create some test data to help in the tests """
        A_pattern = ['a','a','a','c','c','d','e','f']
        B_pattern = ['1','2','2','3']
        C_pattern = ['1','d','f','3']
        X=  pd.DataFrame({  'A' : [random.sample(A_pattern,1)[0] for i in range(self.nsamples)],
                    'B' : [random.sample(B_pattern,1)[0] for i in range(self.nsamples)],
                    'C' : [random.sample(C_pattern,1)[0] for i in range(self.nsamples)]},
                    dtype=object)
        return X

    def generate_Y(self):
        Y_pattern = [0,1,1]
        Y = pd.Series( np.array([random.sample(Y_pattern,1)[0] for i in range(self.nsamples)]).astype('int') )
        return Y

    def generate_Z(self):
        return Partition(self.nsamples,folds=5,reps=0,total_size=self.nsamples)

    def test_transform(self):
        """ test the transform function of the class """
        X = self.generate_X()
        Y = self.generate_Y()
        Z = self.generate_Z()

        task = Cred2w('cmin=4')
        task.fit(Container(X),Y,Z)
        res = task.transform(Container(X),Y,Z)
        # check if name
        self.assertEqual(np.all(res.colnames(**Z[3])==['cred_A_C']),True)
        # check if instance
        self.assertIsInstance(res,Container)

        # test with lower cardinality
        task = Cred2w('cmin=0')
        task.fit(Container(X),Y,Z)
        res = task.transform(Container(X),Y,Z)
        # check if name
        self.assertEqual(np.all(res.colnames(**Z[3])==['cred_A_B','cred_A_C','cred_B_C']),True)
        # check if expected result
        self.assertEqual(np.all(res(**Z[0])>0),True)
        self.assertEqual(np.all(res(**Z[1])<1),True)

        # check with new data
        Xnew= pd.DataFrame({'A':['g','g','g'], 'B':[1,1,1], 'C':['h','h','h']},dtype=object)
        res = task.transform(Container(Xnew),Y,Z)
        # check if new category mean is equal to M
        p = Z[3]
        key = (p['r'],p['k'])
        self.assertEqual(round(res(**p).mean(),8)==round(task.cmap[key][0]['M'],8),True)


    def test_functionality_same(self):
        X = pd.DataFrame({'a': ['a', 'b', 'b', 'a', 'c', 'a', 'b', 'c', 'b', 'b'],
                          'b': ['a', 'c', 'c', 'a', 'b', 'b', 'c', 'a', 'c', 'c']})
        Y = pd.Series([0, 1, 0, 0, 0, 1, 1, 0, 1])
        Z = Partition(9, folds=5, reps=0, total_size=9)

        task = Cred2w('cmin=0')
        task.fit(Container(X),Y,Z)
        res = task.transform(Container(X),Y,Z)
        # check expected result
        check_val = res(r=-1, k=0)
        print check_val
        known_val = np.asarray(
                [[ 0.30612245],
                 [ 0.51785714],
                 [ 0.51785714],
                 [ 0.30612245],
                 [ 0.35714286],
                 [ 0.52380952],
                 [ 0.51785714],
                 [ 0.42857143],
                 [ 0.51785714],
                 [ 0.51785714]])
        np.testing.assert_almost_equal(check_val, known_val)

    def test_pickle(self):
        """ test that the class can be pickled. This is required! """
        X = self.generate_X()
        Y = self.generate_Y()
        Z = self.generate_Z()
        task = Cred2w()
        task.fit(Container(X),Y,Z)
        with tempfile.TemporaryFile(mode='w+b') as tf:
            cPickle.dump(task, tf)


if __name__ == '__main__':
    unittest.main()

