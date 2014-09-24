#########################################################
#
#       Unit Test for Credw Task
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

from ModelingMachine.engine.tasks.cred_converters import Credw
from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.container import Container

class TestCredwTasks(unittest.TestCase):
    """ Test suite for Credw
    """
    nsamples=100

    def test_constant(self):
        """ test it works in presence of a constant value """
        X = pd.DataFrame({  'A' : ['a','a','a','a']}, dtype=object)
        Y = pd.Series( np.array([0,1,0,1]).astype('int') )
        Z = Partition(4,folds=2,reps=0,total_size=4)

        task = Credw('dm=2')
        res = task.fit_transform(Container(X),Y,Z)
        # check if equal to mean of training observations for Y
        self.assertEqual(np.all(res(**Z[0])==Y[Z.T(**Z[0])].mean()),True)

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

        task = Credw('dm=1')
        task.fit(Container(X),Y,Z)
        res = task.transform(Container(X),Y,Z)
        # check if name
        self.assertEqual(np.all(res.colnames(**Z[3])==['cred_A','cred_B','cred_C']),True)
        # check if instance
        self.assertIsInstance(res,Container)
        # check if expected result
        self.assertEqual(np.all(res(**Z[0])>0),True)
        self.assertEqual(np.all(res(**Z[1])<1),True)

        # with 2 ways interaction
        X = self.generate_X()
        task = Credw('dm=2')
        task.fit(Container(X),Y,Z)
        res = task.transform(Container(X),Y,Z)
        # check if name
        self.assertEqual(res.colnames(**Z[3])[3]=='cred_A_B',True)

        # check with new data
        Xnew= pd.DataFrame({'A':['g','g','g'], 'B':[1,1,1], 'C':['h','h','h']},dtype=object)
        res = task.transform(Container(Xnew),Y,Z)
        # check if new category mean is equal to M
        p = Z[3]
        key = (p['r'],p['k'])
        print(res(**p))
        print(task.cmap[key][0]['M'])
        self.assertEqual(round(res(**p).mean(),8)==round(task.cmap[key][0]['M'],8),True)

    def test_functionality_same(self):
        X = pd.DataFrame({'a': ['a', 'b', 'b', 'a', 'c', 'a', 'b', 'c', 'b', 'b'],
                          'b': ['a', 'c', 'c', 'a', 'b', 'b', 'c', 'a', 'c', 'c']})
        Y = pd.Series([0, 1, 0, 0, 0, 1, 1, 0, 1])
        Z = Partition(9, folds=5, reps=0, total_size=9)

        task = Credw('')
        task.fit(Container(X),Y,Z)
        res = task.transform(Container(X),Y,Z)
        # check expected result
        check_val = res(r=-1, k=0)
        known_val = np.asarray(
                    [[ 0.39285714,  0.30612245,  0.30612245],
                     [ 0.51785714,  0.51785714,  0.51785714],
                     [ 0.51785714,  0.51785714,  0.51785714],
                     [ 0.39285714,  0.30612245,  0.30612245],
                     [ 0.35714286,  0.44897959,  0.35714286],
                     [ 0.39285714,  0.44897959,  0.52380952],
                     [ 0.51785714,  0.51785714,  0.51785714],
                     [ 0.35714286,  0.30612245,  0.42857143],
                     [ 0.51785714,  0.51785714,  0.51785714],
                     [ 0.51785714,  0.51785714,  0.51785714]])
        residuals = (known_val - check_val)**2
        self.assertTrue(np.all(residuals < 1e-15))

    def test_regression_large_case(self):
        '''This test case, though random, luckily hits both cases for
        k_min >? v / a, which is good for regression testing
        '''
        rstate = np.random.RandomState(2014)
        class_dist = ['a', 'b', 'c', 'd']
        samples = 200
        X = pd.DataFrame({'One': rstate.choice(class_dist, samples),
                          'Two': rstate.choice(class_dist, samples)})
        Y = pd.Series(rstate.randn(samples))
        Z = Partition(samples, folds=2, reps=0,total_size=samples)

        task = Credw('')
        task.fit(Container(X), Y, Z)
        res = task.transform(Container(X), Y, Z)

        check_val = res(r=-1, k=0)[:8, :]
        known_val = [[ 0.0783982,   0.02671196,  0.3650467 ],
                     [-0.03821625, -0.01083806, -0.05790568],
                     [-0.12373287, -0.01083806,  0.10304914],
                     [ 0.0783982,  -0.08518004,  0.06313686],
                     [-0.03821625, -0.01083806, -0.05790568],
                     [-0.03821625, -0.06458371,  0.15466295],
                     [-0.04876892, -0.08518004, -0.0543023 ],
                     [-0.04876892, -0.01083806,  0.15613508]]
        np.testing.assert_almost_equal(check_val, known_val)

    def test_when_columns_have_no_overlap(self):
        '''Tests a case when this kind of interaction happens:
             a    b
        1    a  NaN
        2  NaN    b

        This test doesn't use _exactly_ this data, though
        Before, this was erroring
        '''
        a = pd.Series(['a', 'a', 'e'], index=[0, 1, 2])
        b = pd.Series(['b', 'b', 'c'], index=[1, 2, 3], name='2')
        X = pd.DataFrame({'a': a, 'b': b}, index=[0, 1, 2, 3, 4])
        Y = pd.Series([1., 2., 4., 9., 16.])
        Z = Partition(2, folds=5, reps=0, total_size=2)

        task = Credw()
        task.fit(Container(X),Y,Z)

    def test_pickle(self):
        """ test that the class can be pickled. This is required! """
        X = self.generate_X()
        Y = self.generate_Y()
        Z = self.generate_Z()
        task = Credw()
        task.fit(Container(X),Y,Z)
        with tempfile.SpooledTemporaryFile(mode='w+b') as tf:
            cPickle.dump(task, tf, cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    unittest.main()

