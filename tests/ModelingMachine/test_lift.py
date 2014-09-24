import unittest
import pytest
import numpy as np
import pandas
import string

from ModelingMachine.engine.lift import lift, lift_by_var, number_format
from ModelingMachine.engine.data_utils import deciles, cut, cut_labels, toplevels

class TestLift(unittest.TestCase):

    @classmethod
    def setupClass(cls):
        np.random.seed(1234)

    def setUp(self):
        self.pred = np.arange(1000)
        self.act = self.pred + np.random.randint(-99,100,size=1000)

    def test_lift(self):
        out = lift(self.act,self.pred,20)

        self.assertIsInstance(out,dict)
        self.assertEqual(set(out.keys()),set(['pred','act','rows','weight']))
        for key in ['pred','act','rows']:
            value = out[key]
            self.assertEqual(len(value),20)
        self.assertEqual(out['weight'],None)

        #lift is not off-balanced by default
        self.assertNotEqual( np.mean(out['pred']), np.mean(out['act']) )

        #test with weight
        constant_weight = 2
        weight = pandas.Series(np.zeros(1000)+constant_weight)
        weight.name = 'blah'
        out2 = lift(self.act, self.pred, 20, weight)
        for key in ['pred','act','rows']:
            self.assertTrue(np.all(constant_weight*np.array(out[key])==np.array(out2[key])))
        self.assertEqual(out2['weight'], 'blah')

    def test_number_format(self):
        testcases = [-.1234, 0.00001234, 1234, 12341234, 0, 23, 12.341234, 1000]
        expected = ['-0.1234', '1.234E-05', '1234', '1.234E+07', '0', '23', '12.34', '1000']
        for x,f in zip(testcases,expected):
            print 'x= %s, f= %s'%(x, number_format(x))
            self.assertEqual(f,number_format(x))

    def test_lift_by_var(self):
        variables = [np.random.randint(0,5,size=1000),
                np.random.randint(0,100,size=1000),
                np.random.sample(1000),
                np.random.choice(list('asdfg'),1000),
                np.random.choice(list(string.letters),1000),
                np.ones(1000),
                np.random.choice([np.NaN,0,1],1000),
                np.array(list(np.random.sample(900))+[np.NaN for i in range(100)]),
                np.random.choice(['a','b',np.NaN],1000)]

        pred = [np.arange(1000), np.ones(1000), np.array([np.NaN]+range(999))]

        constant_weight = 2
        weight = pandas.Series( np.zeros(1000)+constant_weight)
        weight.name = 'blah'

        for x in variables:
            for p in pred:
                for a in pred:
                    act = a + np.random.randint(-99,100,size=1000)
                    xs = pandas.Series(x)
                    out = lift_by_var(act, p, xs, 20)
                    for key,value in out.items():
                        if key=='weight':
                            self.assertEqual(value, None)
                            continue
                        missing_level = 1 if xs.isnull().sum()>0 else 0
                        expected = min(20, xs.nunique()) + missing_level
                        self.assertEqual(len(value), expected)
                        self.assertEqual(np.array(out['rows']).sum(), len(x))

                    #test with weights
                    out2 = lift_by_var(act, p, xs, 20, weight=weight)
                    self.assertEqual(out.keys(), out2.keys())
                    for key in ['pred','act','rows']:
                        self.assertTrue(np.all(constant_weight*np.array(out[key])==np.array(out2[key])))
                    for key in ['order']:
                        self.assertTrue(np.all(np.array(out[key])==np.array(out2[key])))
                    self.assertEqual(out2['weight'], 'blah')







if __name__=='__main__':
    unittest.main()
