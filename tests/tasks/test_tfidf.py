#########################################################
#
#       Unit Test for TfIdf Converter
#
#       Author: Tom de Godoy
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

import unittest
import copy
import os
import pandas
import numpy as np

from ModelingMachine.engine.tasks.converters import TfIdf2
from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.container import Container

from base_task_test import TESTDATA_DIR


class TestTfIdf2(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        #here = os.path.dirname(os.path.abspath(__file__))
        #self.ds = pandas.read_csv(os.path.join(here,'../testdata/kickcars-sample-200.csv'))
        pass

    def create_data(self):
        docs = np.array(['dog cat','Dog','Cat'])
        s = pandas.DataFrame({'docs':docs})
        return Container(s)

    def test_argument_parsing(self):
        task = TfIdf2()
        check = task.get_default_parameters(task.arguments)
        self.assertEqual(task.parameters,check)

        args = 'de=0;sw=1;b=1;id=1;su=0'
        task = TfIdf2(args)
        check = task.get_default_parameters(task.arguments)
        check.update({'decode_error':'strict','stop_words':'english','binary':True,'use_idf':True,'sublinear_tf':False})
        self.assertEqual(task.parameters,check)

        args = 'de=1;sw=0;b=1;id=1;nr=[1,4]'
        task = TfIdf2(args)
        check = task.get_default_parameters(task.arguments)
        check.update({'decode_error':'ignore','stop_words':None,'binary':True,'use_idf':True,'ngram_range':(1,4)})
        self.assertEqual(task.parameters,check)

    def test_01(self):
        x = self.create_data()
        #default arguments = no idf weighting, lower_case=True
        task = TfIdf2()
        task.fit(x)
        self.assertEqual( task.vec['docs'].get_feature_names(), ['cat','dog'])
        out = task.transform(x)
        self.assertTrue(np.all( out().toarray()==np.array([[1,1],[0,1],[1,0]]) ))

    def test_02(self):
        x = self.create_data()
        task = TfIdf2('lc=0')
        #lc = lower_cave (0=False,1=True)
        task.fit(x)
        self.assertEqual( task.vec['docs'].get_feature_names(), ['Cat','Dog','cat','dog'])
        out = task.transform(x)
        self.assertTrue(np.all( out().toarray()==np.array([[0,0,1,1],[0,1,0,0],[1,0,0,0]]) ))

    def test_03(self):
        x = self.create_data()
        #nr = ngram_range
        task = TfIdf2('nr=[1,2]')
        task.fit(x)
        self.assertEqual( task.vec['docs'].get_feature_names(), ['cat','dog','dog cat'])
        out = task.transform(x)
        self.assertTrue(np.all( out().toarray()==np.array([[1,1,1],[0,1,0],[1,0,0]]) ))

    def test_04(self):
        x = self.create_data()
        task = TfIdf2('nr=[1,2];id=1')
        #is = use_idf = 1+log(N/df), where N = number of documents, df = document frequency (of the term)
        task.fit(x)
        self.assertEqual( task.vec['docs'].get_feature_names(), ['cat','dog','dog cat'])
        out = task.transform(x)
        expected = np.array([[1+np.log(3./2.), 1+np.log(3./2.), 1+np.log(3./1.)],
                             [0,               1+np.log(3./2.), 0              ],
                             [1+np.log(3./2.), 0,               0              ]])
        self.assertTrue(np.all( out().toarray()==expected ))

    def test_05(self):
        x = self.create_data()
        task = TfIdf2('nr=[1,2];id=1;sm=1')
        #sm = smooth_idf = add 1 to N and to all df's
        task.fit(x)
        self.assertEqual( task.vec['docs'].get_feature_names(), ['cat','dog','dog cat'])
        out = task.transform(x)
        expected = np.array([[1+np.log(4./3.), 1+np.log(4./3.), 1+np.log(4./2.)],
                             [0,               1+np.log(4./3.), 0              ],
                             [1+np.log(4./3.), 0,               0              ]])
        self.assertTrue(np.all( out().toarray()==expected ))

    def test_06(self):
        docs = np.array(['dog cat','Dog','Cat Cat'])
        x = Container(pandas.DataFrame({'docs':docs}))
        #b = binary
        task = TfIdf2('b=0')
        task.fit(x)
        self.assertEqual( task.vec['docs'].get_feature_names(), ['cat','dog'])
        out = task.transform(x)
        expected = np.array([[1, 1],
                             [0, 1],
                             [2, 0]])
        self.assertTrue(np.all( out().toarray()==expected ))

    def test_07(self):
        docs = np.array(['dog cat','Dog','Cat Cat'])
        x = Container(pandas.DataFrame({'docs':docs}))
        #su = sublinear_tf = replace tf by 1+log(tf)
        task = TfIdf2('b=0;su=1')
        task.fit(x)
        self.assertEqual( task.vec['docs'].get_feature_names(), ['cat','dog'])
        out = task.transform(x)
        expected = np.array([[1,           1],
                             [0,           1],
                             [1+np.log(2), 0]])
        self.assertTrue(np.all( out().toarray()==expected ))

    def test_08(self):
        docs = np.array(['dog cat','Dog','Cat Cat'])
        x = Container(pandas.DataFrame({'docs':docs}))
        #n= normalize (0=None,1=l1,2=l2)
        task = TfIdf2('b=0;n=1')
        task.fit(x)
        self.assertEqual( task.vec['docs'].get_feature_names(), ['cat','dog'])
        out = task.transform(x)
        expected = np.array([[1./2., 1./2.],
                             [0    , 1.   ],
                             [2./2., 0    ]])
        self.assertTrue(np.all( out().toarray()==expected ))

    def test_09(self):
        docs = np.array(['dog cat','Dog','Cat Cat'])
        x = Container(pandas.DataFrame({'docs':docs}))
        #n= normalize (0=None,1=l1,2=l2)
        task = TfIdf2('b=0;n=2')
        task.fit(x)
        self.assertEqual( task.vec['docs'].get_feature_names(), ['cat','dog'])
        out = task.transform(x)
        expected = np.array([[1./np.sqrt(2.), 1./np.sqrt(2.) ],
                             [0             , 1.             ],
                             [2./2.         , 0              ]])
        self.assertTrue(np.all( out().toarray()==expected ))

    def test_11(self):
        docs = np.array(['a dog and cat','a Dog','Cat and Cat'])
        x = Container(pandas.DataFrame({'docs':docs}))
        #sw = remove stop words
        task = TfIdf2('b=0;sw=1')
        task.fit(x)
        self.assertEqual( task.vec['docs'].get_feature_names(), ['cat','dog'])
        out = task.transform(x)
        expected = np.array([[1, 1],
                             [0, 1],
                             [2, 0]])
        self.assertTrue(np.all( out().toarray()==expected ))

    def test_12(self):
        docs = np.array(['dog','Cat'])
        x = Container(pandas.DataFrame({'docs':docs}))
        #a= analyzer (word, char)
        task = TfIdf2('a=1')
        task.fit(x)
        self.assertEqual( task.vec['docs'].get_feature_names(), ['a','c','d','g','o','t'])
        out = task.transform(x)
        expected = np.array([[ 0,0,1,1,1,0 ],
                             [ 1,1,0,0,0,1 ]])
        self.assertTrue(np.all( out().toarray()==expected ))

    def test_13(self):
        docs = np.array(['dog','Cat'])
        x = Container(pandas.DataFrame({'docs':docs}))
        #a= analyzer (word, char)
        task = TfIdf2('a=1;nr=[1,2]')
        task.fit(x)
        self.assertEqual( task.vec['docs'].get_feature_names(), ['a','at','c','ca','d','do','g','o','og','t'])
        out = task.transform(x)
        expected = np.array([[ 0,0,0,0,1,1,1,1,1,0 ],
                             [ 1,1,1,1,0,0,0,0,0,1 ]])
        self.assertTrue(np.all( out().toarray()==expected ))

    def test_14(self):
        docs = np.array(['dog','Cat'])
        x = Container(pandas.DataFrame({'docs':docs}))
        #a= analyzer (word, char)
        task = TfIdf2('a=0;nr=[2,4]') # nr is here set too high
        task.fit(x)
        self.assertEqual( task.vec['docs'].get_feature_names(), ['cat','dog'])
        out = task.transform(x)
        expected = np.array([[ 0,1 ],
                             [ 1,0 ]])
        self.assertTrue(np.all( out().toarray()==expected ))

    def test_unicode(self):
        """Test if TFIDF handles unicode gracefully. """
        df = pandas.read_csv(os.path.join(TESTDATA_DIR, 'amazon_de_reviews_200.csv'),
                             encoding='utf-8')
        # narrow down to the two text columns
        df = df[['summary', 'text']]
        task = TfIdf2()
        task.fit(Container(df))
        assert True  # this means we didn't get any unicode errors!

    def test_order_of_columns_consistent(self):
        """ Test to makes sure that order of colnames is consistent. """
        df = pandas.read_csv(os.path.join(TESTDATA_DIR, 'amazon_de_reviews_200.csv'),
                             encoding='utf-8')
        # narrow down to the two text columns
        df = df[['summary', 'text']]
        Z = Partition(df.shape[0])
        Z.set(max_folds=1, max_reps=-1)
        task1 = TfIdf2()
        task1.fit(Container(df), Z=Z)
        task2 = TfIdf2()
        task2.fit(Container(df), Z=Z)
        out1 = task1.transform(Container(df), Z=Z)
        out2 = task2.transform(Container(df), Z=Z)
        self.assertEqual(out1.colnames((1, -1)), out2.colnames((1, -1)))
        np.testing.assert_allclose(out1((1, -1)).todense(), out2((1, -1)).todense())
        Z.set(max_folds=2, max_reps=-1)
        task1.fit(Container(df), Z=Z)
        out3 = task1.transform(Container(df), Z=Z)
        self.assertEqual(out1.colnames((1, -1)), out3.colnames((1, -1)))
        np.testing.assert_allclose(out1((1, -1)).todense(), out3((1, -1)).todense())
        Z.set(max_folds=-1, max_reps=-1)
        out4 = task1.transform(Container(df), Z=Z)
        self.assertEqual(out1.colnames((1, -1)), out4.colnames((1, -1)))

if __name__ == '__main__':
    unittest.main()
