#########################################################
#
#       Unit Test for TfIdf Converter
#
#       Author: Mark Steadman
#
#       Copyright DataRobot, Inc. 2014
#
########################################################

import unittest
import os
import pandas
import numpy as np
import pytest

from ModelingMachine.engine.container import Container
from ModelingMachine.engine.tasks.converters import TfIdf2, TM_List
from base_task_test import TESTDATA_DIR




class Test_TM_List(unittest.TestCase):
    """Class to unittest the TML task in converters.py"""
    @classmethod
    def setUpClass(cls):
        """Setup class"""
        #here = os.path.dirname(os.path.abspath(__file__))
        #self.ds = pandas.read_csv(os.path.join(here,'../testdata/kickcars-sample-200.csv'))
        pass

    def create_data(self):
        """Setup dataframe"""
        docs = np.array(['dog cat', 'Dog', 'Cat'])
        s = pandas.DataFrame({'docs': docs})
        return Container(s)

    @pytest.mark.skip
    def test_TML_argument_parsing(self):
        """Test argument passing"""
        # TML currently use old style argument passing. Skip for now.
        task = TM_List()
        check = task.get_default_parameters(task.arguments)
        self.assertEqual(task.parameters, check)

        args = 'de=0;sw=1;b=1;id=1;su=0'
        task = TM_List(args)
        check = task.get_default_parameters(task.arguments)
        check.update({'decode_error': 'strict', 'stop_words': 'english',
                      'binary': True, 'use_idf': True, 'sublinear_tf': False})
        self.assertEqual(task.parameters, check)

        args = 'de=1;sw=0;b=1;id=1;nr=[1,4]'
        task = TM_List(args)
        check = task.get_default_parameters(task.arguments)
        check.update({'decode_error': 'ignore', 'stop_words': None,
                      'binary': True, 'use_idf': True, 'ngram_range': (1, 4)})
        self.assertEqual(task.parameters, check)

    def test_TML_default_arguments(self):
        """Test that TML works with default arguments"""
        x = self.create_data()
        #default arguments = no idf weighting, lower_case=True
        task = TM_List()
        task.fit(x)
        self.assertEqual(task.vec['docs'].get_feature_names(), ['cat', 'dog', 'dog cat'])
        out = task.transform(x)
        np.testing.assert_allclose(out.tm_list[0].todense(), np.array([[1.288, 1.288, 1.693],
                                                                       [0, 1.288, 0],
                                                                       [1.288, 0, 0]]), 0.001)

    def test_TML_binary(self):
        """Test that TML works with binary method"""
        x = self.create_data()
        task = TM_List('bnnnn11')
        #lc = lower_cave (0=False,1=True)
        task.fit(x)
        self.assertEqual(task.vec['docs'].get_feature_names(), ['Cat', 'Dog', 'cat', 'dog'])
        out = task.transform(x)
        np.testing.assert_allclose(out.tm_list[0].todense(), np.array([[0, 0, 1, 1],
                                                                       [0, 1, 0, 0],
                                                                       [1, 0, 0, 0]]))

    def test_TML_ngram_range(self):
        """Test that TML calculates bigrams correctly"""
        x = self.create_data()
        #nr = ngram_range
        task = TM_List('bnnnn12')
        task.fit(x)
        self.assertEqual(task.vec['docs'].get_feature_names(), ['Cat', 'Dog', 'cat', 'dog', 'dog cat'])
        out = task.transform(x)
        np.testing.assert_allclose(out.tm_list[0].todense(), np.array([[0, 0, 1, 1, 1],
                                                                       [0, 1, 0, 0, 0],
                                                                       [1, 0, 0, 0, 0]]))

    def test_TML_term_frequency(self):
        """Test that TML calculates term_frequency correctly"""
        x = self.create_data()
        task = TM_List('btnnn12')
        #is = use_idf = 1+log(N/df), where N = number of documents, df = document frequency (of the term)
        task.fit(x)
        self.assertEqual(task.vec['docs'].get_feature_names(), ['Cat', 'Dog', 'cat', 'dog', 'dog cat'])
        out = task.transform(x)
        expected = np.array([[0, 0, 1.693, 1.693, 1.693],
                             [0, 1.693, 0, 0, 0],
                             [1.693, 0, 0, 0, 0]])
        np.testing.assert_allclose(out.tm_list[0].todense(), expected, 0.001)

    @pytest.mark.skip
    def test_TML_smoothing(self):
        """Test that TML works with smoothing"""
        # Not implemented in tfidf so skip for now
        x = self.create_data()
        task = TfIdf2('nr=[1,2];id=1;sm=1')
        #sm = smooth_idf = add 1 to N and to all df's
        task.fit(x)
        self.assertEqual(task.vec['docs'].get_feature_names(), ['cat', 'dog', 'dog cat'])
        out = task.transform(x)
        expected = np.array([[1+np.log(4./3.), 1+np.log(4./3.), 1+np.log(4./2.)],
                             [0,               1+np.log(4./3.), 0              ],
                             [1+np.log(4./3.), 0,               0              ]])
        np.testing.assert_allclose(out.tm_list[0].todense(), expected, 0.001)

    def test_TML_default_dataframe(self):
        """Test that TML works with defaults and DataFrames"""
        docs = np.array(['dog cat', 'Dog', 'Cat Cat'])
        x = pandas.DataFrame({'docs': docs})
        x = Container(x)
        #b = binary
        task = TM_List('bnnnn11')
        task.fit(x)
        self.assertEqual(task.vec['docs'].get_feature_names(), ['Cat', 'Dog', 'cat', 'dog'])
        out = task.transform(x)
        expected = np.array([[0, 0, 1, 1],
                             [0, 1, 0, 0],
                             [1, 0, 0, 0]])
        np.testing.assert_allclose(out.tm_list[0].todense(), expected, 0.001)

    def test_TML_dataframe_sublinear_fit(self):
        """Test that TML works with sublinear fit version"""
        docs = np.array(['dog cat', 'Dog', 'Cat Cat'])
        x = pandas.DataFrame({'docs': docs})
        x = Container(x)
        #su = sublinear_tf = replace tf by 1+log(tf)
        task = TM_List('lnnnn11')
        task.fit(x)
        self.assertEqual(task.vec['docs'].get_feature_names(), ['Cat', 'Dog', 'cat', 'dog'])
        out = task.transform(x)
        expected = np.array([[0, 0, 1, 1],
                             [0, 1, 0, 0],
                             [1+np.log(2), 0, 0, 0]])
        np.testing.assert_allclose(out.tm_list[0].todense(), expected, 0.001)

    def test_TML_dataframe_normalize_l1(self):
        """Test that TML works with l1 normalization"""
        docs = np.array(['dog cat', 'Dog', 'Cat Cat'])
        x = pandas.DataFrame({'docs': docs})
        x = Container(x)
        #n= normalize (0=None,1=l1,2=l2)
        task = TM_List('bn1nn11')
        task.fit(x)
        self.assertEqual(task.vec['docs'].get_feature_names(), ['Cat', 'Dog', 'cat', 'dog'])
        out = task.transform(x)
        expected = np.array([[0, 0, 1./2., 1./2.],
                             [0, 1., 0, 0],
                             [2./2., 0, 0, 0]])
        np.testing.assert_allclose(out.tm_list[0].todense(), expected, 0.001)

    def test_TML_dataframe_normalize_l2(self):
        """Test that TML works with l2 normalization"""
        docs = np.array(['dog cat', 'Dog', 'Cat Cat'])
        x = pandas.DataFrame({'docs': docs})
        x = Container(x)
        #n= normalize (0=None,1=l1,2=l2)
        task = TM_List('bn2nn11')
        task.fit(x)
        self.assertEqual(task.vec['docs'].get_feature_names(), ['Cat', 'Dog', 'cat', 'dog'])
        out = task.transform(x)
        expected = np.array([[0, 0, 1./np.sqrt(2.), 1./np.sqrt(2.)],
                             [0, 1., 0, 0],
                             [2./2., 0, 0, 0]])
        np.testing.assert_allclose(out.tm_list[0].todense(), expected, 0.001)

    def test_TML_dataframe_stopwords(self):
        """Test that TML works with stopwords"""
        docs = np.array(['a dog and cat', 'a Dog', 'Cat and Cat'])
        x = pandas.DataFrame({'docs': docs})
        x = Container(x)
        #sw = remove stop words
        task = TM_List('bnnny11')
        task.fit(x)
        self.assertEqual(task.vec['docs'].get_feature_names(), ['Cat', 'Dog', 'cat', 'dog'])
        out = task.transform(x)
        expected = np.array([[0, 0, 1, 1],
                             [0, 1, 0, 0],
                             [1, 0, 0, 0]])
        np.testing.assert_allclose(out.tm_list[0].todense(), expected, 0.001)

    def test_TML_ngram_range_too_large_works(self):
        """Test that if ngram range does not include anything, it still works
        Note: This is different behavior from the tml list pre - 2/27/2014
        """
        docs = np.array(['dog', 'Cat'])
        x = pandas.DataFrame({'docs': docs})
        x = Container(x)
        #a= analyzer (word, char)
        task = TM_List('bnnnn24')
        task.fit(x)

    @pytest.mark.dscomp
    def test_TML_handles_unicode(self):
        """Test if TFIDF handles unicode gracefully. """
        df = pandas.read_csv(os.path.join(TESTDATA_DIR, 'amazon_de_reviews_200.csv'),
                             encoding='utf-8')
        # narrow down to the two text columns
        df = df[['summary', 'text']]
        task = TM_List()
        task.fit(Container(df))
        assert True  # this means we didn't get any unicode errors!

if __name__ == '__main__':
    unittest.main()
