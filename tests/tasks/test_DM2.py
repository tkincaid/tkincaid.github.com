#########################################################
#
#       Unit Test for DesignMatrix2 Task
#
#       Author: Xavier Conort
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

from __future__ import division
import numpy as np
import pandas
import unittest
import tempfile
import cPickle

from ModelingMachine.engine.tasks.converters import DesignMatrix2
from ModelingMachine.engine.tasks.converters import GroupCat
from ModelingMachine.engine.container import Container


class TestDesignMatrix2Tasks(unittest.TestCase):
    """ Test suite for DesignMatrix2
    """
    def create_testdata(self):
        """ create some test data to help in the tests """
        # create a dataframe
        x=  pandas.DataFrame({'A' : ['a', 'a', 'a', 'c', 'e', 'a', 'a',],
                              'B' : ['0', '0', '1', '4', '1', '1', '4'] }, dtype=object)
        xnew=  pandas.DataFrame({'A' : ['a', 'a', 'c', 'd'],
                                 'B' : ['1', '1', '1', '4'] }, dtype=object)
        return x, xnew

    def test_drop(self):
        """ test the column dropping arg """
        x,xnew = self.create_testdata()
        task = DesignMatrix2('sc=0;cm=999;dc=1')

        res = task.fit_transform(Container(x))
        # check if instance
        self.assertIsInstance(res, Container)
        # check if expected result

        # check if name
        self.assertEqual(res.colnames(), ["A-a", "A-c", "B-0", "B-1"])
        np.testing.assert_array_equal(res().todense(),
            np.array([
                [1, 0, 1, 0],
                [1, 0, 1, 0],
                [1, 0, 0, 1],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 0],
            ]))

        task.fit(Container(x))
        res = task.transform(Container(xnew))
        # check if expected result
        self.assertEqual(np.all(res().shape==(4,4)),True)
        self.assertEqual(np.all(res().todense()==
            np.array([
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [0, 1, 0, 1],
                [0, 0, 0, 0],
            ])),True)
        # check if name
        self.assertEqual(np.all(res.colnames()==["A-a", "A-c", "B-0", "B-1"]),True)

    def test_transform(self):
        """ test the transform function of the class """
        x,xnew = self.create_testdata()
        task = DesignMatrix2('sc=1;cm=2')

        res = task.fit_transform(Container(x))
        # check if instance
        self.assertIsInstance(res,Container)
        # check if expected result
        np.testing.assert_array_equal(res().todense(),
                                      np.array([[1],[1],[1],[0],[0],[1],[1]]))
        # check if name
        self.assertEqual(res.colnames(), ["A-a"])

        task.fit(Container(x))
        res = task.transform(Container(xnew))
        # check if expected result
        self.assertEqual(np.all(res().shape==(4,1)),True)
        np.testing.assert_array_equal(res().todense(),
                                      np.array([[1],[1],[0],[0]]))
        # check if name
        self.assertEqual(res.colnames(), ["A-a"])

    def test_single_col_single_cat_return_empty(self):
        """ test that DM2 properly handles single col, single cat data
        """
        task = DesignMatrix2('sc=1;cm=2;dc=1')
        x = pandas.DataFrame({'A': ['a'] * 100})
        res = task.fit_transform(Container(x))
        self.assertIsInstance(res, Container)
        # Return an empty matrix - 0 columns 100 rows
        self.assertEqual(res().shape, (100, 0))

    def test_no_error_if_numeric_df(self):
        """Test that DM2 doesn't die with numeric data
        """
        task = DesignMatrix2('sc=1;cm=2;dc=1')
        x = pandas.DataFrame({'col1': np.logspace(1, 10, 50)})
        # Act-Assert
        task.fit_transform(Container(x))

    def test_binary_missing_df(self):
        """Test if DM2 binary dropping works with missing vals. """
        task = DesignMatrix2('sc=1;cm=10;dc=0')
        x = pandas.DataFrame({'col1': ['a', 'a', None, None]})
        # Act-Assert
        res = task.fit_transform(Container(x))
        np.testing.assert_array_equal(res().todense(), np.array([1, 1, 0, 0])[:, np.newaxis])

    def test_binary_bool_df(self):
        """Test if DM2 binary dropping works with boolean values. """
        task = DesignMatrix2('sc=1;cm=10;dc=0')
        x = pandas.DataFrame({'col1': [True, True, False, False]})
        # Act-Assert
        res = task.fit_transform(Container(x))
        np.testing.assert_array_equal(res().todense(), np.array([1, 1, 0, 0])[:, np.newaxis])

    def test_pickle(self):
        """ test that the class can be pickled. This is required! """
        x,xnew = self.create_testdata()
        task = DesignMatrix2()
        task.fit(Container(x))
        with tempfile.TemporaryFile(mode='w+b') as tf:
            cPickle.dump(task, tf)


class TestGroupCatTasks(unittest.TestCase):
    """ Test suite for GroupCat
    """
    def create_testdata(self):
        """ create some test data to help in the tests """
        # create a dataframe
        x=  pandas.DataFrame({'A' : ['a', 'a', 'a', 'c', 'a', 'a', 'a',],
                              'B' : ['0', '1', '1', '4', '1', '1', '4'] }, dtype=object)
        xnew=  pandas.DataFrame({'A' : ['a', 'a', 'c', 'd'],
                                 'B': ['1', '1', '1', '4'] }, dtype=object)
        return x, xnew

    def test_transform(self):
        """ test the transform function of the class """
        x,xnew = self.create_testdata()
        task = GroupCat(2)
        task.fit(x)

        res = task.transform(x)
        # check if instance
        self.assertIsInstance(res,pandas.DataFrame)

        res = task.transform(xnew)
        # check if expected result
        self.assertEqual(np.all(np.array(res)==np.array([['a','1'],['a','1'],
                                                         ['small_count','1'],
                                                         ['small_count','small_count']])),True)
        # check if name
        self.assertEqual(np.all(res.columns==['A','B']),True)

    def test_corner(self):
        """test some corner cases with missing values. """
        df = pandas.DataFrame(data={'col': [True, None, True, None, None]})
        gc = GroupCat(2)
        gc.fit(df)
        out = gc.transform(df)
        np.testing.assert_array_equal(out.values.ravel(), np.array(['small_count', None, 'small_count',
                                                                    None, None]))

    def test_corner2(self):
        """Test another corner case when I forgot to sort. """
        df = pandas.DataFrame({'A' : ['a', 'a', 'a', 'c', 'e', 'a', 'a',],
                               'B' : ['0', '0', '1', '4', '1', '1', '4'] }, dtype=object)
        gc = GroupCat(0)
        gc.fit(df)
        out = gc.transform(df)
        np.testing.assert_array_equal(df.values, out.values)

    def test_pickle(self):
        """ test that the class can be pickled. This is required! """
        x,xnew = self.create_testdata()
        task = GroupCat()
        task.fit(x)
        with tempfile.TemporaryFile(mode='w+b') as tf:
            cPickle.dump(task, tf)

if __name__ == '__main__':
    unittest.main()

