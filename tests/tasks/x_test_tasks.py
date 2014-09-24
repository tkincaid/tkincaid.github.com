'''This contains code to generate tests for each task in the taskmap.
For each task we should have enough information to figure out if it
should be tested as a converter, model, or transformer.

'''
import unittest
import os
import time
from tempfile import SpooledTemporaryFile, NamedTemporaryFile
import cPickle

import pandas as pd
import numpy as np

tests_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(tests_path,'../testdata')

from ModelingMachine.engine.task_map import task_map
from ModelingMachine.engine.container import Container
from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.vertex import Vertex

class SupportClass(object):
    '''This class only exists so that we just have to incur the cost of touching
    the disk once.  All the tests just use these data

    '''

    def __init__(self):
        self.text_dataset = self.get_text_dataset()
        self.cat_dataset = self.get_cat_dataset()
        self.reg_dataset = self.get_regress_dataset()
        self.bin_dataset = self.get_binary_dataset()
        self.num_dataset = self.get_num_dataset()

    def get_text_dataset(self):
        x = pd.read_csv(os.path.join(data_path,'fastiron-train-sample-small.csv'),
                        usecols=['fiProductClassDesc', 'SalePrice'],
                        nrows=50)
        Y = x.pop('SalePrice').values
        X = Container()
        X.add(x.values)
        Z = Partition(len(x), folds=5)
        return (x, Y, Z)

    def get_cat_dataset(self):
        x = pd.read_csv(os.path.join(data_path,'fastiron-train-sample-small.csv'),
                        usecols=['ProductGroupDesc', 'SalePrice'],
                        nrows=50)
        X = Container()
        Y = x.pop('SalePrice').values
        X.add(x.values)
        Z = Partition(len(x), folds=5)
        return (x, Y, Z)

    def get_num_dataset(self):
        x = pd.read_csv(os.path.join(data_path, 'credit-sample-200.csv'),
                        usecols=['SeriousDlqin2yrs', 'age', 'RevolvingUtilizationOfUnsecuredLines',
                                 'NumberOfOpenCreditLinesAndLoans'],
                        nrows=50)
        Y = x.pop('SeriousDlqin2yrs').values
        Z = Partition(len(x), folds=5)
        return (x,Y,Z)


    def get_regress_dataset(self):
        x = pd.read_csv(os.path.join(data_path,'credit-sample-200.csv'),
#                        usecols=['SeriousDlqin2yrs', 'age', 'RevolvingUtilizationOfUnsecuredLines',
#                                 'NumberOfOpenCreditLinesAndLoans'],
                        nrows=50)
        x.pop('MonthlyIncome')
        Y = x.pop('age').values
        Z = Partition(len(x), folds=5)
        X = Container()
        X.add(x.values)

        return (X, Y, Z)

    def get_binary_dataset(self):
        x = pd.read_csv(os.path.join(data_path,'credit-sample-200.csv'),
#                        usecols=['SeriousDlqin2yrs', 'age', 'RevolvingUtilizationOfUnsecuredLines',
#                                 'NumberOfOpenCreditLinesAndLoans'],
                        nrows=50)
        x.pop('MonthlyIncome')
        Y = x.pop('SeriousDlqin2yrs').values
        X = Container()
        X.add(x.values)
        Z = Partition(len(x), folds=5)

        return (X, Y, Z)

support = SupportClass()


class TestTasks(unittest.TestCase):
    '''All the tests are generated automatically.  In order to have your
    task tested, it just needs to be in the task_map.  This module will
    detect which type of task it is and test it accordingly.

    '''

    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def setUp(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    @classmethod
    def tearDown(self):
        pass

def generate_predict_test(taskname, dataset):
    def generated(self):
        vertex = Vertex([taskname], None)
        p = vertex.fit_predict(*dataset)
        self.assertIsInstance(p, Container)
        for i in p:
            self.assertEqual(p(**i).shape[0], dataset[0](**i).shape[0])
    return {'testname':'test_'+taskname+'_predict',
            'test':generated}

def generate_transform_test(taskname, dataset):
    def generated(self):
        vertex = Vertex([taskname], None)
        t = vertex.fit_transform(*dataset)
        self.assertIsInstance(t, Container)
        for i in t:
            self.assertEqual(t(**i).shape[0], dataset[0](**i).shape[0])

    return {'testname':'test_'+taskname+'_transform',
            'test':generated}


def generate_converter_test(taskname, dataset):
    def generated(self):
        vertex = Vertex([taskname], None)
        t = vertex.fit_transform(*dataset)
        for i in t:
            self.assertEqual(dataset[0].shape[0], t(**i).shape[0] )

    return {'testname':'test_'+taskname+'_predict',
            'test':generated}

def generate_fit_test(taskname, dataset):
    def generated(self):
        vertex = Vertex([taskname], None)
        x = vertex.fit(*dataset)
        tf = NamedTemporaryFile(delete=False)
        # Test that the result of fit can be pickled
        cPickle.dump(x, tf)
        tf.close()
        assert os.path.exists(tf.name)
        with open(tf.name,'r') as fin:
            obj = cPickle.load(fin)
        os.unlink(tf.name)
    return {'testname':'test_'+taskname+'_fit',
            'test':generated}

def generate_regression_tests(taskname):
    dataset = support.reg_dataset
    return [generate_fit_test(taskname, dataset),
            generate_predict_test(taskname, dataset)]

def generate_binary_tests(taskname):
    dataset = support.bin_dataset
    return [generate_fit_test(taskname, dataset),
            generate_predict_test(taskname, dataset)]

def generate_converter_tests(taskname, types):
    tests = []
    if 'NUM' in types:
        dataset = support.num_dataset
        numtests = [generate_fit_test(taskname, dataset),
                    generate_converter_test(taskname, dataset)]
        for t in numtests: t['testname'] += '_NUM'
        tests.extend(numtests)
    if 'CAT' in types:
        dataset = support.cat_dataset
        cattests = [generate_fit_test(taskname, dataset),
                    generate_converter_test(taskname, dataset)]
        for t in cattests: t['testname'] += '_CAT'
        tests.extend(cattests)
    if 'TXT' in types:
        dataset = support.text_dataset
        txttests = [generate_fit_test(taskname, dataset),
                    generate_converter_test(taskname, dataset)]
        for t in txttests: t['testname'] += '_TXT'
        tests.extend(txttests)
    return tests

def generate_transformer_tests(taskname):
    dataset = support.bin_dataset
    return [generate_fit_test(taskname, dataset),
            generate_transform_test(taskname, dataset)]

def generate_task_tests(taskname):
    task_type = infer_task_type(taskname)
    if task_type == 'transformer':
        return generate_transformer_tests(taskname)
    elif task_type == 'converter':
        return generate_converter_tests(taskname, task_map.get_input_types(taskname) )
    elif task_type == 'regression':
        return generate_regression_tests(taskname)
    elif task_type == 'binary':
        return generate_binary_tests(taskname)


def infer_task_type(taskname):
    '''This logic is how the test infers how a task should be tested'''
    if 'converter_inputs' in task_map[taskname]:
        return 'converter'
    elif 'target_type' in task_map[taskname]:
        ttype = task_map.get_target_type(taskname)
        if ttype == 'b':
            return 'binary'
        else:
            return 'regression'
    else:
        return 'transformer'

###################################################
# These lines of code generate the proper test for
# each of the tasks in the task map.
# This way it is easier to detect which task failed
###################################################

defective_tasks = ['NOISE','CRF','DGLMPE','DGLMTAE','GBCA']

def generate_tests():
    for t in task_map.keys():
        if t not in defective_tasks:
            tests = generate_task_tests(t)
            for test in tests:
                setattr(TestTasks, test['testname'], test['test'])


def generate_one_test():
    '''I used this while developing this test script.  It may prove useful
    if you are finding errors on one particular script (although it shouldn\'t
    be as necessary now that this test runs quickly.

    '''
    task = 'STK'
    tests = generate_task_tests(task)
    for test in tests:
        setattr(TestTasks, test['testname'], test['test'])

generate_tests()
#generate_one_test()

if __name__ == '__main__':
    unittest.main()
