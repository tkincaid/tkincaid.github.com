#########################################################
#
#       Unit Test for Auto-Tuned Text Mining
#
#       Author: Mark Steadman
#
#       Copyright DataRobot, Inc. 2014
#
########################################################

import unittest
import copy
import os
import pandas
import numpy as np
from base_task_test import BaseTaskTest

from ModelingMachine.engine.tasks.text_mining import AutoTunedWordGramRegressor, AutoTunedWordGramClassifier, AutoTunedCharGramClassifier, AutoTunedCharGramRegressor
from ModelingMachine.engine.container import Container
from ModelingMachine.engine.partition import Partition

class TestTextMiningEstimators(BaseTaskTest):
    pass

class TestTextMiningTask(BaseTaskTest):

    def test_ngrams_smoketest_classification(self):
        xdata = np.repeat(np.array(['dog cat dog cat', 'cat dog dog cat', 'cat dog cat dog', 'dog dog cat cat']), 25).reshape(-1, 1)
        X = Container()
        X.add(xdata)
        y = np.repeat(np.array([1, 0, 0, 0]), 25)
        Z = Partition(size=xdata.shape[0], reps=5)
        Z.set(max_folds=0, max_reps=2)
        taskw = AutoTunedWordGramClassifier('num=[1, 2, 3]')
        taskw.fit(X, y, Z)
        transform = taskw.transform(X, y, Z)
        predictions = taskw.predict(X, y, Z)
        taskc = AutoTunedCharGramClassifier('num=[2, 3, 4]')
        taskc.fit(X, y, Z)
        transform = taskc.transform(X, y, Z)
        predictions = taskc.predict(X, y, Z)

    def test_ngrams_smoketest_regressor(self):
        xdata = np.repeat(np.array(['dog cat dog cat', 'cat dog dog cat', 'cat dog cat dog', 'dog dog cat cat']), 25).reshape(-1, 1)
        X = Container()
        X.add(xdata)
        y = np.repeat(np.array([3, 2, 0, 0]), 25)
        Z = Partition(size=xdata.shape[0], reps=5)
        Z.set(max_folds=0, max_reps=2)
        taskw = AutoTunedWordGramRegressor('num=[1, 2, 3]')
        taskw.fit(X, y, Z)
        transform = taskw.transform(X, y, Z)
        predictions = taskw.predict(X, y, Z)
        taskc = AutoTunedCharGramRegressor('num=[2, 3, 4]')
        taskc.fit(X, y, Z)
        transform = taskc.transform(X, y, Z)
        predictions = taskc.predict(X, y, Z)

    def test_ngrams_words_calculates_ace(self):
        xdata = np.repeat(np.array(['dog cat dog cat', 'cat dog dog cat', 'cat dog cat dog', 'dog dog cat cat']), 25).reshape(-1, 1)
        perm = np.random.permutation(xdata.shape[0])
        X = Container()
        X.add(xdata[perm, :])
        y = np.repeat(np.array([1, 1, 0, 0]), 25)[perm]
        Z = Partition(size=xdata.shape[0], reps=5)
        Z.set(max_folds=0, max_reps=2)
        taskbow = AutoTunedWordGramClassifier('num=1;ma=LogLoss')
        taskbow.fit(X, y, Z)
        predictions = taskbow.predict(X, y, Z)
        report = taskbow.report()
        for p in Z:
            key = (p['r'], p['k'])
            self.assertTrue('var_imp_info' in report[key])
            self.assertTrue(report[key]['var_imp_info'] < 0.1)

        taskw = AutoTunedWordGramClassifier('num=4;ma=LogLoss')
        taskw.fit(X, y, Z)
        transform = taskw.transform(X, y, Z)
        predictions = taskw.predict(X, y, Z)
        report = taskw.report()
        for p in Z:
            key = (p['r'], p['k'])
            print key
            print report[key]['var_imp_info']
            print predictions(**p).ravel()
            print predictions(**p) == y
            self.assertTrue('var_imp_info' in report[key])
            self.assertGreater(report[key]['var_imp_info'],  0.9)

    def test_ngrams_error_handled_gracefully(self):
        xdata = np.repeat(np.array(['dog', 'cat', 'cat', 'dog']), 25).reshape(-1, 1)
        X = Container()
        X.add(xdata)
        y = np.repeat(np.array([3, 2, 0, 0]), 25)
        Z = Partition(size=xdata.shape[0], reps=5)
        Z.set(max_folds=0, max_reps=2)
        taskw = AutoTunedWordGramRegressor('num=[1,2, 3];midf=10')
        taskw.fit(X, y, Z)
        transform = taskw.transform(X, y, Z)
        # This will error but should continue on
        taskc = AutoTunedCharGramRegressor('num=[4, 5, 6];il=0;midf=10')
        taskc.fit(X, y, Z)
        transform = taskc.transform(X, y, Z)
        for p in Z:
            np.testing.assert_array_equal(transform(**p), np.zeros(X(**p).shape))
            np.testing.assert_array_equal(taskc.predict(X, y, Z)(**p).flatten(), np.zeros((X(**p).shape[0], 1)).flatten())

if __name__=='__main__':
    unittest.main()
