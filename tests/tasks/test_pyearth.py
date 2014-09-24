#### Author: Mark Steadman
#### Copyright: DataRobot, Inc. 2014
import numpy as np
import pandas as pd
from base_task_test import BaseTaskTest
from ModelingMachine.engine.tasks.mmpyearth import PyEarthTransformer
from ModelingMachine.engine.container import Container
from ModelingMachine.engine.partition import Partition

class TestPyEarth(BaseTaskTest):

    def test_transform_smoketest(self):

        X = np.random.randn(100, 10)
        Y = np.random.randn(100, 1)
        XC = Container()
        XC.add(X)
        Z = Partition(size=100, reps=5, folds=5)
        Z.set(max_reps=1, max_folds=1)

        pe = PyEarthTransformer()
        pe.fit(XC, Y, Z)

        out = pe.transform(XC, Y, Z)
        for p in Z:
            self.assertEqual(out(**p).shape[0], 100)

    def test_transform_smoketest_no_pruning(self):
        X = np.random.randn(100, 10)
        Y = np.random.randn(100, 1)
        XC = Container()
        XC.add(X)
        Z = Partition(size=100, reps=5, folds=5)
        Z.set(max_reps=1, max_folds=1)

        pe = PyEarthTransformer('dp=0')
        pe.fit(XC, Y, Z)

        out = pe.transform(XC, Y, Z)
        for p in Z:
            self.assertEqual(out(**p).shape[0], 100)


    def test_transform_smoketest_weights(self):

        X = np.random.randn(100, 10)
        Y = np.random.randn(100, 1)
        wt = {'weight': pd.Series(np.absolute(np.random.randn(100, 1).flatten()).astype(long))}
        XC = Container()
        XC.add(X)
        XC.initialize(wt)
        Z = Partition(size=100, reps=5, folds=5, cv_method='RandomCV')
        Z.set(max_reps=1, max_folds=1)

        pe = PyEarthTransformer()
        pe.fit(XC, Y, Z)

        out = pe.transform(XC, Y, Z)
        for p in Z:
            self.assertEqual(out(**p).shape[0], 100)

    def test_transform_smoketest_no_pruning_weights(self):
        X = np.random.randn(100, 10)
        Y = np.random.randn(100, 1)
        wt = {'weight': pd.Series(np.absolute(np.random.randn(100, 1).flatten()))}

        XC = Container()
        XC.add(X)
        XC.initialize(wt)
        print XC.get('weight')
        print XC.get('weight').values
        Z = Partition(size=100, reps=5, folds=5)
        Z.set(max_reps=1, max_folds=1)

        pe = PyEarthTransformer('dp=0')
        pe.fit(XC, Y, Z)

        out = pe.transform(XC, Y, Z)
        for p in Z:
            self.assertEqual(out(**p).shape[0], 100)


