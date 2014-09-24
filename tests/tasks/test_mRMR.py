#########################################################
#
#       Unit Test for mRMR transformer
#
#       Author: Sergey Yurgenson
#
#       Copyright DataRobot, Inc. 2014
#
########################################################
import numpy as np
import unittest
import tempfile
import cPickle
import copy

from ModelingMachine.engine.tasks.mRMR import mRMR
from ModelingMachine.engine.container import Container
from ModelingMachine.engine.partition import Partition
from base_task_test import BaseTaskTest
from ModelingMachine.engine.tasks.cat_encoders import OrdinalEncoder


class TestmRMR(BaseTaskTest):
    """ Test suite for mRMR """

    def create_reg_data(self, reps=1, rows=None):
        X = copy.deepcopy(self.ds2)
        if rows is not None and rows < X.shape[0]:
            X = X[:rows]
        Y = X.pop('Claim_Amount').values
        X = X.take(range(9, 29), axis=1)
        Z = Partition(size=X.shape[0], folds=5, reps=reps, total_size=X.shape[0])
        Z.set(max_reps=reps, max_folds=0)
        return X, Y, Z

    def container_from_dataframe(self, X):
        C = Container()
        cat_cols = [c for c in X.columns
                    if X[c].dtype not in (np.int64, np.float64)]

        # ordinal encode categoricals
        enc = OrdinalEncoder(columns=cat_cols, min_support=1, offset=2)
        X = enc.fit_transform(X)

        def cardinality(col):
            # get cardinality of each categorical column
            if col in cat_cols:
                return int(X[col].max() + 1)
            else:
                return 0

        coltypes = [cardinality(c) for c in X.columns]
        colnames = X.columns.tolist()
        C.add(X.values.astype(np.float), colnames=colnames, coltypes=coltypes)
        return C

    def test_smoke(self):
        """Smoke test for mRMR. """
        X, Y, Z = self.create_reg_data()
        C = self.container_from_dataframe(X)

        task = mRMR('N=5;n_bins=6')
        task.fit(C, Y, Z)
        output = task.transform(C, Y, Z)
        #pytest.set_trace()
        p = {'k': -1, 'r': 0}
        self.assertEqual(output.shape[0], C.shape[0])
        self.assertLess(output(**p).shape[1], task.parameters['N'] + 1)
        self.assertGreater(output(**p).shape[1], 0)
        self.assertEqual(task.index_d[(0, -1)].shape, (C().shape[1], ))
        self.assertEqual(np.where(task.index_d[(0, -1)])[0].tolist(),
                         [6, 7, 8, 9, 17])

        # Classification
        Y = (Y > np.mean(Y)).astype(np.int)
        task = mRMR('N=5;n_bins=6')
        task.fit(C, Y, Z)
        output = task.transform(C, Y, Z)
        #pytest.set_trace()
        self.assertEqual(output.shape[0], C.shape[0])
        self.assertLess(output(**p).shape[1], task.parameters['N'] + 1)
        self.assertGreater(output(**p).shape[1], 0)
        self.assertEqual(task.index_d[(0, -1)].shape, (C().shape[1], ))
        self.assertEqual(np.where(task.index_d[(0, -1)])[0].tolist(),
                         [2, 6, 7, 8, 9])

    def test_pickle(self):
        """ test that the class can be pickled. This is required! """
        X, Y, Z = self.create_reg_data()
        C = self.container_from_dataframe(X)
        task = mRMR('N=2;n_bins=10')
        task.fit(C, Y, Z)
        with tempfile.SpooledTemporaryFile(mode='w+b') as tf:
            cPickle.dump(task, tf)

if __name__ == '__main__':
    unittest.main()
