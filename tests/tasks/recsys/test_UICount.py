#########################################################
#
#       Unit Test for UICount Task
#
#       Author: Xavier Conort
#
#       Copyright DataRobot, Inc. 2014
#
########################################################
import pytest
import numpy as np
import pandas as pd
import cPickle

from base_task_test import BaseTaskTest
from ModelingMachine.engine.tasks.recsys.converters import UICOUNT
from ModelingMachine.engine.container import Container


class TestUICOUNT(BaseTaskTest):
    """ Test suite for UICOUNT
    """

    def test_transform(self):
        X, Y, Z = self.create_cf_syn_data(n_samples=20, n_users=20,
            n_items=10, categoricals=False)

        task = UICOUNT('thr=2')
        task.fit(X, Y, Z)

        res = task.transform(X)
        # check if instance
        self.assertIsInstance(res, Container)
        self.assertEqual(res.shape[0], X.shape[0])
        # check if name
        self.assertEqual(res.colnames(), ['user_id_count', 'item_id_count',
                                          'item_id_count_user_id_median',
                                          'user_id_count_item_id_median',
                                          'user_id_item_id_count'])
        #test on new data
        X2, Y, Z = self.create_cf_syn_data(n_samples=10, n_users=50,
            n_items=15, categoricals=False)
        res = task.transform(X2)
        # check shape
        for p in Z:
            self.assertEqual(res(**p).shape, (10, 5))
            self.assertEqual(np.all(res(**p).mean(0)==[1.2, 1.5, 1.85, 1.2, 1]), True)


