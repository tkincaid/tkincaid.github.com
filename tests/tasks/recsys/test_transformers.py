#########################################################
#
#       Unit Test for recommender systems transformer
#
#       Author: Peter Prettenhofer
#
#       Copyright DataRobot, Inc. 2014
#
########################################################
import numpy as np
import pytest

from base_task_test import BaseTaskTest
from ModelingMachine.engine.tasks.recsys.transformers import NumericalFeatureDev
from ModelingMachine.engine.tasks.recsys.transformers import CredEstDev
from ModelingMachine.engine.tasks.cfconverter import CFConverter


class TestUserItemMapping(BaseTaskTest):

    @pytest.mark.dscomp
    def test_NumericalFeatureDev(self):
        X, Y, Z = self.create_cf_syn_data(n_samples=500, n_users=20, n_items=5, categoricals=True)
        cfc = CFConverter()
        cfc.fit(X, Y, Z)
        c = cfc.transform(X, Y, Z)
        # add some numerical features to the mix
        for p in Z:
            c.add(np.random.rand(X.shape[0], 10), colnames=['NUM%d' % i for i in range(10)])

        task = NumericalFeatureDev('per=2')
        task.fit(c, Y, Z)
        o = task.transform(c, Y, Z)

        for p in Z:
            self.assertEqual(o(**p).shape[0], X.shape[0])
            self.assertEqual(o(**p).shape[1], 20)

        # with new data
        #----------------------------------------------------------------------
        X2, Y, Z = self.create_cf_syn_data(n_samples=10, n_users=50, n_items=10, categoricals=True)
        c2 = cfc.transform(X2, Y, Z)
        # add some numerical features to the mix
        for p in Z:
            c2.add(np.random.rand(X2.shape[0], 10), colnames=['NUM%d' % i for i in range(10)])

        o = task.transform(c2, Y, Z)
        for p in Z:
            self.assertEqual(o(**p).shape[1], 20)

        # test absolute diff
        #----------------------------------------------------------------------
        task = NumericalFeatureDev('per=2;abs=1')
        task.fit(c, Y, Z)
        o = task.transform(c, Y, Z)

        for p in Z:
            self.assertEqual(np.all(o(**p) >= 0), True)


    @pytest.mark.dscomp
    def test_CredEstDev(self):
        X, Y, Z = self.create_cf_syn_data(n_samples=20, n_users=10, n_items=5, categoricals=True)
        cfc = CFConverter()
        cfc.fit(X, Y, Z)
        c = cfc.transform(X, Y, Z)
        # add credibility features to the mix
        for p in Z:
            c.add(np.random.rand(X.shape[0], 2), colnames=['DR_Cred_user_id', 'DR_Cred_item_id'])

        task = CredEstDev('per=2;thr=2')
        task.fit(c, Y, Z)
        o = task.transform(c, Y, Z)
        for p in Z:
            self.assertEqual(o(**p).shape[0], X.shape[0])
            self.assertEqual(o(**p).shape[1], 2)
            self.assertEqual(o.colnames(**p),
                ['DR_Cred_user_id_dev_from_median_per_user_id', 'DR_Cred_item_id_dev_from_median_per_item_id'])

        # with new data
        X2, Y, Z = self.create_cf_syn_data(n_samples=10, n_users=50, n_items=10, categoricals=True)
        c2 = cfc.transform(X2, Y, Z)
        # add some numerical features to the mix
        for p in Z:
            c2.add(np.random.rand(X2.shape[0], 2), colnames=['DR_Cred_user_id', 'DR_Cred_item_id'])

        o = task.transform(c2, Y, Z)
        for p in Z:
            self.assertEqual(o(**p).shape[1], 2)
