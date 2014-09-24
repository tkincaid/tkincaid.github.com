#########################################################
#
#       Unit Test for recommender systems mapping tasks
#
#       Author: Peter Prettenhofer
#
#       Copyright DataRobot, Inc. 2014
#
########################################################
import numpy as np
import pytest

from base_task_test import BaseTaskTest
from ModelingMachine.engine.tasks.recsys.sgdsvd import SGDSVDRecommender
from ModelingMachine.engine.tasks.recsys.sgdsvd import LanczosSVDRecommender
from ModelingMachine.engine.tasks.cfconverter import CFConverter


class TestSGDSVDRecommender(BaseTaskTest):

    def test_sgdsvd_smoke(self):
        X, Y, Z = self.create_cf_syn_data(n_samples=10000, n_users=1000, n_items=100)

        cfc = CFConverter()
        cfc.fit(X, Y, Z)
        c = cfc.transform(X, Y, Z)

        task = SGDSVDRecommender('c=10;a=0.01;e=0.01;mi=10')
        task.fit(c, Y, Z)
        p = task.predict(c, Y, Z)

        # some cold users
        X.dataframe['user_id'] += 1000
        c = cfc.transform(X, Y, Z)
        p = task.predict(c, Y, Z)

        # some cold items
        X.dataframe['user_id'] -= 500
        X.dataframe['item_id'] += 500
        c = cfc.transform(X, Y, Z)
        p = task.predict(c, Y, Z)

        t = task.transform(c, Y, Z)
        for p in Z:
            self.assertEqual(t(**p).shape, (X.shape[0], 20))

    @pytest.mark.dscomp
    def test_lanczos_svd_smoke(self):
        X, Y, Z = self.create_cf_syn_data(n_samples=10000, n_users=1000, n_items=100)

        cfc = CFConverter()
        cfc.fit(X, Y, Z)
        c = cfc.transform(X, Y, Z)

        task = LanczosSVDRecommender('c=10')
        task.fit(c, Y, Z)
        p = task.predict(c, Y, Z)

        t = task.transform(c, Y, Z)
        for p in Z:
            self.assertEqual(t(**p).shape, (X.shape[0], 20))

    def test_sgdsvd_topk(self):
        n_items = 100
        n_users = 1000
        X, Y, Z = self.create_cf_syn_data(n_samples=10000, n_users=n_users, n_items=n_items)
        Z = Z.set(partitions=[(-1, -1)])
        self.assertEqual(list(Z), [{'k': -1, 'r': -1}])

        cfc = CFConverter()
        cfc.fit(X, Y, Z)
        c = cfc.transform(X, Y, Z)

        task = SGDSVDRecommender('c=10;a=0.01;e=0.01;mi=10')
        task.fit(c, Y, Z)

        key = (-1, -1)
        topk_items = task.topk_[key].items
        topk_ratings = task.topk_[key].ratings
        k = min(task.n_recommended_items, n_items)
        self.assertEqual(topk_items.shape, (1000, k))
        self.assertEqual(topk_ratings.shape, (1000, k))
        np.testing.assert_array_equal(topk_items[0, :5], [75, 94, 39, 9, 77])
        self.assertEqual(task.topk_[key].known_items.shape, (n_users, n_items))
