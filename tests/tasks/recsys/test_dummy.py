#########################################################
#
#       Unit Test for recommender systems mapping tasks
#
#       Author: Peter Prettenhofer
#
#       Copyright DataRobot, Inc. 2014
#
########################################################
import pytest
import numpy as np

from base_task_test import BaseTaskTest
from ModelingMachine.engine.tasks.recsys.dummy import MostPopularItemsRecommender
from ModelingMachine.engine.tasks.recsys.dummy import UserMeanRecommender
from ModelingMachine.engine.tasks.recsys.dummy import GlobalMeanRecommender
from ModelingMachine.engine.tasks.recsys.dummy import MeanRegressor
from ModelingMachine.engine.tasks.cfconverter import CFConverter


class TestUserItemMapping(BaseTaskTest):

    @pytest.mark.dscomp
    def test_most_popular_items_cat(self):
        X, Y, Z = self.create_cf_syn_data(categoricals=True)

        cfc = CFConverter()

        cfc.fit(X, Y, Z)
        c = cfc.transform(X, Y, Z)

        task = MostPopularItemsRecommender()
        task.fit(c, Y, Z)
        p = task.predict(c, Y, Z)

    @pytest.mark.dscomp
    def test_most_popular_items_cold(self):
        X, Y, Z = self.create_cf_syn_data(categoricals=False)

        cfc = CFConverter()

        cfc.fit(X, Y, Z)
        c = cfc.transform(X, Y, Z)

        task = MostPopularItemsRecommender()
        task.fit(c, Y, Z)
        p = task.predict(c, Y, Z)

        # offset user ids so that we create some cold ones
        X.dataframe['item_id'] += 10
        c = cfc.transform(X, Y, Z)
        p = task.predict(c, Y, Z)

    @pytest.mark.dscomp
    def test_most_popular_items_gs(self):
        X, Y, Z = self.create_cf_syn_data(categoricals=False)

        cfc = CFConverter()

        cfc.fit(X, Y, Z)
        c = cfc.transform(X, Y, Z)

        task = MostPopularItemsRecommender('cw=[0.0, 10.0, 20.0]')
        task.fit(c, Y, Z)
        p = task.predict(c, Y, Z)

    @pytest.mark.dscomp
    def test_user_mean(self):
        X, Y, Z = self.create_cf_syn_data()

        cfc = CFConverter()

        cfc.fit(X, Y, Z)
        c = cfc.transform(X, Y, Z)

        task = UserMeanRecommender()
        task.fit(c, Y, Z)
        p = task.predict(c, Y, Z)

    @pytest.mark.dscomp
    def test_user_mean_w_unk(self):
        """Test that unkown user or items are ignored. """
        X, Y, Z = self.create_cf_syn_data(n_samples=1000, n_users=10)
        cfc = CFConverter()
        cfc.fit(X, Y, Z)
        c = cfc.transform(X, Y, Z)
        ctx = c.get_user_item_context(r=0, k=-1)

        X = c(r=0, k=-1)
        y = Y
        sample_weight = np.ones_like(y)

        for method in ['user', 'item']:
            model = MeanRegressor(method='user', ctx_=ctx)
            model.fit(X, y, sample_weight)

            X_ = np.r_[X, np.ones((10, 2), dtype=np.int) * -1]
            y_ = np.concatenate((y, np.ones(10)))
            sample_weight_ = np.concatenate((sample_weight, np.ones(10)))
            model2 = MeanRegressor(method='user', ctx_=ctx)
            model2.fit(X_, y_, sample_weight_)

            np.testing.assert_array_almost_equal(model.user_mean_,
                                                 model2.user_mean_)
            np.testing.assert_array_almost_equal(model.user_ratings_,
                                                 model2.user_ratings_)

    @pytest.mark.dscomp
    def test_global_mean(self):
        X, Y, Z = self.create_cf_syn_data()

        cfc = CFConverter()

        cfc.fit(X, Y, Z)
        c = cfc.transform(X, Y, Z)

        task = GlobalMeanRecommender()
        task.fit(c, Y, Z)
        p = task.predict(c, Y, Z)
