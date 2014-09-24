#########################################################
#
#       Unit Test for recommender systems neighbors techniques
#
#       Author: Peter Prettenhofer
#
#       Copyright DataRobot, Inc. 2014
#
########################################################
import pytest
import numpy as np

from base_task_test import BaseTaskTest
from ModelingMachine.engine.tasks.recsys.neighbors import ItemKNNRecommender
from ModelingMachine.engine.tasks.cfconverter import CFConverter


class TestNeighborsRecommender(BaseTaskTest):

    @pytest.mark.dscomp
    def test_neighbors_smoke(self):
        X, Y, Z = self.create_cf_syn_data(n_samples=10000, n_users=1000, n_items=100)

        cfc = CFConverter()
        cfc.fit(X, Y, Z)
        c = cfc.transform(X, Y, Z)

        task = ItemKNNRecommender('k=5')
        task.fit(c, Y, Z)
        p = task.predict(c, Y, Z)

        X.dataframe['item_id'] += 10
        c = cfc.transform(X, Y, Z)
        p = task.predict(c, Y, Z)


    # #@patch('ModelingMachine.engine.blueprint_interpreter.BuildData')
    # def test_neighbors_coldstart_metrics(self):
    #     X, Y, Z = self.create_cf_syn_data(n_samples=1000, n_users=1000, n_items=100)
    #     Z.set(partitions=[(0,-1)])
    #     p = {'r': 0, 'k': -1}

    #     cfc = CFConverter()
    #     cfc.fit(X, Y, Z)
    #     c = cfc.transform(X, Y, Z)

    #     print(c(**p)[:, 0])
    #     print('_'*80)

    #     task = ItemKNNRecommender('k=5')
    #     task.fit(c, Y, Z)
    #     pred = task.predict(c, Y, Z)

    #     print(c(**p)[:, 0])
    #     print('_'*80)
    #     train_users = set(X.iloc[Z.T(**p)]['user_id'].tolist())
    #     test_users = set(X.iloc[Z.S(**p)]['user_id'].tolist())
    #     cold_users = test_users - train_users
    #     print(cold_users)
    #     print('n_colds: %d' % len(cold_users))

    #     test_enc_user_ids = c(**p)[Z.S(**p), 0]
    #     X_test = X.iloc[Z.S(**p)]
    #     mask = X_test['user_id'].isin(cold_users).values
    #     #print(pred(**p)[mask])
    #     print(test_enc_user_ids[mask])

    #     pytest.set_trace()
