import os
import logging
import pytest
import pandas as pd
import numpy as np

from config.app_config import config as app_config
from predictionapi.prediction_api import app as api_app
from common.engine import metrics
from tests.IntegrationTests.integration_test_base import IntegrationTestBase
from tests.IntegrationTests.test_predictionapi import PredictionAPIUtilMixin

logger = logging.getLogger('datarobot')


class TestPredictionConsistency(IntegrationTestBase, PredictionAPIUtilMixin):
    """Test suite that checks if download predictions and prediction api
    deliver consistent results.
    """

    @classmethod
    def setUpClass(self):
        super(TestPredictionConsistency, self).setUpClass()
        self.api_app = api_app.test_client()

    def test_prediction_full_dataset(self):
        filename = 'amazon-sample-1000.csv'
        self.set_dataset(filename, 'ACTION', metric=metrics.AUC)
        pid = self.create_project()
        q_item, leaderboard_item = self.execute_fit(pid)

        lid = leaderboard_item['_id']

        #Submit item to predict
        item_to_predict = q_item
        self.assertIsNotNone(item_to_predict)

        new_test_file = self.path_to_test_file(filename)

        # upload file
        scoring_dataset_id = self.upload_new_file(self.app, pid,
                                                  new_test_file=new_test_file)
        data = {'predict':1, 'scoring_dataset_id' : scoring_dataset_id,
                'lid':lid, 'new_lid':False, 'partition_stats':{str((0,-1)):'blah'}}
        item_to_predict.update(data)

        # submit predict request
        not_started = self.add_request_to_queue(pid, item_to_predict, lid)
        self.assertEqual(len(not_started), 1)

        predict_item = not_started.pop()

        logger.info('predict_item: %r', predict_item)
        self.assertEqual(predict_item['dataset_id'], item_to_predict['dataset_id'])
        self.assertEqual(predict_item['pid'], item_to_predict['pid'])

        # Kick off new work and wait for the prediciton to finish
        self.wait_for_q_item_to_complete(self.app, pid=pid,
                                         qid=str(predict_item['qid']), timeout=35)

        # Download prediction - will be stored in DOWNLOAD_FOLDER
        response = self.app.get('/predictions/%s/%s' % (lid, scoring_dataset_id))
        self.assertEqual(response.status_code, 200,
                         'Could not find predictions for lid: {0} and dataset_id: '
                         '{1}. Server responded: {2}'.format(lid, scoring_dataset_id,
                                                             response.data))
        dwn_pred = pd.read_csv(os.path.join(app_config["DOWNLOAD_FOLDER"],
              os.path.split(response.headers.get('X-Accel-Redirect'))[-1]),
                               index_col=0)
        dwn_preds = dwn_pred['Prediction']
        logger.info(dwn_preds.head(4))

        # Now get predictions from the prediction api
        pa_preds = None
        username = self.registered_user['username']
        pwd = self.registered_user['password']
        with self.api_app as pred_api:
            api_token = self.get_api_token(pred_api, username, pwd)
            logger.info('api_token: %r', api_token)
            with open(new_test_file, 'rb') as test_file:
                response = self.post_prediction(pred_api, pid, lid, username, api_token,
                                                data=test_file.read(),
                                                headers={'content-type': 'text/plain; charset=utf8'})
                self.assertEqual(response.status_code, 200)
                pa_preds = self.api_response_to_dataframe(response)

        # compare the two
        self.assertTrue(pa_preds is not None)
        self.assertLess(np.max(np.abs(dwn_preds - pa_preds)), 1e-14)
