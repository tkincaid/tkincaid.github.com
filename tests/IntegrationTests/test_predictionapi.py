# -*- coding: utf-8 -*-
############################################################################
#
#       integration test for prediction api
#
#       Author: Sean Cronin & Peter Prettenhofer
#
#       Copyright DataRobot, Inc. 2013
#
###########################################################################
import json
import logging
import base64
import numpy as np
import pandas as pd
import unittest
import os
from bson import ObjectId
from mock import patch

from cStringIO import StringIO

import config.test_config
from MMApp.entities.user import UserService
from common.entities.dataset import DatasetServiceBase
from predictionapi.prediction_api import app as api_app
from common.wrappers import database
from config.engine import EngConfig

from tests.IntegrationTests.integration_test_base import IntegrationTestBase

logger = logging.getLogger('datarobot')

api_app.config['TESTING'] = True


class PredictionAPIUtilMixin(object):
    """An integration test utility mixin for prediction api tests. """

    def get_api_token(self, app, user, pwd):
        """Utility to get an api token """
        authorization = ('Basic ' + base64.b64encode(user + ":" + pwd))
        api_token_resp = app.post('/v1/api_token', headers={'Authorization': authorization})
        if api_token_resp.status != '200 OK':
            raise ValueError(api_token_resp.status)
        api_token = json.loads(api_token_resp.data)['api_token']
        return api_token

    def post_prediction(self, app, pid, lid, username, api_token, headers=None,
                        data=None):
        """Post a prediction api request using ``data`` and returns response.

        If ``data`` is None it uses ``self.new_test_file_name`` as a file upload.
        """
        authorization = ('Basic ' + base64.b64encode(username + ":" + api_token))
        with open(self.new_test_file, 'rb') as test_file:
            file_content = StringIO(test_file.read())
            url = '/v1/%s/%s/predict' % (pid, lid)
            if data is None:
                data = {'file': (file_content, self.new_test_file_name)}
            headers_ = {'Authorization': authorization}
            if headers is not None:
                headers_.update(headers)
            response = app.post(url, data=data, headers=headers_)
            return response

    def api_response_to_dataframe(self, response):
        assert response.status_code == 200
        predictions = json.loads(response.data)
        score = lambda p: p['prediction']
        if predictions['task'] == 'Binary':
            score = lambda p: p['class_probabilities']['1']
        preds = sorted([(p['row_id'], score(p)) for p in predictions['predictions']])
        index, vals = map(np.array, zip(*preds))
        preds = pd.Series(index=index + 1, data=vals)
        return preds

    def post_data(self, app, pid, username, api_token, filename, headers=None):
        authorization = ('Basic ' + base64.b64encode(username + ":" + api_token))
        with open(filename, 'rb') as test_file:
            url = '/v1/%s/data' % pid
            data = {'file': test_file}
            headers_ = {'Authorization': authorization}
            if headers is not None:
                headers_.update(headers)
            response = app.post(url, data=data, headers=headers_)
            return response

    def post_model_refresh(self, app, pid, lid, username, api_token, headers=None):
        authorization = ('Basic ' + base64.b64encode(username + ":" + api_token))
        url = '/v1/%s/%s/refresh' % (pid, lid)
        headers_ = {'Authorization': authorization}
        if headers is not None:
            headers_.update(headers)
        response = app.post(url, headers=headers_)
        return response



class PredictionApiTest(IntegrationTestBase, PredictionAPIUtilMixin):
    """Integration tests for our prediction API """

    @classmethod
    def setUpClass(self):
        super(PredictionApiTest, self).setUpClass()
        self.api_app = api_app.test_client()
        self.persistent = database.new('persistent')

    def tearDown(self):
        self.logout(self.app)

    def check_predict_transient_smoke(self):
        """Uploads kickcars, puts a models in the queue and gets predictions. """
        self.set_dataset('kickcars-sample-200.csv', 'IsBadBuy')
        ## 1. Fit a model so that we have something to predict
        # upload file
        username = self.registered_user['username']
        pwd = self.registered_user['password']

        pid = self.create_project()
        self.wait_for_stage(self.app, pid, 'modeling')
        # set items in queue
        self.set_q_with_n_items(pid, 1, self.app)

        # Make sure one (any) q item finishes
        id_to_predict = self.wait_for_q_item_to_complete(self.app, pid=pid, qid=None,
                                                         timeout=30)

        # Get models
        leaderboard = self.get_models(self.app, pid)
        leaderboard_item = leaderboard[0]
        lid = leaderboard_item['_id']
        self.logger.info('Predict with lid:%s: %s', lid, leaderboard_item['model_type'])

        ## 2. Now lets make an actual prediction
        with self.api_app as c_api:
            self.logger.info('c_api: %r', c_api)

            api_token = self.get_api_token(c_api, username, pwd)

            response = self.post_prediction(c_api, pid, lid, username, api_token)
            self.assertEqual(response.status_code, 200, response.data)
            out = json.loads(response.data)
            self.assertEqual(out.get('model_id'), lid, out)
            self.assertEqual(out['task'], 'Binary')
            self.assertIn('prediction', out['predictions'][0])
            self.assertIn('row_id', out['predictions'][0])

            with open(self.new_test_file, 'rb') as test_file:
                response = self.post_prediction(
                    c_api, pid, lid, username, api_token, data=test_file.read(),
                    headers={'content-type': 'text/csv; charset=utf8'})

                self.assertEqual(response.status_code, 200, response.data)
                out = json.loads(response.data)
                self.assertEqual(out.get('model_id'), lid, out)
                self.assertEqual(out['task'], 'Binary')
                preds = [p['class_probabilities']['1'] for p in out['predictions']]
                # if this test fails on decimal 1 then please update decimal
                np.testing.assert_almost_equal(np.sum(preds), 63.326, decimal=2)

            with open(self.new_test_file, 'rb') as test_file:
                df = pd.read_csv(test_file)
                df_json = df.iloc[:2].to_json(orient='records')
                headers = {'content-type': 'application/json; charset=utf8'}
                response = self.post_prediction(
                    c_api, pid, lid, username, api_token, data=df_json,
                    headers=headers)

                self.assertEqual(response.status_code, 200, response.data)
                out = json.loads(response.data)
                self.assertEqual(out['model_id'], lid)
                self.assertEqual(out['task'], 'Binary')
                self.assertEqual(len(out['predictions']), 2)
                preds = [p['class_probabilities']['1'] for p in out['predictions']]
                np.testing.assert_array_almost_equal(preds, [0.001, 0.999])
                    # if this test fails on decimal 1 then please update decimal
                np.testing.assert_almost_equal(np.sum(preds), 1.0, decimal=2)

            ## Test a unicode request
            df_ = df.iloc[0].to_dict()
            df_['WheelType'] = u'Ällöy'
            print('########### %r' % df_)
            df_json = json.dumps([df_])
            headers = {'content-type': 'application/json; charset=utf8'}
            response = self.post_prediction(
                c_api, pid, lid, username, api_token, data=df_json,
                headers=headers)
            self.assertEqual(response.status_code, 200, response.data)
            out = json.loads(response.data)
            self.assertEqual(out['model_id'], lid)
            self.assertEqual(out['task'], 'Binary')
            self.assertEqual(len(out['predictions']), 1)

            ## Test empty data as input
            headers = {'content-type': 'application/json; charset=utf8'}
            response = self.post_prediction(
                c_api, pid, lid, username, api_token, data='[]',
                headers=headers)
            self.assertEqual(response.status_code, 400, response.data)
            out = json.loads(response.data)
            self.assertRegexpMatches(out['status'], 'empty JSON input')
            self.assertEqual(out['code'], 400)

            ## Test a missing column
            df_ = df.iloc[:1]
            df_.pop('WheelType')
            df_json = df_.to_json(orient='records')
            headers = {'content-type': 'application/json; charset=utf8'}
            response = self.post_prediction(
                c_api, pid, lid, username, api_token, data=df_json,
                headers=headers)
            self.assertEqual(response.status_code, 400, response.data)
            out = json.loads(response.data)
            self.assertEqual(out['status'], 'Column WheelType is missing')
            self.assertEqual(out['code'], 400)

            ## Test a request with bad authorization
            url = '/v1/%s/%s/predict' % (pid, lid)
            bad_authorization = ('Basic ' + base64.b64encode(username + ":" +
                                                             ''.join(reversed(api_token))))
            headers = {'content-type': 'application/json; charset=utf8',
                       'Authorization': bad_authorization}
            response = c_api.post(url, headers=headers, data=df_json)
            self.assertEqual(response.status_code, 403)

            ## Test a request from another user
            another_username = 'foobar@example.com'
            user_service = UserService(another_username)
            another_password = '4HQujem1A$E12$78'
            user_service.create_account(another_password)
            authorization = ('Basic ' + base64.b64encode(another_username + ":" +
                                                         another_password))

            api_token_resp = c_api.post('/v1/api_token',
                                        headers={'Authorization': authorization})
            self.assertEqual(api_token_resp.status, '200 OK')
            another_api_token = json.loads(api_token_resp.data)['api_token']

            # authentication for another user but project of user
            authorization = ('Basic ' + base64.b64encode(another_username + ":" +
                                                         another_api_token))
            url = '/v1/%s/%s/predict' % (pid, lid)
            headers = {'content-type': 'application/json; charset=utf8',
                       'Authorization': authorization}
            response = c_api.post(url, headers=headers, data=df_json)
            self.assertEqual(response.status_code, 403)

    def check_upload_data_and_refresh_model(self):
        self.set_dataset('kickcars-sample-200.csv', 'IsBadBuy')
        ## 1. Fit a model so that we have something to predict
        # upload file
        username = self.registered_user['username']
        pwd = self.registered_user['password']

        pid = self.create_project()
        self.wait_for_stage(self.app, pid, 'modeling')
        # set items in queue
        self.set_q_with_n_items(pid, 1, self.app)

        # Make sure one (any) q item finishes
        id_to_predict = self.wait_for_q_item_to_complete(self.app, pid=pid, qid=None,
                                                         timeout=30)

        # Get models
        leaderboard = self.get_models(self.app, pid)
        leaderboard_item = leaderboard[0]
        lid = leaderboard_item['_id']
        self.logger.info('Selected Model lid:%s: %s', lid, leaderboard_item['model_type'])

        with self.api_app as c_api:
            self.logger.info('c_api: %r', c_api)

            api_token = self.get_api_token(c_api, username, pwd)

            ## 2. predict on orignial model
            response = self.post_prediction(c_api, pid, lid, username, api_token)
            predict1 = json.loads(response.data)

            ## 3. upload new data via the api
            filename = self.path_to_test_file('kickcars-sample-400.csv')
            response = self.post_data(c_api, pid, username, api_token, filename)

            self.assertEqual(response.status_code, 200)
            out = json.loads(response.data)
            self.assertEqual(out, {'success':'1'})
            ds = DatasetServiceBase(pid, persistent=self.persistent)
            out = ds.query({'api_upload':True})
            self.assertEqual(len(out),1)
            name = out[0]['originalName']
            self.assertEqual(os.path.basename(name), 'kickcars-sample-400.csv')

            ## 4. refresh the model on new data
            response = self.post_model_refresh(c_api, pid, lid, username, api_token)

            self.assertEqual(response.status_code, 200)
            out = json.loads(response.data)
            self.assertEqual(out, {'Status':'Confirmed'})
            id_to_predict = self.wait_for_q_item_to_complete(self.app, pid=pid, qid=None, timeout=30)

            l = self.persistent.read({'_id':ObjectId(lid)}, table='leaderboard', result={})
            self.assertTrue('build_id' in l)

            m = self.persistent.read({'_id':l['build_id']}, table='model_refresh', result={})
            self.assertTrue('build_datetime' in m)
            self.assertEqual(l['_id'], m['lid'])

            ## 5. predict on refreshed model
            response = self.post_prediction(c_api, pid, lid, username, api_token)
            predict2 = json.loads(response.data)

            self.assertEqual(len(predict1['predictions']), len(predict2['predictions']))
            self.assertNotEqual(predict1['predictions'], predict2['predictions'])


    @patch.dict(EngConfig, {'WORKER_MODEL_CACHE': 0})
    @patch.dict(EngConfig, {'PREDICTION_API_COMPUTE': False})
    def test_predict_transient_smoke_old(self):
        """Test the prediction API in the old server setting.

        Requests are sent to the secure worker processes and predictions
        are transmitted via the data base.
        """
        self.check_predict_transient_smoke()

    @patch.dict(EngConfig, {'WORKER_MODEL_CACHE': 3})
    @patch.dict(EngConfig, {'PREDICTION_API_COMPUTE': True})
    def test_predict_transient_smoke_new(self):
        """Tests the prediction api in the new server setting.

        Prediction API requests are directly handled in the web server (gunicorn)
        processes.
        """
        self.check_predict_transient_smoke()

    @patch.dict(EngConfig, {'WORKER_MODEL_CACHE': 0})
    @patch.dict(EngConfig, {'PREDICTION_API_COMPUTE': False})
    @patch.dict(EngConfig, {'MODEL_REFRESH_API': True})
    def test_upload_data_and_refresh_model_old(self):
        self.check_upload_data_and_refresh_model()

    @patch.dict(EngConfig, {'WORKER_MODEL_CACHE': 3})
    @patch.dict(EngConfig, {'PREDICTION_API_COMPUTE': True})
    @patch.dict(EngConfig, {'MODEL_REFRESH_API': True})
    def test_upload_data_and_refresh_model_new(self):
        self.check_upload_data_and_refresh_model()






if __name__ == '__main__':
    unittest.main()
