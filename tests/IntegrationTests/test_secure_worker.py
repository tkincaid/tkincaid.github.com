import os
import unittest
import json
import time

from integration_test_base import IntegrationTestBase
from config.engine import EngConfig
from ModelingMachine.engine.secure_worker import SecureWorker
from ModelingMachine.engine.worker_request import WorkerRequest

class TestSecureWorker(IntegrationTestBase):

    @classmethod
    def setUpClass(self):
        super(TestSecureWorker, self).setUpClass()
        TestSecureWorker.pid = None

    def setUp(self):
        self.set_dataset('kickcars-sample-200.csv', 'IsBadBuy')
        if not TestSecureWorker.pid:
            TestSecureWorker.pid = self.create_project()
            TestSecureWorker.uid = self.registered_user['uid']

        self.pid = TestSecureWorker.pid

    def test_ping(self):
        self.ping_worker(EngConfig['SECURE_WORKER_CLIENT_BROKER'], 'secure_worker')

    def test_report_error(self):
        pid = self.pid
        # from the base
        profile = self.get_profile()
        uid = profile['id']
        request = {
            'pid' : pid,
            'uid' : uid,
            'lid' : pid,
            'qid' : '1',
            'command': 'does-not-matter',
            'samplepct': 64,
            'partitions': [(0,1)]
        }
        request = WorkerRequest(request)

        worker = SecureWorker(request, None)
        worker.worker_id = "1"
        result = False
        try:
            raise Exception('BOOM!')
        except Exception, e:
            result = worker.report_error(e)

        self.assertTrue(result)

    def test_prediction(self):
        pid = self.upload_file_and_select_response(self.app)
        q_item, leaderboard_item = self.execute_fit(pid)

        lid = leaderboard_item['_id']

        #Submit item to predict
        item_to_predict = q_item
        self.assertIsNotNone(item_to_predict)

        scoring_dataset_id = self.upload_new_file(self.app, pid)
        data = {'predict':1, 'scoring_dataset_id' : scoring_dataset_id, 'lid':lid, 'new_lid':False, 'partition_stats':{str((0,-1)):'whatever'}}
        item_to_predict.update(data)

        not_started = self.add_request_to_queue(pid, item_to_predict, lid)
        self.assertEqual(len(not_started), 1)

        predict_item = not_started.pop()

        self.assertEqual(predict_item['dataset_id'], item_to_predict['dataset_id'])
        self.assertEqual(predict_item['pid'], item_to_predict['pid'])

        # Kick off new work and wait for the prediciton to finish
        self.wait_for_q_item_to_complete(self.app, pid = pid, qid = str(predict_item['qid']), timeout = 35)

        # Download prediction
        response = self.app.get('/predictions/%s/%s' % (lid, scoring_dataset_id))
        self.assertEqual(response.status_code, 200,  'Could not find predictions for lid: {0} and dataset_id: {1}. Server responded: {2}'.format(lid, scoring_dataset_id, response.data))

    def test_different_dataset(self):
        pid = self.upload_file_and_select_response(self.app)
        self.wait_for_stage(self.app, pid, ['post-aim', 'eda2', 'modeling'])
        original_q_item = self.set_q_with_simple_blueprint(pid)

        dataset_id = original_q_item['dataset_id']
        response = self.app.get('/project/{0}/dataset/{1}'.format(pid, dataset_id))
        self.assertEqual(response.status_code, 200)
        dataset = json.loads(response.data)['data']

        raw_features = {fx['name'] for fx in dataset['columns']}

        coef, leaderboard_item = self.best_coef(pid, raw_features, None, timeout=30)
        self.logger.info("Best coefficient: %s" % coef)
        self.assertIsNotNone(leaderboard_item)

        # Create new feature list without best coef
        new_features = [feature for feature in dataset['columns']
                        if feature['name'] != coef]
        self.assertNotEqual(len(dataset['columns']), len(new_features),
                            '%s not found in original dataset' % coef)
        request = {
            'name': 'testWithoutBestCoef',
            'columns': new_features
        }
        response = self.app.post('/project/%s/dataset' % pid,
                          content_type="application/json",
                          data=json.dumps(request))

        self.assertEqual(response.status_code, 200)
        new_dataset_id = json.loads(response.data)
        self.assertEqual(len(new_dataset_id), 24)

        new_q_item = {
            'blueprint_id' : 'ebac339655455b45d66aa89bd9f8508f',
            'dataset_id' : new_dataset_id,
            'model_type' : 'Simple stuff',
            'samplepct' : 100,
            'partitions': ([-1,-1],),
            'bp' : 1,
            'max_reps' : 1,
            'features' : '',
            'pid' : pid,
            'uid' : original_q_item['uid'],
            'blueprint' : {'1':[['NUM'],['NI'],'T'], '2':[['1'],['GLMB'],'P']}
        }

        not_started = self.add_request_to_queue(pid,[new_q_item])

        self.assertGreaterEqual(len(not_started), 1)

        new_q_item = not_started.pop()

        self.assertEqual(new_q_item['dataset_id'],
                         new_dataset_id)

        self.assertEqual(new_q_item['pid'],
                         original_q_item['pid'])

        self.wait_for_q_item_to_complete(self.app, pid=pid,
                                         qid=new_q_item['qid'],
                                         timeout=35)

        new_coef, leaderboard_item = self.best_coef(pid, raw_features, new_dataset_id, timeout=10)
        self.assertEqual(leaderboard_item['dataset_id'], new_dataset_id)
        self.logger.info("New best coefficient: %s" % new_coef)
        self.assertNotEqual(coef, new_coef)

    def best_coef(self, pid, raw_features, dataset_id, timeout):
        def untransform(var):
            # dirty way of untransforming
            # kickcars has only transformed features
            if '-' in var:
                var = var[:var.index('-')]
            elif var.startswith('logit_'):
                var = var.split('logit_')[0]
            return var

        # If no best_coef is found, it's because this is being run
        # on models that only show coeffs of transformed vars
        # Try incrasing set_q_with_n_items
        lids_checked = {}
        coef_var = ''
        coef_max = 0
        coef_item = None
        for i in range(10):
            leaderboard = self.get_models(self.app, pid)
            for leaderboard_item in leaderboard:
                if dataset_id and str(leaderboard_item['dataset_id']) \
                        != str(dataset_id):
                    continue
                leaderboard_item = self.get_model(pid, leaderboard_item['_id'])
                for key in leaderboard_item.get('extras', {}):
                    coeffs = leaderboard_item['extras'][key]\
                        .get('coefficients')
                    if not coeffs:
                        continue
                    for var, weight in coeffs:
                        var = untransform(var)
                        if var not in raw_features:
                            continue  # tranformed, we don't want these
                        weight = abs(weight)
                        if weight > coef_max:
                            coef_max = weight
                            coef_var = var
                            coef_item = leaderboard_item
            if coef_max:
                break
            self.wait_for_q_item_to_complete(self.app, pid=pid, qid=None,
                                             timeout=timeout)
            timeout = int(timeout/2)
        return coef_var, coef_item

if __name__ == '__main__':
    unittest.main()
