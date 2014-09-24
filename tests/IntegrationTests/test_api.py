from bson import ObjectId

import config.test_config
from common.wrappers import database
from common.api.api_client import APIClient
from config.engine import EngConfig
from tests.IntegrationTests.integration_test_base import IntegrationTestBase

import os
import tarfile
import tempfile

class IntegrationTestAPI(IntegrationTestBase):
    '''
        Tests both: the Web API and the API client (no mocks)
    '''

    @classmethod
    def setUpClass(self):
        super(IntegrationTestAPI, self).setUpClass()
        self.api_client = APIClient(EngConfig['WEB_API_LOCATION'])
        IntegrationTestAPI.pid = None

    @classmethod
    def tearDownClass(self):
        self.persistent.conn.Default.drop()

    def setUp(self):
        super(IntegrationTestAPI, self).setUp()
        self.set_dataset('kickcars-sample-200.csv', 'IsBadBuy')
        if not IntegrationTestAPI.pid:
            IntegrationTestAPI.pid = self.create_project()
            IntegrationTestAPI.uid = self.registered_user['uid']
            self.logout(self.app)

        self.pid = IntegrationTestAPI.pid

    def test_save_leaderboard_item(self):
        l_item = {'uid' : self.uid, 'pid' : self.pid, 'token' : str(ObjectId())}

        lid = self.persistent.create(table = 'leaderboard', values = l_item)
        l_item['_id'] = str(lid)

        response = self.api_client.save_leaderboard_item(lid, l_item)
        self.assertTrue(response)

        result = self.persistent.read(table = 'leaderboard',
            condition = {'_id' : lid}, result = {})

        self.assertItemsEqual(result, l_item)

    def test_save_leaderboard_item_with_ace(self):
       
        l_item = {'uid': self.uid, 'pid': self.pid, 'token': str(ObjectId())}

        lid = self.persistent.create(table = 'leaderboard', values = l_item)

        l_item['_id'] = str(lid)

        l_item.update({'var_imp_info': 0.1453, 'var_imp_var': 'Feature X'})

        response = self.api_client.save_leaderboard_item(lid, l_item)
        self.assertTrue(response)

        result = self.persistent.read(table = 'eda',
            condition = {'pid' : ObjectId(self.pid)}, result = {})

        self.assertEqual(result['eda']['Feature X']['profile']['info'], 0.1453)


    def test_create_leaderboard_item(self):
        l_item = {'uid' : self.uid, 'pid' : self.pid}

        response = self.api_client.create_leaderboard_item(l_item)
        self.assertTrue(response)

        response.pop('_id')

        self.assertItemsEqual(l_item, response)

    def test_save_predictions(self):
        predictions = {
            'uid' : self.uid,
            'pid' : self.pid,
            'dataset_id' : str(ObjectId()),
            'lid' : str(ObjectId())
            }

        predictions_id = self.persistent.create(table = 'predictions',
            values = predictions)

        predictions.pop('_id')

        response = self.api_client.save_predictions(predictions)
        self.assertTrue(response)

        result = self.persistent.read(table = 'predictions',
            result={})

        result.pop('_id')
        predictions['lid'] = str(predictions_id)

        self.assertItemsEqual(result, predictions)

    def test_save_ide(self):
        tf = tempfile.NamedTemporaryFile(mode='w', delete=False)
        tf.close()

        tar = tarfile.open(tf.name, "w:gz")
        tar.close()

        response = self.api_client.save_ide(tf.name, self.uid, self.pid)
        self.assertTrue(response)

        os.remove(tf.name)

    def test_get_ide_url(self):
        response = self.api_client.get_ide_url(self.uid, self.pid)
        self.assertIsNone(response)

    def test_report_complete_stores_instance_id(self):
        data = {'pid': self.pid,
                'qid': 'test',
                'uid': self.uid,
                'lid': 'test',
                'worker_id': 'test',
                'command': 'foo',
                'start_time': 'test',
                'end_time': 'test',
                'instance_id': 'test_instance_id'}
        result = self.api_client.report_complete(data)
        query = self.persistent.read(table='request_tracker', 
                condition={'pid': self.pid}, result=[])
        self.assertEqual(len(query),1)
        self.assertEqual(query[0]['instance_id'], 'test_instance_id')
