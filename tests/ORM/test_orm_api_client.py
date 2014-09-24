import unittest
from mock import patch
from bson import ObjectId

from MMApp.entities.instance import InstanceRequestModel
from ORM.orm_api_client import ORMAPIClient
from config.engine import EngConfig
from config.test_config import db_config

class TestORMApiClient(unittest.TestCase):

    def setUp(self):
        self.addCleanup(self.stopPatching)
        self.api = ORMAPIClient(EngConfig['RESOURCE_MANAGER_API_LOCATION'])
        self.json_header = {'content-type': 'application/json'}

        self.patchers = []
        requests_patch = patch('ORM.orm_api_client.requests')
        self.RequestsMock = requests_patch.start()
        self.patchers.append(requests_patch)


    def stopPatching(self):
        super(TestORMApiClient, self).tearDown()
        for patcher in self.patchers:
            if patcher:
                patcher.stop()

    def test_setup_instance(self):
        mock_response = self.RequestsMock.post.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'OK'}

        request = {
            '_id': '5390d734637aba560282102e'
        }

        result = self.api.setup_instance(InstanceRequestModel.from_dict(request))

        self.assertTrue(result)

    def test_start(self):
        mock_response = self.RequestsMock.post.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'OK'}
        instance_id =  ObjectId()
        result = self.api.start(instance_id)
        self.assertTrue(result)

    def test_stop(self):
        mock_response = self.RequestsMock.post.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'OK'}
        instance_id =  ObjectId()
        result = self.api.stop(instance_id)
        self.assertTrue(result)

    def test_terminate(self):
        mock_response = self.RequestsMock.post.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'OK'}
        instance_id =  ObjectId()
        result = self.api.terminate(instance_id)
        self.assertTrue(result)