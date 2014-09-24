import unittest
from mock import patch, Mock
from bson import ObjectId
from ModelingMachine.client import MMClient, LARGE_WORKER_NAME, SMALL_WORKER_RAM

import json

class TestMMClient(unittest.TestCase):

    def setUp(self):
        self.addCleanup(self.stopPatching)

        self.patchers = []

        client_patch = patch('ModelingMachine.client.Client')
        self.client_patch = client_patch.start()
        self.patchers.append(client_patch)

        flippers_patch = patch('ModelingMachine.client.FLIPPERS')
        self.MockFlippers = flippers_patch.start()
        self.patchers.append(flippers_patch)

        mock_resource_service = patch('ModelingMachine.client.ResourceService')
        self.MockResourceService = mock_resource_service.start()
        self.patchers.append(mock_resource_service)

        self.MockFlippers.allow_worker_options = True
        self.MockFlippers.enable_resource_estimation = True

    def stopPatching(self):
        super(TestMMClient, self).tearDown()
        for patcher in self.patchers:
            if patcher:
                patcher.stop()


    def test_next_steps(self):
        c = MMClient('datarobot-broker')
        expected_dataset_id = None

        def check_request(service, request_json, worker_id):
            request = json.loads(request_json)
            self.actual_dataset_id = request['dataset_id']

        pid = ObjectId()
        uid = ObjectId()

        c.client.send.side_effect = check_request
        self.MockFlippers.request_accounting = False
        with patch.object(c, 'get_service_name', return_value = 'EDA'):
            c.next_steps(pid = pid, uid = uid, dataset_id = expected_dataset_id)

        self.assertEqual(self.actual_dataset_id, expected_dataset_id)

    def test_get_worker_size_defaults_to_large_worker(self):
        c = MMClient('datarobot-broker')

        resource_service = self.MockResourceService.return_value
        estimates = [None, {}, {'max_RAM':0}, {'max_RAM':None}, {'max_RAM':''}]
        resource_service.estimate_resources.side_effect = estimates
        for i in estimates:
            worker_size = c.get_worker_size(Mock(), Mock())
            self.assertEqual(LARGE_WORKER_NAME, worker_size)

    def test_get_worker_size_with_little_RAM(self):
        c = MMClient('datarobot-broker')
        resource_service = self.MockResourceService.return_value
        resource_service.estimate_resources.return_value = {'max_RAM' : 1}
        worker_size = c.get_worker_size(Mock(), Mock())
        self.assertIsNone(worker_size)

    def test_get_worker_size_with_big_RAM(self):
        c = MMClient('datarobot-broker')
        resource_service = self.MockResourceService.return_value
        resource_service.estimate_resources.return_value = {'max_RAM' : SMALL_WORKER_RAM + 1}
        worker_size = c.get_worker_size(Mock(), Mock())
        self.assertEqual(LARGE_WORKER_NAME, worker_size)

    def test_get_service_name(self):
        c = MMClient('datarobot-broker')
        ds = 'kickcars-sample-200.csv'
        bp = 'blueprint-definition'
        service_id = 'ABC'

        project_info = {
            'worker_options' : {
                'service_id' : service_id
            },
            'originalName': ds
        }

        stats = {
            'dataset':ds,
            'blueprint':bp,
            'stats': {
                'max_RAM': SMALL_WORKER_RAM + 1,
                'total_CPU': 1,
                'max_cores': 1}
         }

        resource_service = self.MockResourceService.return_value
        resource_service.estimate_resources.return_value = stats

        with patch.object(c.persistent, 'read', return_value = project_info):
            request = {'blueprint' : bp, 'pid' : ObjectId()}
            service = 'fit'

            actual_full_service = c.get_service_name(service, request)
            actual_service, actual_worker_size, actual_service_id = actual_full_service.split(' ')

            self.assertEqual(service, actual_service)
            self.assertEqual(LARGE_WORKER_NAME, actual_worker_size)
            self.assertEqual(service_id, actual_service_id)