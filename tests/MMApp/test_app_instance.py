import unittest
import json
from mock import patch
from config.engine import EngConfig
from tests.mocks.mock_instance import MockInstanceService
from MMApp.entities.user import UserModel
from MMApp.entities.instance import InstanceModel
from bson import ObjectId

class TestInstanceBlueprint(unittest.TestCase):
    def setUp(self):
        self.addCleanup(self.stopPatching)
        self.app = None
        with patch.dict(EngConfig, {'TEST_MODE': True }, clear = False):
            import MMApp.app
            self.app = MMApp.app.app.test_client()

        self.patchers = []
        instance_service_patch = patch('MMApp.app_instance.InstanceService')
        self.MockInstanceService = instance_service_patch.start()
        self.MockInstanceService.side_effect = MockInstanceService
        self.patchers.append(instance_service_patch)

        get_user_session_patch = patch('MMApp.app_instance.get_user_session')
        self.mock_get_user_session = get_user_session_patch.start()
        self.patchers.append(get_user_session_patch)

        role_provider_patch = patch('MMApp.app_instance.RoleProvider')
        role_provider_patch.start()
        self.patchers.append(role_provider_patch)

        mock_user = {
            'uid': ObjectId(),
            'username':'project-owner@datarobot.com'
        }

        self.mock_get_user_session.return_value = UserModel(**mock_user)

    def stopPatching(self):
        super(TestInstanceBlueprint, self).tearDown()
        for patcher in self.patchers:
            if patcher:
                patcher.stop()

    def test_get_all_instances(self):
        response = self.app.get('/instance')
        self.assertEqual(response.status_code, 200, response.data)
        instances = json.loads(response.data)
        self.assertIsInstance(instances, list)
        self.assertGreater(len(instances), 1, instances)

    def test_get_instance(self):
        instance_id = '5390d734637aba560282102e'
        response = self.app.get('/instance/{}'.format(instance_id))
        self.assertEqual(response.status_code, 200)
        instances = json.loads(response.data)
        self.assertEqual(instance_id, instances['_id'])


    def test_get_instance(self):
        response = self.app.get('/instance/types')
        self.assertEqual(response.status_code, 200)
        instance_types = json.loads(response.data)
        self.assertIsInstance(instance_types, list)

    def test_launch_instance(self):
        payload = {
            'type': 'x3.small',
            'resource': 'prediction'
        }

        response = self.app.post('/instance/launch', content_type='application/json',
            data=json.dumps(payload))

        self.assertEqual(response.status_code, 200)

    def test_terminate(self):
        response = self.app.post('/instance/{}/terminate'.format(ObjectId()))
        self.assertEqual(response.status_code, 200)
        instance = json.loads(response.data)
        self.assertIsNotNone(instance['_id'])

    def test_start(self):
        response = self.app.post('/instance/{}/start'.format(ObjectId()))
        self.assertEqual(response.status_code, 200)
        instance = json.loads(response.data)
        self.assertIsNotNone(instance['_id'])

    def test_stop(self):
        response = self.app.post('/instance/{}/stop'.format(ObjectId()))
        self.assertEqual(response.status_code, 200)
        instance = json.loads(response.data)
        self.assertIsNotNone(instance['_id'])
