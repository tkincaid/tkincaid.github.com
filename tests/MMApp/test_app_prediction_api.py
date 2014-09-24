import unittest
import json
from mock import patch
from config.engine import EngConfig
from tests.mocks.mock_instance import MockApiInstanceService
from MMApp.entities.user import UserModel
from MMApp.entities.instance import InstanceModel
from bson import ObjectId

class TestPredictionApiBlueprint(unittest.TestCase):
    def setUp(self):
        self.addCleanup(self.stopPatching)
        self.app = None
        with patch.dict(EngConfig, {'TEST_MODE': True }, clear = False):
            import MMApp.app
            self.app = MMApp.app.app.test_client()

        self.patchers = []
        instance_service_patch = patch('MMApp.app_prediction_api.ApiInstanceService')
        self.MockApiInstanceService = instance_service_patch.start()
        self.MockApiInstanceService.side_effect = MockApiInstanceService
        self.patchers.append(instance_service_patch)

        get_user_session_patch = patch('MMApp.app_prediction_api.get_user_session')
        self.mock_get_user_session = get_user_session_patch.start()
        self.patchers.append(get_user_session_patch)

        project_service_patch = patch('MMApp.app_prediction_api.ProjectService')
        self.MockProjectService = project_service_patch.start()
        self.patchers.append(project_service_patch)

        mock_user = {
            'uid': ObjectId(),
            'username':'project-owner@datarobot.com'
        }

        self.mock_get_user_session.return_value = UserModel(**mock_user)

    def stopPatching(self):
        super(TestPredictionApiBlueprint, self).tearDown()
        for patcher in self.patchers:
            if patcher:
                patcher.stop()

    def test_activate_model(self):
        project_service = self.MockProjectService.return_value
        project_service.get_api_instances_for_model.return_value = {
            '_id' : '538e17ba622b3e6f3c8637ab',
            'api_activated' : 1 ,
            'api_instances' : [
                {
                    '_id': '5383e637ab6f3c8e17ba622b',
                    'activation_status': 3,
                    'activated_on' : '1402346958649'
                }
            ]
        }

        instance_id = 'DEFAULT'
        model_id = '5390d72e637aba560282102d'
        response = self.app.post('/prediction-api/instance/{}/model/{}/activate'.format(instance_id, model_id))
        self.assertEqual(response.status_code, 200)

    def test_deactivate_model(self):
        project_service = self.MockProjectService.return_value
        project_service.get_api_instances_for_model.return_value = {
            '_id' : '538e17ba622b3e6f3c8637ab',
            'api_activated' : 1 ,
            'api_instances' : [
                {
                    '_id': '5383e637ab6f3c8e17ba622b',
                    'activation_status': 3,
                    'activated_on' : '1402346958649'
                }
            ]
        }

        instance_id = 'DEFAULT'
        model_id = '5390d72e637aba560282102d'
        response = self.app.post('/prediction-api/instance/{}/model/{}/deactivate'.format(instance_id, model_id))
        self.assertEqual(response.status_code, 200)

    def test_get_models_for_instance(self):
        instance_id = ObjectId()
        project_service = self.MockProjectService.return_value
        project_service.get_models_for_instance.return_value = [{
           '_id': ObjectId(),
           'dataset_id': ObjectId(),
           'dataset_name': 'Informative Features',
           'features': [],
           'api_instances' : [
                {
                    '_id': instance_id,
                    'activation_status': 3,
                    'activated_on' : '1402346958649'
                }
            ]
        }]

        response = self.app.get('/prediction-api/instance/{}/model/'.format(instance_id))
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIsInstance(data['models'], list)
        self.assertEqual(len(data['models']), 1)
