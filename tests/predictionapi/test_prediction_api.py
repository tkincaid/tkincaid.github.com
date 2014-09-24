import unittest
import base64
import pytest
import json

from mock import patch

from predictionapi import prediction_api
from predictionapi.prediction_io import UserTopkResponse
from predictionapi.prediction_io import UserTopkRecommendation
from predictionapi.prediction_io import JsonResponseSerializer
from predictionapi.entities.prediction import PredictionServiceUserError


def get_auth_header(username, password):
    auth_hash = base64.standard_b64encode('%s:%s' % (username, password))
    return ('Authorization', 'Basic %s' % auth_hash)


@pytest.mark.unit
class TestPredictionApi(unittest.TestCase):
    def setUp(self):
        self.app = prediction_api.app.test_client()

    def test_all_require_basic_auth(self):
        # TODO: Write a test using url_map to verify that all
        # endpoints use basic auth
        pass

    def test_GET_api_token(self):
        with self.app as c:
            # No authorization sent
            response = c.get('/v1/api_token')
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.content_type, 'application/json')
            self.assertTrue('Authorization required' in response.data)

            # Error if username is not an email address
            response = c.get('/v1/api_token',
                             headers=[get_auth_header('notanemail', 'password')])
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.content_type, 'application/json')
            self.assertTrue('Invalid email address' in response.data)

            # Invalid credentials
            with patch('predictionapi.prediction_api.UserService') as mock_UserService:
                mock_UserService.return_value.login.return_value = False
                response = c.get('/v1/api_token',
                                headers=[get_auth_header('user@example.com', 'invalidpassword')])
                self.assertEqual(response.status_code, 401)
                self.assertEqual(response.content_type, 'application/json')
                self.assertTrue('Invalid credentials' in response.data)

            # Valid credentials
            with patch('predictionapi.prediction_api.UserService') as mock_UserService:
                mock_UserService.return_value.login.return_value = True
                mock_UserService.return_value.get_api_token.return_value = "5gIf_eVEpQUK2xs8SE5v6PujTTbyPCjd"
                response = c.get('/v1/api_token',
                                headers=[get_auth_header('user@example.com', 'validpassword')])
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content_type, 'application/json')
                self.assertTrue('5gIf_eVEpQUK2xs8SE5v6PujTTbyPCjd' in response.data)
                print response.data

    def test_POST_api_token(self):
        with self.app as c:
            # No authorization sent
            response = c.post('/v1/api_token')
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.content_type, 'application/json')
            self.assertTrue('Authorization required' in response.data)

            # Error if username is not an email address
            response = c.post('/v1/api_token',
                             headers=[get_auth_header('notanemail', 'password')])
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.content_type, 'application/json')
            self.assertTrue('Invalid email address' in response.data)

            # Invalid credentials
            with patch('predictionapi.prediction_api.UserService') as mock_UserService:
                mock_UserService.return_value.login.return_value = False
                response = c.post('/v1/api_token',
                                headers=[get_auth_header('notauser@example.com', 'invalidpassword')])
                self.assertEqual(response.status_code, 401)
                self.assertEqual(response.content_type, 'application/json')
                self.assertTrue('Invalid credentials' in response.data)

            # Valid credentials
            with patch('predictionapi.prediction_api.UserService') as mock_UserService:
                mock_UserService.return_value.login.return_value = True
                mock_UserService.return_value.create_api_token.return_value = "1C6IPAaO7GUp88jBKqn4XpSTCxeNEhje"
                response = c.post('/v1/api_token',
                                headers=[get_auth_header('user@example.com', 'validpassword')])
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content_type, 'application/json')
                self.assertTrue('1C6IPAaO7GUp88jBKqn4XpSTCxeNEhje' in response.data)
                print response.data


@pytest.mark.unit
@patch('predictionapi.prediction_api.RoleProvider')
@patch('predictionapi.prediction_api._validate_api_token')
@patch('predictionapi.prediction_api._get_user_service')
class TestPredictionQuickFail(unittest.TestCase):
    def setUp(self):
        pid = '5223deadbeefdeadbeef1234'
        lid = '5123abbbcaaadbbbaeee0000'
        self.url = '/v1/{}/{}/predict'.format(pid, lid)

    def test_empty_json_stream_raises_error(self, mock_get_user_service,
                                            mock_validation,
                                            MockRoleProvider):
        with patch.dict(prediction_api.EngConfig,
                        {'PREDICTION_API_COMPUTE': False},
                        clear=False):
            with prediction_api.app.test_client() as c:
                request_data = ''
                content_type = 'application/json'
                headers=[get_auth_header('notanemail', 'password')]
                response = c.post(self.url, data=request_data,
                                  content_type=content_type,
                                  headers=headers)
            self.assertEqual(response.status_code, 400)
            self.assertIn('No data was received', response.data)

    def test_empty_json_stream_raises_error_new(self, mock_get_user_service,
                                                mock_validation,
                                                MockRoleProvider):
        with patch.dict(prediction_api.EngConfig,
                        {'PREDICTION_API_COMPUTE': True},
                        clear=False):
            with prediction_api.app.test_client() as c:
                request_data = ''
                content_type = 'application/json'
                headers=[get_auth_header('notanemail', 'password')]
                response = c.post(self.url, data=request_data,
                                  content_type=content_type,
                                  headers=headers)
                print response.data
            self.assertEqual(response.status_code, 400)
            self.assertIn('No data was received', response.data)


@pytest.mark.unit
@patch('predictionapi.prediction_api.RoleProvider')
@patch('predictionapi.prediction_api._validate_api_token')
@patch('predictionapi.prediction_api._get_user_service')
@patch('predictionapi.prediction_api.UserTopkService')
class TestUserTopkRecommendations(unittest.TestCase):
    """Test UserTopkRecommendation route by mocking the service class. """

    def setUp(self):
        self.pid = '5223deadbeefdeadbeef1234'
        self.lid = '5123abbbcaaadbbbaeee0000'
        self.url = '/v1/{}/{}/predict_user_topk'.format(self.pid, self.lid)

    def test_smoke(self, MockTopkSvcClass,
                                 mock_get_user_service,
                                 mock_validation,
                                 MockRoleProvider):
        """Smoke test if everything is connected correctly. """
        pred = UserTopkResponse(self.lid, 'Regression', 1, [])
        svc_instance = MockTopkSvcClass.return_value
        svc_instance.predict.return_value = pred
        with prediction_api.app.test_client() as c:

            request_data = '{"user_id": [1], "known_items": true}'
            content_type = 'application/json'
            headers=[get_auth_header('notanemail', 'password')]
            response = c.post(self.url, data=request_data,
                              content_type=content_type,
                              headers=headers)
        self.assertEqual(response.status_code, 200, response)
        self.assertEqual(response.content_type, 'application/json')

        encoder = JsonResponseSerializer('v1')
        self.assertEqual(response.data, encoder.serialize_success(pred, 200))

    def test_ok(self, MockTopkSvcClass,
                                 mock_get_user_service,
                                 mock_validation,
                                 MockRoleProvider):
        """Test with a list of topk recommendations. """
        preds = [UserTopkRecommendation(1, ['i0', 'i1'], [4.2, 3.8]),
                 UserTopkRecommendation(2, ['i1', 'i0'], [3.2, 1.7]),
                 UserTopkRecommendation(3, ['i0', 'i1'], [4.9, 2.8]),
                 UserTopkRecommendation(4, ['i0', 'i2'], [5.2, 3.8])]
        pred = UserTopkResponse(self.lid, 'Regression', 1, preds)
        svc_instance = MockTopkSvcClass.return_value
        svc_instance.predict.return_value = pred
        with prediction_api.app.test_client() as c:

            request_data = '{"user_id": [1, 2, 3, 4], "known_items": true}'
            content_type = 'application/json'
            headers=[get_auth_header('notanemail', 'password')]
            response = c.post(self.url, data=request_data,
                              content_type=content_type,
                              headers=headers)
        self.assertEqual(response.status_code, 200, response)
        self.assertEqual(response.content_type, 'application/json')

        encoder = JsonResponseSerializer('v1')
        self.assertEqual(response.data, encoder.serialize_success(pred, 200))

    def test_predict_raise_error(self, MockTopkSvcClass,
                                 mock_get_user_service,
                                 mock_validation,
                                 MockRoleProvider):
        pred = UserTopkResponse(self.lid, 'Regression', 1, [])
        svc_instance = MockTopkSvcClass.return_value
        svc_instance.predict.side_effect = PredictionServiceUserError('foobar')
        with prediction_api.app.test_client() as c:

            request_data = '{"user_id": [1], "known_items": true}'
            content_type = 'application/json'
            headers=[get_auth_header('notanemail', 'password')]
            response = c.post(self.url, data=request_data,
                              content_type=content_type,
                              headers=headers)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.content_type, 'application/json')

        self.assertEqual(json.loads(response.data), {'status': 'foobar', 'code': 400,
                                                     'version': 'v1'})
