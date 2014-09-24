import unittest
from flask import Response
from mock import patch, call, Mock, MagicMock
from bson import ObjectId
import json

from MMApp.entities.user import UserModel
from config.engine import EngConfig
from werkzeug.exceptions import RequestEntityTooLarge, BadRequestKeyError

class TestUploadServer(unittest.TestCase):

    def setUp(self):
        self.addCleanup(self.stopPatching)

        self.app = None
        with patch.dict(EngConfig, {'TEST_MODE': True }, clear = False):
            import MMApp.upload_server
            import MMApp.app_upload
            self.app = MMApp.upload_server.app.test_client()
            self.progress = MMApp.app_upload.progress

        self.patchers = []

        get_user_session_patch = patch('MMApp.app_upload.get_user_session')
        self.mock_get_user_session = get_user_session_patch.start()
        self.patchers.append(get_user_session_patch)
        mock_user = {
            'uid': ObjectId(),
            'username':'project-owner@datarobot.com'
        }
        self.mock_get_user_session.return_value = UserModel(**mock_user)

        mock_user_service = patch('MMApp.app_upload.UserService')
        self.MockUserService = mock_user_service.start()
        self.patchers.append(mock_user_service)

        ProjectService = patch('MMApp.app_upload.ProjectService')
        self.MockProjectService = ProjectService.start()
        self.patchers.append(ProjectService)

        mock_dataset_service = patch('MMApp.app_upload.DatasetService')
        self.MockDatasetService = mock_dataset_service.start()
        self.patchers.append(mock_dataset_service)

        mock_file_transaction = patch('MMApp.app_upload.FileTransaction')
        self.MockFileTransaction = mock_file_transaction.start()
        self.patchers.append(mock_file_transaction)

        mock_io = patch('MMApp.app_upload.io')
        self.MockIO = mock_io.start()
        self.patchers.append(mock_io)

        mock_time = patch('MMApp.app_upload.time.sleep')
        mock_time.start()
        self.patchers.append(mock_time)

    def stopPatching(self):
        super(TestUploadServer, self).tearDown()
        for patcher in self.patchers:
            if patcher:
                patcher.stop()

    @patch('MMApp.app_upload.os')
    @patch('MMApp.app_upload.request')
    def test_file_upload(self, MockRequest, MockOs):
        self.MockIO.inspect_uploaded_file.return_value = (Mock(), Mock())
        response = self.app.post('/upload/{}',format(ObjectId()))
        self.assertEquals(response.status_code, 200)


    def test_upload_url(self):
        with self.app as c:
            with patch('MMApp.app_upload.process_upload') as upload_data_mock:
                upload_data_mock.return_value = Response('{}', 200, mimetype='application/json')

                payload = {
                    'url': 'http://someurl/',
                    'pid' : str(ObjectId())
                }

                response = c.post('/upload/url', data='{}',
                    content_type='application/json')
                self.assertEquals(response.status_code, 400)

                response = c.post('/upload/url', data=json.dumps(payload),
                    content_type='application/json')
                self.assertEquals(response.status_code, 200)

    def test_file_upload_fails_and_deletes_project(self):
        project_service = self.MockProjectService.return_value
        response = self.app.post('/upload/{}'.format(ObjectId()))
        self.assertEquals(response.status_code, 400)

        self.assertTrue(project_service.delete_project.called)
        self.assertIn('did not contain any file', response.data)

    def test_upload_url_fails_and_notifies_the_right_user(self):
        pid = str(ObjectId())
        payload = {
            'url': 'http://someurl/',
            'pid' : pid
        }

        dataset_service = self.MockDatasetService.return_value
        dataset_service.process_url_upload.side_effect = Exception('BOOM!')

        self.progress.set_ids = Mock()

        response = self.app.post('/upload/url', data=json.dumps(payload),
            content_type='application/json')

        self.progress.set_ids.assert_called_once_with(pid)
        self.assertIn('Could not upload file', response.data)

    def test_progress_ids_is_reset_after_each_request(self):

        # Set ids
        self.progress.pid = 1
        self.progress.qid = 2
        self.progress.lid = 3

        # Make a random request
        response = self.app.post('/upload/url', data={}, content_type='application/json')

        # Make sure ids are reset
        self.assertIsNone(self.progress.pid)
        self.assertIsNone(self.progress.qid)
        self.assertIsNone(self.progress.lid)

    def test_upload_prediction(self):
        with patch('MMApp.app_upload.process_upload') as upload_data_mock:
            upload_data_mock.return_value = Response('{}', 200, mimetype='application/json')

            pid = str(ObjectId())
            response = self.app.post('/upload/{}?is_prediction=1'.format(pid))
            self.assertEquals(response.status_code, 200)

            upload_data_mock.assert_has_call(call(pid, True))

    def test_ping(self):
        token = str(ObjectId())
        response = self.app.get('/upload/ping?token=' + token)
        self.assertEqual(response.status_code, 200)
        resp_data = json.loads(response.data)
        self.assertEqual(resp_data['response'], 'pong')
        self.assertEqual(resp_data['token'], token)

    @patch('MMApp.app_upload.request')
    def test_max_upload_file_size_exceeeded(self, MockRequest):
        MockRequest.files = MagicMock()
        MockRequest.files.__getitem__.side_effect = RequestEntityTooLarge()

        response = self.app.post('/upload/{}'.format(ObjectId()))

        self.assertEquals(response.status_code, 400)
        self.assertIn('file uploaded is greater than', response.data)

    def test_eda_column_names_message(self):
        from MMApp.app_upload import eda_column_names_message
        total_columns = 10
        chunk_size = 2

        columns = range(0, total_columns)

        for i, sliced_columns in enumerate(eda_column_names_message(columns, chunk_size = chunk_size)):
            self.assertEqual(len(sliced_columns), chunk_size)

            self.assertEqual(sliced_columns[0]['name'], i * chunk_size)
            self.assertEqual(sliced_columns[1]['name'], i * chunk_size + 1)
