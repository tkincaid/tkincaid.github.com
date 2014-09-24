####################################################################
#
#       MMApp Unit Tests
#
#       Author: David Lapointe
#
#       Copyright (C) 2013 DataRobot Inc.
####################################################################

import json
import unittest
from collections import OrderedDict

from mock import patch, DEFAULT, call, Mock
import pytest
from flask import Response
from bson import ObjectId

from MMApp.utilities.error import RequestError
import config.app_config as app_config
from config.engine import EngConfig

from MMApp.entities.user import UserModel, UserException
from MMApp.entities.admin import AdminAccessError

@pytest.mark.unit
class TestApp(unittest.TestCase):
    def generate_id(self):
        """ Generate a new ObjectId for tests """
        return str(ObjectId())
    def setUp(self):
        self.addCleanup(self.stopPatching)

        self.patchers = OrderedDict()
        self.app = None
        with patch.dict(EngConfig, {'TEST_MODE': True }, clear = False):
            import MMApp.app
            self.app = MMApp.app.app.test_client()
            self.progress = MMApp.app.progress
            self.patchers['MMApp.app.is_private_mode'] = patch('MMApp.app.is_private_mode')
            self.mock_is_private_mode = self.patchers['MMApp.app.is_private_mode'].start()

        def raise_for_pid(pid=None, *args, **kwargs):
            try:
                ObjectId(pid)
            except:
                raise RequestError(status=400)
            return DEFAULT
        def raise_for_uid_and_pid(uid=None, pid=None, *args, **kwargs):
            try:
                ObjectId(uid)
                ObjectId(pid)
            except Exception, e:
                raise RequestError(status=400)
            return DEFAULT


        self.patchers['MMApp.app.UserService'] = patch('MMApp.app.UserService')
        self.patchers['MMApp.app.IdeService'] = patch('MMApp.app.IdeService')
        self.patchers['MMApp.app.QueueService'] = patch('MMApp.app.QueueService')
        self.patchers['MMApp.app.DatasetService'] = patch('MMApp.app.DatasetService')
        self.patchers['MMApp.app.ProjectService'] = patch('MMApp.app.ProjectService')
        self.patchers['MMApp.app.MMClient'] = patch('MMApp.app.MMClient')
        self.patchers['MMApp.app.RoleProvider'] = patch('MMApp.app.RoleProvider')
        self.patchers['MMApp.app.NotificationService'] = patch('MMApp.app.NotificationService')
        self.patchers['MMApp.app.AutopilotService'] = patch('MMApp.app.AutopilotService')
        self.patchers['MMApp.app.google_auth'] = patch('MMApp.app.google_auth')
        self.patchers['MMApp.app.EdaService'] = patch('MMApp.app.EdaService')
        self.patchers['MMApp.app.delete_models'] = patch('MMApp.app.delete_models')
        self.patchers['MMApp.app.SecureBrokerClient'] = patch('MMApp.app.SecureBrokerClient')
        self.patchers['MMApp.app.FeaturelistService'] = patch('MMApp.app.FeaturelistService')

        self.MockUserService = self.patchers['MMApp.app.UserService'].start()
        self.MockIdeService = self.patchers['MMApp.app.IdeService'].start()
        self.MockIdeService.side_effect = raise_for_uid_and_pid
        self.MockQueueService = self.patchers['MMApp.app.QueueService'].start()
        self.MockQueueService.side_effect = raise_for_pid
        self.MockDatasetService = self.patchers['MMApp.app.DatasetService'].start()
        self.MockDatasetService.return_value.get_unseen_analysis.side_effect =\
            raise_for_uid_and_pid
        self.MockProjectService = self.patchers['MMApp.app.ProjectService'].start()
        self.MockProjectService.side_effect = raise_for_pid
        self.MockProjectService.return_value.get_all_metrics_list.return_value = [
            {   'short_name'    : 'LogLoss',
                'default'       : True,
                'recommend'     : False,
                'weighted'      : False,
                'weight+rec'    : False
            }
        ]
        self.MockMMClient = self.patchers['MMApp.app.MMClient'].start()
        self.MockRoleProvider = self.patchers['MMApp.app.RoleProvider'].start()
        self.MockNotificationService = self.patchers['MMApp.app.NotificationService'].start()
        self.MockAutopilotService = self.patchers['MMApp.app.AutopilotService'].start()
        self.MockAutopilotService.side_effect = raise_for_pid
        self.MockGoogleAuth = self.patchers['MMApp.app.google_auth'].start()
        self.MockEdaService = self.patchers['MMApp.app.EdaService'].start()
        self.MockEdaService.side_effect = raise_for_pid
        self.MockEdaService.return_value.get.return_value = {}
        self.MockDeleteModels = self.patchers['MMApp.app.delete_models'].start()
        self.MockSecureBrokerClient = self.patchers['MMApp.app.SecureBrokerClient'].start()
        self.MockFeaturelistService = self.patchers['MMApp.app.FeaturelistService'].start()

    def stopPatching(self):
        super(TestApp, self).tearDown()
        for patcher in self.patchers.itervalues():
            if patcher:
                patcher.stop()


    def assertResponse(self, response, status_code, content_type=None,
            non_empty=True):
        self.assertEqual(response.status_code, status_code)
        if content_type:
            self.assertEqual(response.content_type, content_type)
        if non_empty:
            self.assertGreater(len(response.data), 0, "Expected non-empty response")
    def dummySuccess(self):
        response = Response('{"status": "OK", "testing": "1"}', 200,
            mimetype="application/json")
        return response
    def requireUser(self, client, url, expected=404, method="GET"):
        """
            Makes 2 requests using the client, url and method specified
            - The first is performed without mocking get_user_info (no user in the session object)
            - The second call mocks get_user_info. Both requests verify the response against the expected parameter
        """
        if method == "GET":
            client_method = client.get
        elif method == "POST":
            client_method = client.post
        elif method == "PUT":
            client_method = client.put
        elif method == "DELETE":
            client_method = client.delete
        elif method == "PATCH":
            client_method = client.patch
        else:
            raise ValueError("Expected GET|POST|PUT|DELETE|PATCH method")

        self.MockUserService.return_value.get_user_info.return_value = None
        response = client_method(url)
        self.assertResponse(response, 400)

        self.MockUserService.return_value.get_user_info.return_value = \
            {"uid": self.generate_id(),"username":"project-owner@datarobot.com"}
        response = client_method(url)
        self.assertResponse(response, expected)
        return self.MockUserService

    def test_route_front(self):
        front_urls = ["/", "/account", "/eda", "/home", "/insights",
            "/models", "/python", "/project", "/new", "/refine", "/r_studio" ]

        with self.app as c:
            for url in front_urls:
                response = c.get(url)
                self.assertResponse(response, 200, 'text/html; charset=utf-8')

    def test_no_nginx_conflict(self):
        test_urls = ['/rstudio', '/api']
        with self.app as c:
            for url in test_urls:
                response = c.get(url)
                self.assertResponse(response, 404)

    def test_route_jsconfig(self):
        with self.app as c:
            response = c.get('/config.js')
            self.assertResponse(response, 200, 'application/javascript')

            criteria = ['logging', 'socketio', 'version', 'runmode']
            self.assertTrue(all(k in response.data for k in criteria), 'Response %s did not include the expected keys: %s' % (response.data, criteria))

    def test_join_get(self):
        user_service = self.MockUserService.return_value
        user_service.validate_invite.side_effect = UserException('BOOM!')

        response = self.app.get('/join')
        self.assertEqual(response.status_code, 302)


        user_service = self.MockUserService.return_value
        user_service.validate_invite.return_value = True

        response = self.app.get('/join')
        self.assertEqual(response.status_code, 302)

    def test_join_post(self):

        user_service = self.MockUserService.return_value
        user_service.needs_approval.return_value  = False
        user_service.get_account.return_value  = {}

        payload = {
            'email': 'user@example.com',
            'password': 'testing123',
            'passwordConfirmation': 'testing123',
            'firstName': 'first',
            'lastName': 'last',
            'tos': True
        }
        with self.app.session_transaction() as session:
            session['email'] = payload['email']
            session['invite_code'] = 123

        response = self.app.post('/join', content_type='application/json', data=json.dumps(payload))
        self.assertEqual(response.status_code, 200)

    def test_route_newdata_no_uid(self):
        with self.app as c:
            self.MockUserService.return_value.get_user_info.return_value = None
            response = c.get("/project/some_pid/newdata")
            self.assertResponse(response, 400)

    def test_route_newdata_bad_pid(self):
        with self.app as c:
            self.MockUserService.return_value.get_user_info.return_value = {'uid': 'a_uid'}
            response = c.get("/project/invalid_pid/newdata")
            self.assertResponse(response, 400)

    def test_route_newdata(self):
        with self.app as c:
            self.MockUserService.return_value.get_user_info.return_value = {'uid': 'a_uid'}
            self.MockProjectService.return_value.get_all_metadata.return_value = []
            response = c.get("/project/" + self.generate_id() + "/newdata")
            self.assertResponse(response, 200)

    def test_route_aim_no_data(self):
        with self.app as c:
            self.MockUserService.return_value.get_user_info.return_value = None
            response = c.post("/aim")
            self.assertResponse(response, 422)

    def test_route_aim_empty_data(self):
        with self.app as c:
            self.MockUserService.return_value.get_user_info.return_value = \
                {"uid": self.generate_id()}
            response = c.post("/aim", data='{}',
                content_type='application/json')
            self.assertResponse(response, 422)

    def test_route_aim_no_permissions(self):
        self.MockRoleProvider.return_value.has_permission.return_value = False
        self.MockUserService.return_value.get_user_info.return_value = {"uid": str(self.generate_id())}

        data = {
            "target": "ATarget",
            "pid": self.generate_id(),
            "mode": 1
        }

        response = self.app.post("/aim", data=json.dumps(data),
            content_type='application/json')
        self.assertResponse(response, 404)

    def test_set_aim_when_target_was_just_submitted(self):
        project_service = self.MockProjectService.return_value
        project_service.get_target.return_value = None
        project_service.get_submitted_target.return_value = 'some-var'
        project_service.metric = 'metric'

        data = {
            'target': 'ATarget',
            'pid': self.generate_id(),
            'mode': 1
        }

        response = self.app.post('/aim', data=json.dumps(data),
            content_type='application/json')
        self.assertResponse(response, 200)
        response_data = json.loads(response.data)
        self.assertEqual(response_data['status'], 'pre-selected')

    def test_set_aim_when_target_exists(self):
        project_service = self.MockProjectService.return_value
        project_service.get_target.return_value = {'name' : 'some-var'}
        project_service.metric = 'metric'

        data = {
            'target': 'ATarget',
            'pid': self.generate_id(),
            'mode': 1
        }

        response = self.app.post('/aim', data=json.dumps(data),
            content_type='application/json')
        self.assertResponse(response, 200)
        response_data = json.loads(response.data)
        self.assertEqual(response_data['status'], 'pre-selected')

        self.assertFalse(project_service.get_submitted_target.called)

    def test_route_aim_should_succeed(self):
        project_service = self.MockProjectService.return_value
        project_service.get.return_value = {}
        project_service.get_target.return_value = None
        project_service.get_submitted_target.return_value = None
        project_service.validate_cv_method.return_value = {}
        self.MockRoleProvider.return_value.get_uids_by_permission.return_value = []
        self.MockUserService.return_value.get_user_info.return_value = {"uid": str(self.generate_id())}

        data = {
            "target": "ATarget",
            "pid": self.generate_id(),
            "mode": 1
        }

        with patch("MMApp.app.time.sleep"):
            response = self.app.post("/aim", data=json.dumps(data),
                content_type='application/json')
            self.assertResponse(response, 200)
            self.assertEqual(project_service.metric, 'Gini')

    def test_route_eda(self):
        self.MockDatasetService.return_value.get_eda.return_value = {"data": []}
        with self.app as c:
            with self.requireUser(c, "/eda/invalid_pid", 400):
                response = c.get("/eda/" + self.generate_id() + "")
                self.assertResponse(response, 404)

    def test_route_get_eda_names(self):
        self.MockFeaturelistService.return_value.get_universe_feature_names.return_value = {"data": []}
        with self.app as c:
            with self.requireUser(c, "/eda/names/invalid_pid", 400):
                response = c.get("/eda/names/" + self.generate_id())
                self.assertResponse(response, 200)

    def test_route_get_eda_profile(self):
        self.MockDatasetService.return_value.get_universe_profile.return_value = {"data": []}
        with self.app as c:
            with self.requireUser(c, "/eda/profile/invalid_pid", 400):
                response = c.get("/eda/profile/" + self.generate_id())
                self.assertResponse(response, 200)

    def test_route_get_eda_graphs(self):
        self.MockDatasetService.return_value.get_universe_graphs.return_value = {"data": []}
        with self.app as c:
            with self.requireUser(c, "/eda/graphs/invalid_pid", 400):
                response = c.get("/eda/graphs/" + self.generate_id())
                self.assertResponse(response, 200)

    def test_route_predictions_bad_lid(self):
        with self.requireUser(self.app, "/predictions/invlid/fileid", 400):
            self.MockProjectService.return_value.get_predictions.return_value = {}
            response = self.app.get("/predictions/invlidpid/fileid")
            self.assertResponse(response, 400)


    def test_route_predictions_bad_dataset_id(self):
        with self.requireUser(self.app, "/predictions/invlid/fileid", 400):
            self.MockProjectService.return_value.get_predictions.return_value = {}
            response = self.app.get("/predictions/" + self.generate_id() + "/invlid")
            self.assertResponse(response, 404)


    def test_route_predictions_good(self):
        with self.requireUser(self.app, "/predictions/invlid/fileid", 400):
            retval = {"_id": None, "lid": None, "dataset_id": None, "actual": None, 'predicted-0': [1,2,3,4], 'row_index': [0,1,2,3], 'Full Model 80%': [0,1,2,3]}
            self.MockProjectService.return_value.get_predictions.return_value = retval
            self.MockProjectService.return_value.get_leaderboard_item.return_value = {'samplepct':64,'model_type':'GBM','bp':'2','training_dataset_id':'blah'}
            response = self.app.get("/predictions/" + self.generate_id() + "/valid")
            self.assertResponse(response, 200, non_empty=False)


    def test_route_project(self):
        with self.app as c:
            with self.requireUser(c, "/project/invalid_pid", 400):
                self.MockProjectService.return_value.get_project_data.return_value = {}
                response = c.get("/project/" + self.generate_id() + "")
                self.assertResponse(response, 200)

    def test_route_project_delete(self):
        uid = self.generate_id()
        pid = self.generate_id()

        # This is one way to patch around the user service - see
        # ``get_user_info`` in MMApp.app for why it has to be like
        # this
        class FakeUserService(object):
            def get_user_info(self):
                return {'pid': pid, 'uid': uid, 'username': 'user@userdom.com'}

        self.MockProjectService.return_value.uid = uid
        self.MockProjectService.return_value.pid = pid
        self.MockUserService.return_value = FakeUserService()

        with self.app as c:
            response = c.delete('/project/' + pid)
            self.assertResponse(response, 200)

            self.MockIdeService.return_value.delete.assert_called_once_with()

    def test_route_project_post_normal_behavior(self):
        '''The POST route for /project/<pid> will be used to clone the given
        project and return information on the new project
        '''
        uid = self.generate_id()
        pid = self.generate_id()
        new_pid = self.generate_id()

        # Stole this idea from above
        class FakeUserService(object):
            def get_user_info(self):
                return {'pid': pid, 'uid': uid, 'username': 'user@userdom.com'}

        mock_project_service = self.MockProjectService.return_value

        mock_project_service.uid = uid
        mock_project_service.pid = pid
        mock_project_service.get.return_value = {
            'originalName': 'kickcars-sample-200.csv',
            'default_dataset_id': '5223deadbeefdeadbeef1234'}
        mock_project_service.get_project_info_for_user.return_value = {
            'originalName': 'kickcars-sample-200.csv',
            'default_dataset_id': '5223deadbeefdeadbeef1235'}
        self.MockUserService.return_value = FakeUserService()

        with self.app as c:
            response = c.post('/project/{}'.format(pid),
                              content_type='application/json',
                              data=json.dumps({'newProjectId': str(new_pid)}))
            self.assertResponse(response, 200)
            self.MockMMClient.return_value.startCSV.assertCalled()

    def test_route_listprojects(self):
        with self.app as c:
            self.MockProjectService.return_value.get_project_list.return_value = {}
            with self.requireUser(c, "/listprojects", 200):
                return

    def test_route_queue_GET(self):
        with self.app as c:
            self.MockQueueService.return_value.get.return_value = {}
            with self.requireUser(c, "/project/invalid_pid/queue", 400):
                response = c.get("/project/" + self.generate_id() + "/queue")
                self.assertResponse(response, 200)

    def test_route_queue_by_id(self):
        with self.app as c:
            with self.requireUser(c, "/project/invalid_pid/queue/qid", 400, method="PUT"):
                response = c.put("/project/" + self.generate_id() + "/queue/-1",
                    data='{"parallel": 0}',
                    content_type='application/json')
                self.assertResponse(response, 200)

    @patch("MMApp.app.time.sleep")
    def test_route_queue_DELETE(self, pch1):
        with self.app as c:
            with self.requireUser(c, "/project/invalid_pid/queue/qid", 400, method="DELETE"):
                response = c.delete("/project/" + self.generate_id() + "/queue/1")
                self.assertResponse(response, 200)

    def test_route_models_GET(self):
        with self.app as c:
            with self.requireUser(c, "/project/invalid_pid/models", 400):
                self.MockProjectService.return_value.get_leaderboard.return_value = \
                    []
                response = c.get("/project/" + self.generate_id() + "/models")
                self.assertResponse(response, 200)

    def test_remove_models(self):
        response = self.app.post('/project/{}/models/delete'.format(self.generate_id()),
            content_type='application/json', data=json.dumps({'lids': [1,2,3]}))
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'OK')

    def test_remove_models_no_ids(self):
        response = self.app.post('/project/{}/models/delete'.format(self.generate_id()),
            content_type='application/json', data=json.dumps({}))
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'error')
        self.assertIn(data['message'], 'Invalid request')

    def test_route_job_POST(self):
        with self.app as c:
            response = c.post("/project/invalid_pid/job", data=json.dumps({'command':'something'}),
                content_type='application/json')
            self.assertResponse(response, 400)

            response = c.post("/project/%s/job"%self.generate_id(), data=json.dumps({'command':'something'}),
                content_type='application/json')
            self.assertResponse(response, 200)

    def test_route_get_tabulation(self):
        with self.app as c:
            with self.requireUser(c, "/project/invalid_pid/models/invalid_lid/tabulation", 400):
                self.MockProjectService.return_value.get_prediction_tabulation.return_value = {}
                response = c.get("/project/%s/models/%s/tabulation"%(self.generate_id(), self.generate_id()))
                self.assertResponse(response, 200)

    def test_route_autopilot_POST(self):
        with self.app as c:
            response = c.post("/project/invalid_pid/autopilot", data=json.dumps({'pause':True}),
                    content_type='application/json')
            self.assertResponse(response, 400)

            self.MockAutopilotService.return_value.set.return_value = ''
            response = c.post("/project/%s/autopilot"%self.generate_id(), data=json.dumps({'pause':True}),
                    content_type='application/json')
            self.assertResponse(response, 200)


    @patch("MMApp.app.time.sleep")
    def test_route_service(self, pch1):
        self.MockQueueService.return_value.next.return_value = None
        self.MockQueueService.return_value.start_new_tasks.return_value = 0
        with self.app as c:
            with self.requireUser(c, "/project/invalid_pid/service", 400):
                response = c.get("/project/" + self.generate_id() + "/service")
                self.assertResponse(response, 200)

    def test_get_project_metric(self):
        with self.app as c:
            self.MockProjectService.return_value.metric = 'RMSLE'
            response = c.get("/project/"+self.generate_id() + "/metric")
            self.assertResponse(response, 200)
            resp_data = json.loads(response.data)
            self.assertIn('metric', resp_data)
            self.assertEqual(resp_data['metric'], 'RMSLE')

    def test_get_all_metrics_list(self):
        response = self.app.get('/eda/metrics/' + self.generate_id())
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data.get('data'))

    def test_set_project_metric(self):
        response = self.app.post('/project/' + self.generate_id() + '/metric',
                          data=json.dumps({'metric':'AUC'}),
                          content_type='application/json')
        self.assertResponse(response, 200)
        resp_data = json.loads(response.data)
        self.assertIn('metric', resp_data)
        self.assertEqual(resp_data['metric'], 'AUC')

    def test_route_account_signup(self):

        self.MockUserService.return_value.get_error.return_value = 'an_error'
        self.MockUserService.return_value.create_account.return_value = 'a_username'

        self.mock_is_private_mode.return_value = False
        response = self.app.post("/account/signup", data='')
        self.assertResponse(response, 400)

        response = self.app.post("/account/signup", data='{}',
            content_type='application/json')
        self.assertResponse(response, 400)

        response = self.app.post("/account/signup", data=json.dumps({"username": "Test",
            "password": "testpass"}), content_type='application/json')
        self.assertResponse(response, 200)


    def test_change_password_do_not_match(self):
        payload = {
            'current': 'pwd',
            'new': 'new-pwd',
            'confirm': 'different-pwd'
        }
        response = self.app.post('/account/password', content_type='application/json', data=json.dumps(payload))

        self.assertEqual(response.status_code, 400)
        self.assertIn('passwords do not match', response.data.lower())

    def test_change_password_missing_data(self):
        payload = {
            'current': None,
            'new': 'new-pwd',
            'confirm': 'different-pwd'
        }
        response = self.app.post('/account/password', content_type='application/json', data=json.dumps(payload))

        self.assertEqual(response.status_code, 400)
        self.assertIn('invalid request', response.data.lower())

    def test_change_password_success(self):
        payload = {
            'current': 'old-pwd',
            'new': 'new-pwd',
            'confirm': 'new-pwd'
        }
        response = self.app.post('/account/password', content_type='application/json', data=json.dumps(payload))

        self.assertEqual(response.status_code, 200)
        self.assertIn('ok', response.data.lower())

    def test_change_password_fail(self):
        user_service = self.MockUserService.return_value
        user_service.change_password.side_effect = UserException('Invalid Password')

        payload = {
            'current': 'old-pwd',
            'new': 'new-pwd',
            'confirm': 'new-pwd'
        }

        response = self.app.post('/account/password', content_type='application/json', data=json.dumps(payload))

        self.assertEqual(response.status_code, 200)
        self.assertIn('invalid password', response.data.lower())

    def test_reqest_reset_password_get(self):
        response = self.app.get('/account/request/password/reset')
        self.assertResponse(response, 400)

        user_service = self.MockUserService.return_value
        user_service.reset_key_exists.return_value = False

        response = self.app.get('/account/request/password/reset?code=code')
        self.assertResponse(response, 400)


        user_service.reset_key_exists.return_value = True

        response = self.app.get('/account/request/password/reset?code=code', follow_redirects = True)
        self.assertResponse(response, 200)

    def test_reqest_reset_password_post(self):
        # Invalid value passed to the POST
        response = self.app.post('/account/request/password/reset', content_type='application/json',
            data=json.dumps({"username": None}))
        self.assertResponse(response, 400)
        self.assertTrue("Invalid request" in response.data)

        # Invalid email POSTed
        self.MockUserService.valid_email.return_value = False
        response = self.app.post('/account/request/password/reset', content_type='application/json',
                data=json.dumps({"username": "testuser"}))
        self.assertResponse(response, 400, content_type='application/json')
        self.assertTrue('Invalid email' in response.data)

        # Valid email, reset email successfully sent
        self.MockUserService.return_value.send_forgot_password_email.return_value = True
        self.MockUserService.valid_email.return_value = True
        response = self.app.post('/account/request/password/reset', content_type='application/json',
                data=json.dumps({"username": "testuser@example.com"}))
        self.assertResponse(response, 200, content_type='application/json')
        self.assertTrue('OK' in response.data)

        # Valid email, reset email unsuccessfully sent
        self.MockUserService.return_value.send_forgot_password_email.return_value = False
        self.MockUserService.return_value.get_error.return_value = 'Error when sending password reset email'
        response = self.app.post('/account/request/password/reset', content_type='application/json',
                data=json.dumps({"username": "testuser@example.com"}))
        self.assertResponse(response, 200, content_type='application/json')
        self.assertTrue('Error when sending password reset email' in response.data)


    def test_reset_password_no_code(self):
        response = self.app.post('/account/password/reset', content_type = 'application/json')
        self.assertResponse(response, 400)

    def test_reset_password_success(self):
        user_service = self.MockUserService.return_value
        user_service.reset_key_exists.return_value = True
        user_service.get_account.return_value = {}
        user_service.username = ''
        with self.app.session_transaction() as session:
            session['reset_key'] = 'OwPHA4G4E7-G9078aId-YwdHepPO4xHF'

        data = {
            'password' : 'password',
            'passwordConfirmation' : 'password'
        }
        response = self.app.post('/account/password/reset',
            content_type = 'application/json', data = json.dumps(data))

        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertEqual(response_data['message'], 'OK')

    def test_reset_password_invalid_code(self):
        with self.app.session_transaction() as session:
            session['reset_key'] = 'OwPHA4G4E7-G9078aId-YwdHepPO4xHF'

        data = {
            'password' : 'password',
            'passwordConfirmation' : 'password'
        }

        user_service = self.MockUserService.return_value
        user_service.reset_password.return_value = False
        response = self.app.post('/account/password/reset',
            content_type = 'application/json', data = json.dumps(data))

        self.assertResponse(response, 400)

    def test_reset_password_login_fails(self):
        with self.app.session_transaction() as session:
            session['reset_key'] = 'OwPHA4G4E7-G9078aId-YwdHepPO4xHF'

        data = {
            'password' : 'password',
            'passwordConfirmation' : 'password'
        }
        user_service = self.MockUserService.return_value
        user_service.reset_password.return_value = True
        user_service.login.return_value = False
        user_service.get_error.return_value = 'error somewhere'
        user_service.username = 'does-not-matter@datarobot.com'
        response = self.app.post('/account/password/reset',
            content_type = 'application/json', data = json.dumps(data))

        self.assertResponse(response, 400)

    def test_route_account_signup_private_mode(self):
        self.mock_is_private_mode.return_value = True
        response = self.app.post('/account/signup', data=json.dumps({'username': 'Test',
            'password': 'testpass'}), content_type='application/json')
        #Assert a valid request returns 404 when the private mode is on
        self.assertResponse(response, 404)

    def test_route_account_login(self):
        self.MockUserService.return_value.get_error.return_value = "an_error"
        self.MockUserService.return_value.login.return_value = True
        with self.app as c:
            response = c.post("/account/login", data='')
            self.assertResponse(response, 404)

            response = c.post("/account/login", data='{}',
                content_type='application/json')
            self.assertResponse(response, 404)

            response = c.post("/account/login", data=json.dumps({"username": "Test",
                "password": "testpass"}), content_type='application/json')
            self.assertResponse(response, 200)

    def test_route_account_logout(self):
        self.MockUserService.return_value.get_user_info.return_value = {"uid": "" + self.generate_id() + ""}
        with self.app as c:
            response = c.get("/account/logout")
            self.assertResponse(response, 302)

    def test_route_account_username(self):
        user_service_instance = self.MockUserService.return_value
        user_service_instance.username = "Test"
        user_service_instance.is_guest.return_value = 0
        user_service_instance.get_user_info.return_value = {'_id':'123'}
        user_service_instance.get_account.return_value = {'id': '123',
                                                          'username': 'Test',
                                                          'statusCode': '0',
                                                          'userhash': 'adfebc'}
        with self.app as c:
            response = c.get("/account/profile")
            self.assertResponse(response, 200)

    def test_route_api_token(self):
        self.MockProjectService.return_value.get_token.return_value = "a_token"
        self.MockUserService.return_value.get_user_info.return_value = {'uid':'123'}
        with self.app as c:
            response = c.get("/project/invalid_pid/session")
            self.assertResponse(response, 400)

            response = c.get("/project/" + self.generate_id() + "/session")
            self.assertResponse(response, 200)

    def test_report_ide_error(self):
        with self.app as c:
            with self.requireUser(c, "/ide/rstudio/error/invalid_pid", 400, method = 'POST'):
                response = c.post("/ide/rstudio/error/" + self.generate_id())
                self.assertResponse(response, 200)


    def test_route_ide_status(self):
        ide_status = self.MockIdeService.return_value.get_status.return_value
        ide_status.status = 'some status'
        ide_status.to_encoded_dict.return_value = {'status' : 'some status'}
        setup_mock = self.MockIdeService.return_value.setup.return_value
        setup_mock.status = 'some status'
        setup_mock.to_encoded_dict.return_value = {'status' : 'some status'}

        with self.app as c:
            with self.requireUser(c, "/project/invalid_pid/ide_status", 400):
                response = c.get("/project/" + self.generate_id() + "/ide_status")
                self.assertResponse(response, 200)


    def test_route_resource_usage(self):
        self.MockQueueService.return_value.resources.return_value = {"cpu": [], "mem": []}
        with self.app as c:
            with self.requireUser(c, "/project/invalid_pid/resource_usage", 400):
                response = c.get("/project/" + self.generate_id() + "/resource_usage")
                self.assertResponse(response, 200)

    def test_route_next_steps_GET(self):
        with self.app as c:
            with self.requireUser(c, "/project/invalid_pid/next_steps", 400):
                response = c.get("/project/" + self.generate_id() + "/next_steps")
                self.assertResponse(response, 200)

    def test_route_blueprint_GET(self):
        self.MockProjectService.return_value.get_blueprint_diagram_from_blueprint_id.return_value = {}
        with self.app as c:
            with self.requireUser(c, "/project/invalid_pid/bp/invalid_id", 400):
                response = c.get("/project/" + self.generate_id() + "/bp/anything")
                self.assertResponse(response, 200)

    def test_service_with_other_projects(self):
        '''Simulate a condition where there are already 6 jobs inprogress, but the
        queue limit is 4.  This should trigger a "do nothing", which we test by
        ensuring ``task_available_workers`` is never called
        '''
        self.MockQueueService.return_value.start_new_tasks.return_value = 0
        with self.app as c:
            pid_0 = self.generate_id()
            uid = self.generate_id()
            #Simulate three jobs in progress for each project
            self.MockQueueService.return_value.inprogress.hvals.return_value = [
                    dict([(str(i),i) for i in xrange(5)]) for j in xrange(3)]
            #Simulate two projects associated with this user
            user_service = self.MockUserService.return_value
            user_service.get_user_info.return_value = {'uid':uid}
            user_service.get_user_projects.return_value = [
                    {'_id':pid_0,
                     'uid':uid},
                    {'_id':self.generate_id(),
                     'uid':uid}
                    ]

            url = '/project/{}/service'.format(pid_0)
            response = c.get(url)
            resp_data = json.loads(response.data)
            self.assertEqual(resp_data.get('count'), 0)

    def test_get_all_roles(self):
        with self.app as c:
            r = c.get('/roles')
            self.assertResponse(r, 200)
            roles = json.loads(r.data)
            self.assertIsInstance(roles, list)
            self.assertGreater(len(roles), 0)

    def test_change_roles(self):
        with self.app as c:
            with self.requireUser(c, '/project/bad_pid/team/roles', 400, method='POST'):
                role_provider_instance = self.MockRoleProvider.return_value
                role_provider_instance.has_permission.return_value = True

                project_service = self.MockProjectService.return_value
                project_service.get_project_info_for_user.return_value = {}

                payload = {'team_member_uid' : self.generate_id(), 'roles' : ['OBSERVER']}
                r = c.post('/project/{0}/team/roles'.format(self.generate_id()), data = json.dumps(payload),
                    content_type='application/json')

                self.assertResponse(r, 200)

    def test_invite_existing_user(self):

        user_service = self.MockUserService.return_value
        user_service.get_user_info.return_value  = {'uid' : str(ObjectId())}

        project_owner = {
            'guest': 0,
            'max_workers': 12,
            'time': '1399057205.56',
            'uid': '535eadb93d86335190d685c5',
            'username': 'project-owner@datarobot.com',
        }

        new_team_member_id = str(self.generate_id())
        new_user = {
            'username': 'existing-user@datarobot.com',
            '_id': new_team_member_id
        }

        user_service = self.MockUserService.return_value
        user_service.get_user_info.return_value = project_owner
        user_service.find_user.return_value = UserModel(**new_user)

        project_service = self.MockProjectService.return_value
        project_service.get_project_info_for_user.return_value = {}

        payload = {'email' : new_user['username']}
        r = self.app.post('/project/{0}/team'.format(self.generate_id()), data = json.dumps(payload),
            content_type='application/json')

        self.assertResponse(r, 200)

        notification_service = self.MockNotificationService.return_value
        invite_data = notification_service.invite_user_to_project.call_args[0][0]


        project_service.add_team_member.assert_called_once_with(new_team_member_id)

        self.assertEqual(user_service.create_invite.call_count,0)
        self.assertEqual(invite_data['sender'], 'project-owner@datarobot.com')
        self.assertFalse(user_service.get_inviter.called)


    def test_invite_new_user(self):
        # Invite new user
        email = 'two-invites@email.com'
        new_team_member_id = str(self.generate_id())
        new_user = {
            'username': 'existing-user@datarobot.com',
            '_id': new_team_member_id
        }

        project_owner = {
            'uid': self.generate_id(),
            'username': 'project-owner@datarobot.com',
            'first_name' : None,
            'last_name' : None,
            'activated': 1 }

        user_service = self.MockUserService.return_value
        user_service.get_user_info.return_value = project_owner
        user_service.find_user.return_value = None
        user_service.create_invite.return_value = (new_team_member_id, 'invite-code')

        project_service = self.MockProjectService.return_value
        project_service.get_project_info_for_user.return_value = {}

        response = self.app.post('/project/{}/team'.format(self.generate_id()),
                content_type='application/json', data=json.dumps(new_user))
        self.assertEqual(response.status_code, 200, response.data)

        notification_service = self.MockNotificationService.return_value
        self.assertEqual(notification_service.invite_new_user_to_project.call_count, 1)
        self.assertEqual(notification_service.invite_user_to_project.call_count, 0)

        project_service.add_team_member.assert_called_once_with(new_team_member_id)

        user_service.create_invite.assert_called_once_with(project_owner['username'])

    def test_invite_unregistered_user(self):
        # Invite new user
        email = 'two-invites@email.com'
        new_team_member_id = str(self.generate_id())
        new_user = {
            'username': 'existing-user@datarobot.com',
            '_id': new_team_member_id,
            'needs_approval' : True
        }

        project_owner = {
            'uid': self.generate_id(),
            'username': 'project-owner@datarobot.com',
            'first_name' : None,
            'last_name' : None,
            'activated': 1 }

        user_service = self.MockUserService.return_value
        user_service.get_user_info.return_value = project_owner
        user_service.find_user.return_value = UserModel(**new_user)

        project_service = self.MockProjectService.return_value
        project_service.get_project_info_for_user.return_value = {}

        response = self.app.post('/project/{}/team'.format(self.generate_id()),
                content_type='application/json', data=json.dumps(new_user))
        self.assertEqual(response.status_code, 200, response.data)

        notification_service = self.MockNotificationService.return_value

        self.assertEqual(notification_service.invite_new_user_to_project.call_count, 1)
        self.assertEqual(notification_service.invite_user_to_project.call_count, 0)

        project_service.add_team_member.assert_called_once_with(new_team_member_id)

        self.assertEqual(user_service.create_invite.call_count,0)

    def test_go_to_google_login_page_pops_invite_code_for_login(self):

        link = 'google-server.com'

        self.MockGoogleAuth.get_authorization_url.return_value = link

        with self.app.session_transaction() as session:
            session['invite_code'] = 'should-be-removed-when-loging-in'

        response = self.app.get('/account/google/login')

        with self.app.session_transaction() as session:
            self.assertNotIn('invite_code', session)


        # Redirects to google server login_with_google
        self.assertEqual(response.status_code, 302, response.data)
        self.assertIn(link, response.data)

    def test_go_to_google_login_page_for_login(self):

        link = 'google-server.com'

        self.MockGoogleAuth.get_authorization_url.return_value = link

        with self.app.session_transaction() as session:
            session['invite_code'] = 'should-be-removed-when-loging-in'

        response = self.app.get('/account/google/login?email=does-not-matter@datarobot.com')

        with self.app.session_transaction() as session:
            self.assertIn('invite_code', session)


        # Redirects to google server login_with_google
        self.assertEqual(response.status_code, 302, response.data)
        self.assertIn(link, response.data)

    def test_go_to_google_login_creates_link_with_schema(self):
        # Get URL: /account/google/login
        with patch.dict(EngConfig, {'SSL': False }, clear = False):
            response = self.app.get('/account/google/login')
        # Redirects to google server login_with_google
        self.assertIn('http://', self.MockGoogleAuth.REDIRECT_URI)

        with patch.dict(EngConfig, {'SSL': True }, clear = False):
            response = self.app.get('/account/google/login')
        # Redirects to google server login_with_google
        self.assertIn('https://', self.MockGoogleAuth.REDIRECT_URI)


    def test_login_with_google_and_matching_username(self):
        # self.MockGoogleAuth.get_credentials.return_value = {}
        self.MockGoogleAuth.get_user_info.return_value = {'email' : 'test@google.com'}
        user_service = self.MockUserService.return_value
        user_service.find_by_linked_account.return_value = UserModel()

        # Redirects to project
        response = self.app.get('/account/google/auth_return?code={}'.format('code'), follow_redirects = True)
        self.assertIn('/project', response.data)

        # Link actually works
        response = self.app.get('/account/google/auth_return?code={}'.format('code'), follow_redirects = True)
        self.assertEqual(response.status_code, 200, response.data)


    def test_login_with_google_and_different_username(self):
        # self.MockGoogleAuth.get_credentials.return_value = {}
        self.MockGoogleAuth.get_user_info.return_value = {'email' : 'test@google.com'}
        user_service = self.MockUserService.return_value
        user_service.find_by_linked_account.return_value = None

        # Redirects to "public" url to link account
        response = self.app.get('/account/google/auth_return?code={}'.format('code'))
        self.assertIn('/p/link', response.data)

        # Link actually works
        response = self.app.get('/account/google/auth_return?code={}'.format('code'), follow_redirects = True)
        self.assertEqual(response.status_code, 200, response.data)

    def test_cannot_create_guest_account_in_private_mode(self):
        self.mock_is_private_mode.return_value = True
        response = self.app.post('/project')

        self.assertEqual(response.status_code, 400)

    def test_create_project_guest_user(self):
        pid = ObjectId()
        project_service = self.MockProjectService.return_value
        self.MockProjectService.create_project.return_value = pid
        project_service.get_token.return_value = '123'
        project_service.get_project_info_for_user.return_value = {}


        user_service = self.MockUserService.return_value
        user_service.create_random_key.return_value  = 'random-user'
        user_service.get_user_info.return_value  = {'uid' : str(ObjectId()), 'username': str(ObjectId())}
        user_service.get_account.return_value = {
            'username': '61fa4292-75e0-4f62-975f-fe0b00b28c55',
            'guest': 1,
            'userhash': 'ee29869c9f1ec92a3fd28b1425058df3',
            'flags': {'cant_create_projects': False, 'is_locked': False, 'graybox_disabled': False, 'model_debug': False, 'cant_invite': False, 'can_manage_user_flags': False
            },
            'id': '537786ea637aba65640fa7df',
            'statusCode': '1'
        }

        self.mock_is_private_mode.return_value = False
        response = self.app.post('/project')

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['pid'], str(pid))
        self.assertIn('project', data)
        self.assertIn('guest_profile', data)

        # We created a guest user
        user_service.create_guest.assert_called_once_with('random-user')

    def test_create_project_registered_user(self):
        pid = ObjectId()
        project_service = self.MockProjectService.return_value
        self.MockProjectService.create_project.return_value = pid
        project_service.get_token.return_value = '123'
        project_service.get_project_info_for_user.return_value = {}

        user_service = self.MockUserService.return_value

        with self.app.session_transaction() as session:
            session['user'] = 'registered-user'
        user_service.get_user_info.return_value  = {'uid' : str(ObjectId())}

        response = self.app.post('/project')
        self.assertEqual(response.status_code, 200)
        auth_data = json.loads(response.data)
        self.assertEqual(auth_data['pid'], str(pid))
        self.assertIn('project', auth_data)

        # We did NOT create a guest user
        self.assertFalse(user_service.create_guest.called)

    @patch('MMApp.app.FLIPPERS', autospec = True)
    def test_create_project_with_worker_options(self, MockFLIPPERS):
        pid = ObjectId()
        project_service = self.MockProjectService.return_value
        self.MockProjectService.create_project.return_value = pid
        project_service.get_token.return_value = '123'
        project_service.get_project_info_for_user.return_value = {}

        user_service = self.MockUserService.return_value

        with self.app.session_transaction() as session:
            session['user'] = 'registered-user'
        user_service.get_user_info.return_value  = {'uid' : str(ObjectId())}

        worker_options ={
            'worker_options': {'mb_version': 'mb8_6_7b', 'worker_size':'m3.xlarge'}
        }

        MockFLIPPERS.allow_worker_options = True
        response = self.app.post('/project', content_type='application/json', data=json.dumps(worker_options))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(project_service.update.called)

    def test_ping(self):
        with self.app as c:
            response = c.get('/ping?token=' + '123')
            self.assertResponse(response, 200, 'application/json')
            resp_data = json.loads(response.data)
            self.assertEqual(resp_data['response'], 'pong')
            self.assertEqual(resp_data['token'], '123')

