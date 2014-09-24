import unittest
import json
import os
from cStringIO import StringIO
from mock import patch, Mock, sentinel
import pytest

from bson.objectid import ObjectId

from test_base import TestBase
from MMApp.entities.jobqueue import QueueException
from MMApp.entities.user_tasks import TaskAccessError
from MMApp.entities.ide import IdeSetupStatus
from MMApp.utilities.error import RequestError
from config.engine import EngConfig
from ModelingMachine.client import BrokerServiceResponse
from MMApp.api import get_r_model_file_name

@pytest.mark.unit
class TestApi(unittest.TestCase):
    """
        TestApi tests the API by mocking the service methods since these methods are tested individually
    """

    def setUp(self):

        self.leaderboard_items = [
            {
                "_id": ObjectId("5214d012637aba171e0bbb7a"),
                "pid" : "52dc0ba5637aba2829195fcd",
                "blend": 0,
                "blueprint": {
                    "1": [
                        [
                            "NUM"
                        ],
                        [
                            "NI",
                            "CCZ"
                        ],
                        "T"
                    ],
                    "2": [
                        [
                            "1"
                        ],
                        [
                            "RCC .98",
                            "ST",
                            "TWS glm"
                        ],
                        "P"
                    ]
                },
                "bp": 2,
                "dsid": "5214d011637aba17000bbb7b",
                "extras": {
                    "(0, -1)": {
                        "coefficients": [
                            [
                                "z",
                                766.239452
                            ],
                            [
                                "x",
                                -83.925223
                            ]
                        ],
                        "coefficients2": [
                            [
                                "z",
                                0.692961
                            ],
                            [
                                "x",
                                -0.563459
                            ]
                        ]
                    }
                },
                "features": [
                    "Numeric Inputs (no splines)"
                ],
                "hash": "8315b571e7281d5eb801d2eeff3637575d6d3765",
                "icons": [
                    0
                ],
                "insights": "effects",
                "lift": {
                    "(0,-1)": {
                        "act": [
                            2.0,
                            8.0,
                            10.0,
                            14.0,
                            11.0,
                            17.0,
                            20.0,
                            16.0,
                            14.0,
                            17.0,
                            17.0,
                            23.0,
                            24.0,
                            30.0,
                            30.0,
                            32.0,
                            31.0,
                            33.0,
                            36.0
                        ],
                        "pred": [
                            1.612492562791807,
                            8.420008971376443,
                            10.426332360434653,
                            12.1266169347727,
                            13.503465308305215,
                            14.792533526484261,
                            15.745207393495987,
                            16.7762591406743,
                            17.474530240944347,
                            18.350456340019626,
                            19.511072302937468,
                            20.63898554261928,
                            20.810045575574954,
                            21.912157692912167,
                            23.277049069771486,
                            25.45271904508055,
                            26.892080659281508,
                            32.157396001585504,
                            47.554261394692595
                        ],
                        "rows": [
                            8.0,
                            4.0,
                            4.0,
                            4.0,
                            4.0,
                            4.0,
                            4.0,
                            4.0,
                            4.0,
                            4.0,
                            4.0,
                            4.0,
                            4.0,
                            4.0,
                            4.0,
                            4.0,
                            4.0,
                            4.0,
                            4.0
                        ]
                    }
                },
                "max_folds": 0,
                "max_reps": 1,
                "model_type": "Two Stage Regression (GLM)",
                "part_size": [
                    [
                        "1",
                        320,
                        80
                    ]
                ],
                "parts": [
                    [
                        "1",
                        "0.06",
                        "3"
                    ]
                ],
                "parts_label": [
                    "partition",
                    "thresh",
                    "NonZeroCoefficients"
                ],
                "qid": 2,
                "s": 0,
                "samplesize": 400.0,
                "task_cnt": 5,
                "test": {
                    "%Error<=05%": [
                        0.2
                    ],
                    "%Error<=10%": [
                        0.3375
                    ],
                    "%Error<=15%": [
                        0.55
                    ],
                    "%Error<=20%": [
                        0.7375
                    ],
                    "%Error<=30%": [
                        0.8625
                    ],
                    "%Error<=50%": [
                        0.95
                    ],
                    "%ExpError<=05%": [
                        0.125
                    ],
                    "%ExpError<=10%": [
                        0.15
                    ],
                    "%ExpError<=15%": [
                        0.1625
                    ],
                    "%ExpError<=20%": [
                        0.1625
                    ],
                    "%ExpError<=30%": [
                        0.2625
                    ],
                    "%ExpError<=50%": [
                        0.3625
                    ],
                    "Gini": [
                        0.1469
                    ],
                    "Gini Norm": [
                        0.92934
                    ],
                    "MAD": [
                        1.00449
                    ],
                    "R Squared": [
                        0.74197
                    ],
                    "R Squared 20/80": [
                        -32.36271
                    ],
                    "RMSE": [
                        1.35612
                    ],
                    "RMSLE": [
                        0.24026
                    ],
                    "labels": [
                        "(0,-1)"
                    ],
                    "metrics": [
                        "%Error<=05%",
                        "%Error<=10%",
                        "%Error<=15%",
                        "%Error<=20%",
                        "%Error<=30%",
                        "%Error<=50%",
                        "%ExpError<=05%",
                        "%ExpError<=10%",
                        "%ExpError<=15%",
                        "%ExpError<=20%",
                        "%ExpError<=30%",
                        "%ExpError<=50%",
                        "Gini",
                        "Gini Norm",
                        "MAD",
                        "R Squared",
                        "R Squared 20/80",
                        "RMSE",
                        "RMSLE"
                    ]
                },
                "time_real": [
                    [
                        "1",
                        "0.01008"
                    ]
                ],
                "total_size": 400,
                "uid": "5214d010637aba16eb0bbb7a",
                "vertex_cnt": 2,
                "wsid": "5214d011637aba17000bbb7a"
            }]

        self.patchers = []
        self.app = None
        with patch.dict(EngConfig, {'TEST_MODE': True }, clear = False):
            import MMApp.api
            self.app = MMApp.api.app.test_client()

        self.secret_key_header = {'web-api-key' : MMApp.api.app.web_api_key}
        self.bad_secret_key_header = {'web-api-key' : 'bad-secret-key'}


        self.addCleanup(self.stopPatching)

    def stopPatching(self):
        super(TestApi, self).tearDown()
        for patcher in self.patchers:
            if patcher:
                patcher.stop()

    def test_post_leaderboard(self):
        leaderboard_item = self.leaderboard_items[0]
        with self.app as api:
            with patch('MMApp.api.ProjectService') as MockProjectService:
                mock_ps = MockProjectService.return_value
                mock_ps.create_leaderboard_item.return_value = '5239fc3d637aba7f1a53ed2f'
                data = {}
                data.update(leaderboard_item)
                data.pop('_id')
                response = api.post('/leaderboard',
                                    content_type='application/json',
                                    headers=self.secret_key_header,
                                    data = json.dumps(data) )

                self.assertEqual(response.status_code, 200, response.data)
                new_leaderboard_item = json.loads(response.data)

                self.assertIsNotNone(new_leaderboard_item)
                self.assertTrue('_id' in new_leaderboard_item)
                self.assertIsNotNone(new_leaderboard_item['_id'])
                self.assertDictContainsSubset(data, new_leaderboard_item)

                new_leaderboard_item.pop('_id')
                self.assertItemsEqual(new_leaderboard_item, data)

                self.assertTrue(mock_ps.create_leaderboard_item.called)

                response = api.post('/leaderboard',
                                    content_type='application/json',
                                    headers=self.bad_secret_key_header,
                                    data = json.dumps(data) )
                self.assertEqual(response.status_code, 403)

                mock_ps.create_leaderboard_item.side_effect = Exception()
                response = api.post('/leaderboard',
                                    content_type='application/json',
                                    headers=self.secret_key_header,
                                    data = json.dumps(data) )
                self.assertEqual(response.status_code, 400)

                data.pop('pid')
                response = api.post('/leaderboard',
                                    content_type='application/json',
                                    headers=self.secret_key_header,
                                    data = json.dumps(data) )
                self.assertEqual(response.status_code, 400)

    def test_put_leaderboard(self):
        # Arrange
        leaderboard_item = self.leaderboard_items[0]
        leaderboard_item['_id'] = str(leaderboard_item['_id'])
        leaderboard_item['pred'] = "predictions"
        l_id = leaderboard_item['_id']
        with self.app as api:
            with patch('MMApp.api.ProjectService') as MockProjectService:
            # Act
                mock_ps = MockProjectService.return_value
                mock_ps.save_leaderboard_item.return_value = None
                response = api.put('/leaderboard/%s' % l_id, content_type='application/json',
                                   headers=self.secret_key_header, data = json.dumps(leaderboard_item))
                # Assert
                self.assertEqual(response.status_code, 200, response.data)
                self.assertTrue(mock_ps.save_leaderboard_item.called)

                response = api.put('/leaderboard/%s' % l_id, content_type='application/json',
                                   headers=self.bad_secret_key_header, data = json.dumps(leaderboard_item))
                self.assertEqual(response.status_code, 403)

                mock_ps.save_leaderboard_item.side_effect = Exception()
                response = api.put('/leaderboard/%s' % l_id, content_type='application/json',
                                   headers=self.secret_key_header, data = json.dumps(leaderboard_item))
                self.assertEqual(response.status_code, 400)

                leaderboard_item.pop('pid')
                response = api.put('/leaderboard/%s' % l_id, content_type='application/json',
                                   headers=self.secret_key_header, data = json.dumps(leaderboard_item))
                self.assertEqual(response.status_code, 400)

    def test_save_leaderboard_item_calls_update_eda_ace_if_var_imp_info_present(self):
        leaderboard_item = self.leaderboard_items[0]
        leaderboard_item['_id'] = str(leaderboard_item['_id'])
        leaderboard_item['pred'] = "predictions"
        leaderboard_item['var_imp_info'] = 0.5
        leaderboard_item['var_imp_var'] = 'test_var'
        l_id = leaderboard_item['_id']

        with self.app as api:
            with patch('MMApp.api.ProjectService') as MockProjectService:
            # Act
                mock_ps = MockProjectService.return_value
                mock_ps.save_leaderboard_item.return_value = None
                response = api.put('/leaderboard/%s' % l_id, content_type='application/json',
                                   headers=self.secret_key_header, data = json.dumps(leaderboard_item))
                # Assert
                self.assertEqual(response.status_code, 200, response.data)
                self.assertTrue(mock_ps.update_eda_ace.called)


    def test_worker_projects(self):
        with self.app as api:
            #Arrange
            data = {'worker_type' : "secure-workers", 'worker_id' : "1"}
            # Act
            with patch('MMApp.api.QueueService') as MockQueueService:

                MockQueueService.get_worker_jobs.return_value = ["pid1:qid1"]
                MockQueueService.parse_worker_job.return_value = ("pid1","qid1")

                response = api.post('/worker_projects', content_type='application/json',
                                    headers=self.secret_key_header, data = json.dumps(data))

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.data, json.dumps({"project_ids": ["pid1"]}))

    def test_worker_projects_invalid_request(self):
        with self.app as api:
            #Arrange
            data = {'worker_id' : "1"}
            # Act
            with patch('MMApp.api.QueueService') as MockQueueService:

                MockQueueService.get_worker_jobs.return_value = ["pid1:qid1"]
                MockQueueService.parse_worker_job.return_value = ("pid1","qid1")

                response = api.post('/worker_projects', content_type='application/json',
                                    headers=self.secret_key_header, data = json.dumps(data))

                self.assertEqual(response.status_code, 400)

    def test_worker_projects_invalid_worker_type(self):
        with self.app as api:
            #Arrange
            data = {'worker_type' : "invalid", 'worker_id' : "1"}
            # Act
            with patch('MMApp.api.QueueService') as MockQueueService:

                MockQueueService.get_worker_jobs.return_value = ["pid1:qid1"]
                MockQueueService.parse_worker_job.return_value = ("pid1","qid1")

                response = api.post('/worker_projects', content_type='application/json',
                                    headers=self.secret_key_header, data = json.dumps(data))

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.data, json.dumps({"project_ids": []}))

    @patch('common.services.queue_service_base.ProjectService.assert_has_permission')
    def test_report_error(self,m):
        with self.app as api:
            #Arrange
            pid = TestBase.test_pid
            qid = TestBase.test_qid
            lid = TestBase.test_lid
            uid = TestBase.test_uid
            error = 'Bad error'
            data = {'pid' : pid, 'qid' : qid, 'lid': lid, 'uid': uid, 'command':'fit', 'worker_id': "1", 'message' : {'lid':lid, 'error': error}}
            # Act
            with patch('MMApp.api.QueueService') as MockQueueService:

                mock_qs = MockQueueService.return_value
                mock_qs.set_error.return_value = None
                response = api.post('/report/error', content_type='application/json',
                                    headers=self.secret_key_header, data = json.dumps(data))
                # Assert
                self.assertEqual(response.status_code, 200)

                # Make sure QueueService.set_error was called with the right parameters
                mock_qs.set_error.assert_called_once_with(qid, lid, data['message'], should_use_er=True)

                response = api.post('/report/error', content_type='application/json',
                                    headers=self.bad_secret_key_header, data = json.dumps(data))
                self.assertEqual(response.status_code, 403)

                response = api.post('/report/error', content_type='application/json',
                                    headers=self.secret_key_header, data = json.dumps({"missing_key": "value"}))
                self.assertEqual(response.status_code, 400)

                mock_qs.set_error.side_effect = Exception()
                response = api.post('/report/error', content_type='application/json',
                                    headers=self.secret_key_header, data = json.dumps(data))
                self.assertEqual(response.status_code, 400)

    def test_report_error_with_uid_none(self):
        with self.app as api:
            #Arrange
            pid = TestBase.test_pid
            qid = TestBase.test_qid
            lid = TestBase.test_lid
            error = 'Bad error'

            no_uid_data = {'pid' : pid, 'qid' : qid, 'lid': lid, 'uid': None, 'command':'fit', 'worker_id': "1", 'message' : {'lid':lid, 'error': error}}
            response = api.post('/report/error', content_type='application/json',
                        headers=self.secret_key_header, data = json.dumps(no_uid_data))

            self.assertEqual(response.status_code, 400, response.data)

    def test_report_error_with_uid_empty(self):
        with self.app as api:
            #Arrange
            pid = TestBase.test_pid
            qid = TestBase.test_qid
            lid = TestBase.test_lid
            error = 'Bad error'

            no_uid_data = {'pid' : pid, 'qid' : qid, 'lid': lid, 'uid': '', 'command':'fit', 'worker_id': "1", 'message' : {'lid':lid, 'error': error}}
            self.assertRaisesRegexp(RequestError, 'Invalid ID', api.post, '/report/error', content_type='application/json', headers=self.secret_key_header, data = json.dumps(no_uid_data))

    def test_report_complete(self):
        with self.app as api:
            #Arrange
            pid = TestBase.test_pid
            qid = TestBase.test_qid
            lid = TestBase.test_lid
            uid = TestBase.test_uid
            data = {'pid' : pid, 'qid' : qid, 'worker_id': '1', 'lid': lid,
                'uid':uid, 'dataset_id' : str(ObjectId())}
            # Act
            with patch('MMApp.api.QueueService') as MockQueueService:

                mock_qs = MockQueueService.return_value
                mock_qs.set_as_complete.return_value = None
                response = api.post('/report/complete', content_type='application/json',
                                    headers=self.secret_key_header, data = json.dumps(data))
                # Assert
                self.assertEqual(response.status_code, 200)
                # Make sure QueueService.set_as_complete was called with the right parameters
                mock_qs.set_as_complete.assert_called_once_with(qid, lid, None)

                response = api.post('/report/complete', content_type='application/json',
                                    headers=self.bad_secret_key_header, data = json.dumps(data))
                self.assertEqual(response.status_code, 403)

                response = api.post('/report/complete', content_type='application/json',
                                    headers=self.secret_key_header, data = json.dumps({"missing_key": "value"}))
                self.assertEqual(response.status_code, 400)

                mock_qs.set_as_complete.side_effect = Exception()
                response = api.post('/report/complete', content_type='application/json',
                                    headers=self.secret_key_header, data = json.dumps(data))
                self.assertEqual(response.status_code, 400)


    def test_report_started(self):
        with self.app as api:
            #Arrange
            pid = TestBase.test_pid
            qid = TestBase.test_qid
            data = {'pid' : pid, 'qid' : qid}
            # Act
            with patch('MMApp.api.QueueService') as MockQueueService:

                mock_qs = MockQueueService.return_value
                mock_qs.set_as_started.return_value = None
                response = api.post('/report/started', content_type='application/json',
                                    headers=self.secret_key_header, data = json.dumps(data))
                # Assert
                self.assertEqual(response.status_code, 200)
                # Make sure QueueService.set_as_complete was called with the right parameters
                mock_qs.set_as_started.assert_called_once_with(qid)

                response = api.post('/report/started', content_type='application/json',
                                    headers=self.bad_secret_key_header, data = json.dumps(data))
                self.assertEqual(response.status_code, 403)

                response = api.post('/report/started', content_type='application/json',
                                    headers=self.secret_key_header, data = json.dumps({"missing_key": "value"}))
                self.assertEqual(response.status_code, 400)

                mock_qs.set_as_started.side_effect = Exception()
                response = api.post('/report/started', content_type='application/json',
                                    headers=self.secret_key_header, data = json.dumps(data))
                self.assertEqual(response.status_code, 400)


    def test_get_queue(self):
        with self.app as api:
            with patch('MMApp.api.QueueService') as MockQueueService:
                pid = str(ObjectId())
                mock_qs = MockQueueService.return_value
                mock_qs.light_get.return_value = {'data': 'value'}
                response = api.get('/queue/{0}'.format(pid), headers=self.secret_key_header)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(json.loads(response.data)['status'], "OK")

                mock_qs.light_get.side_effect = Exception()
                response = api.get('/queue/{0}'.format(pid), headers=self.secret_key_header)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(json.loads(response.data)['status'], "FAIL")

    @patch('MMApp.api.time.sleep')
    def test_service(self, mock_sleep):
        with self.app as api:
            with patch('MMApp.api.QueueService') as MockQueueService:
                with patch('MMApp.api.ProjectService') as MockProjectService:
                    pid = str(ObjectId())
                    uid = str(ObjectId())
                    data = {'uid': uid}
                    mock_qs = MockQueueService.return_value
                    mock_qs.start_new_tasks.return_value = "count_value"
                    mock_ps = MockProjectService.return_value
                    mock_ps.get.return_value = {}

                    response = api.post('/service/{0}'.format(pid), content_type='application/json',
                        headers=self.secret_key_header, data=json.dumps(data))
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(json.loads(response.data), {'status': "OK", 'count': 'count_value'})

                    data = {'missing_uid': "value"}
                    response = api.post('/service/{0}'.format(pid), content_type='application/json',
                        headers=self.secret_key_header, data=json.dumps(data))
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(json.loads(response.data)['status'], "FAIL")

    @patch('MMApp.api.MMClient')
    @patch('MMApp.api.time.sleep')
    def test_next_steps(self, mock_sleep, mock_client):
        with self.app as api:
            pid = str(ObjectId())
            uid = str(ObjectId())
            data = {'uid': uid}
            response = api.post('/next_steps/{0}'.format(pid), content_type='application/json',
                headers=self.secret_key_header, data=json.dumps(data))
            self.assertEqual(response.status_code, 200)
            self.assertEqual(json.loads(response.data), {'status': "OK"})

            data = {'missing_uid': "value"}
            response = api.post('/next_steps/{0}'.format(pid), content_type='application/json',
                headers=self.secret_key_header, data=json.dumps(data))
            self.assertEqual(response.status_code, 200)
            self.assertEqual(json.loads(response.data)['status'], "FAIL")

    def test_report_log(self):
        with self.app as api:
            response = api.post('/report/log', content_type='application/json', data=json.dumps({'log_message': 'msg'}))
            self.assertEqual(response.status_code, 200)

    def test_get_project(self):
        with self.app as api:
            with patch('MMApp.api.ProjectService') as MockProjectService:
                MockProjectService.return_value.get.return_value = "response"
                pid = str(ObjectId())
                uid = str(ObjectId())
                response = api.get('/project/%s/%s' % (pid, uid), headers=self.secret_key_header)
                self.assertEqual(response.status_code, 200)

                response = api.get('/project/%s/%s' % (pid, uid), headers=self.bad_secret_key_header)
                self.assertEqual(response.status_code, 403)

    def test_local_get_data_url(self):
        with patch.dict(EngConfig, {'ENVIRON': 'local' }, clear = False) as mock_engconfig:
            with self.app as api:
                pid = str(ObjectId())
                uid = str(ObjectId())
                filename = '/projects/pid/raw/009e6c39-046c-42f1-ad5e-c174faac9634'

                with patch('MMApp.api.FileTransaction') as MockFileStorage:
                    instance = MockFileStorage.return_value
                    instance.url.return_value = None

                    payload = {'filename': filename}

                    #No credentials
                    response = api.get('/project/%s/%s/download?filename=%s' % (pid, uid, filename))
                    self.assertEqual(response.status_code, 403)

                    #Wrong credentials
                    response = api.get('/project/%s/%s/download?filename=%s' % (pid, uid, filename),
                        headers = self.bad_secret_key_header)
                    self.assertEqual(response.status_code, 403)

                    #Good credentials
                    response = api.get('/project/%s/%s/download?filename=%s' % (pid, uid, filename),
                        headers = self.secret_key_header)
                    self.assertEqual(response.status_code, 404)

    def test_aws_get_data_url(self):
        with patch.dict(EngConfig, {'ENVIRON': 'AWS' }, clear = False) as mock_engconfig:
            with self.app as api:
                pid = str(ObjectId())
                uid = str(ObjectId())
                filename = '/projects/pid/raw/009e6c39-046c-42f1-ad5e-c174faac9634'

                with patch('MMApp.api.FileTransaction') as MockFileStorage:
                    instance = MockFileStorage.return_value
                    url = 'http://'
                    instance.url.return_value = url

                    #No credentials
                    response = api.get('/project/%s/%s/download?filename=%s' % (pid, uid, filename))
                    self.assertEqual(response.status_code, 403)

                    #Wrong credentials
                    response = api.get('/project/%s/%s/download?filename=%s' %  (pid, uid, filename),
                        headers = self.bad_secret_key_header)
                    self.assertEqual(response.status_code, 403)

                    #Good credentials
                    response = api.get('/project/%s/%s/download?filename=%s' %  (pid, uid, filename),
                        headers = self.secret_key_header)
                    self.assertEqual(response.status_code, 200)

                    MockFileStorage.assert_called_once_with(filename, user_id=uid, project_id=pid)

                    instance.url.side_effect = Exception()
                    response = api.get('/project/%s/%s/download?filename=%s' %  (pid, uid, filename),
                        headers = self.secret_key_header)
                    self.assertEqual(response.status_code, 400)

    @patch('MMApp.api.UserService')
    def test_isonline(self, MockUserService):
        uid = ObjectId()
        with self.app as api:
            userservice = MockUserService.return_value
            userservice.is_online.return_value = True

            #authentication failure
            response = api.get('/online/%s' % (uid), headers=self.bad_secret_key_header)
            self.assertEqual(response.status_code, 403)

            #online
            response = api.get('/online/%s' % (uid), headers=self.secret_key_header)
            self.assertEqual(response.status_code, 200)

            #offline
            userservice.is_online.return_value = False
            response = api.get('/online/%s' % (uid), headers=self.secret_key_header)
            self.assertEqual(response.status_code, 404)

    def test_ide_setup_status_complete(self):
        with self.app as api:
            uid = 'd78716583729252263f383d9'
            pid = '52263f383d9d787165837292'
            status = IdeSetupStatus(IdeSetupStatus.COMPLETED)
            status.username = 'user'
            status.password = 'pass'
            status.location = '192.168.1.115:49211'

            data = {'command': 'ide_setup', 'status': status.status,
                    'username': status.username,
                    'password': status.password,
                    'location': status.location}
            payload = json.dumps(data)

            with patch('MMApp.api.IdeService') as MockIdeService:
                with patch('MMApp.api.IdeSetupStatus') as MockIdeSetupStatus:
                    MockIdeSetupStatus.return_value = sentinel.status

                    #authentication failure
                    response = api.post('/ide/{0}/{1}/status'.format(uid, pid),
                                    headers=self.bad_secret_key_header, data = payload, content_type = 'application/json')
                    self.assertEqual(response.status_code, 403)

                    #complete
                    response = api.post('/ide/{0}/{1}/status'.format(uid, pid),
                                    headers=self.secret_key_header, data = payload, content_type = 'application/json')
                    self.assertEqual(response.status_code, 200)

                    data.pop('command')
                    MockIdeSetupStatus.assert_called_once_with(data.pop('status'), **data)
                    MockIdeService.assert_called_once_with(uid, pid)
                    mock_ide_service = MockIdeService.return_value
                    mock_ide_service.set_status.assert_called_once_with(sentinel.status)

                    #missing command
                    response = api.post('/ide/{0}/{1}/status'.format(uid, pid),
                                    headers=self.secret_key_header, data = json.dumps(data), content_type = 'application/json')
                    self.assertEqual(response.status_code, 400)

                    #remove command
                    data['command'] = 'ide_remove'
                    data['status'] = status.status
                    response = api.post('/ide/{0}/{1}/status'.format(uid, pid),
                                    headers=self.secret_key_header, data = json.dumps(data), content_type = 'application/json')
                    self.assertEqual(response.status_code, 200)

                    #error
                    MockIdeService.return_value.set_remove_status.side_effect = Exception()
                    response = api.post('/ide/{0}/{1}/status'.format(uid, pid),
                                    headers=self.secret_key_header, data = json.dumps(data), content_type = 'application/json')
                    self.assertEqual(response.status_code, 400)


    def test_ide_setup_status_failed(self):
        with self.app as api:
            uid = 'd78716583729252263f383d9'
            pid = '52263f383d9d787165837292'
            status = IdeSetupStatus(IdeSetupStatus.FAILED)

            data = {'command' : 'ide_setup', 'status' : status.status}
            payload = json.dumps(data)

            with patch('MMApp.api.IdeService') as MockIdeService:
                with patch('MMApp.api.IdeSetupStatus') as MockIdeSetupStatus:
                    MockIdeSetupStatus.return_value = sentinel.status

                    response = api.post('/ide/{0}/{1}/status'.format(uid, pid),
                                    headers=self.secret_key_header, data = payload, content_type = 'application/json')

                    self.assertEqual(response.status_code, 200)

                    data.pop('command')
                    MockIdeSetupStatus.assert_called_once_with(data.pop('status'), **data)
                    MockIdeService.assert_called_once_with(uid, pid)
                    mock_ide_service = MockIdeService.return_value
                    mock_ide_service.set_status.assert_called_once_with(sentinel.status)

    def test_api_ping(self):
        with self.app as api:
            response = api.get('/ping?token=' + '123')
            self.assertEqual(response.status_code, 200)
            resp_data = json.loads(response.data)
            self.assertEqual(resp_data['response'], 'pong')
            self.assertEqual(resp_data['token'], '123')


    def test_queue_no_target_variable(self):
        pid = 'd78716583729252263f383d9'
        uid = '52263f383d9d787165837292'

        user_model_request = {
            "key": "1",
            "model_type": "user model 1",
            "modelfit": "function(response,data) {\n  library(gbm);\n  datasub = data[,c(\"VehYear\",\"VehBCost\")];\n  gbm.fit(datasub,response,n.trees=500, interaction.depth=10,shrinkage=0.1,bag.fraction=0.5,keep.data=FALSE, verbose=FALSE);\n}\n",
            "modelpredict": "function(model,data) {\n  library(gbm);\n  datasub = data[,c(\"VehYear\",\"VehBCost\")];\n  predict.gbm(model,datasub,n.trees=500,type=\"response\");\n}\n",
            "pid": pid,
            "uid": uid
        }

        with self.app as api:
            payload = json.dumps(user_model_request)

            with patch('MMApp.api.QueueService') as MockQueueService:
                with patch('MMApp.api.ProjectService') as MockProjectService:
                    queue_service = MockQueueService.return_value
                    blah = {'dataset_id':'ofdhfakjsdlfa', 'samplepct':64}
                    blah.update(user_model_request)
                    queue_service.validate_user_model.return_value = blah
                    queue_service.get.return_value = []
                    project_service = MockProjectService.return_value
                    project_service.read_leaderboard_item.return_value = {}

                    #authentication failure
                    response = api.post('/queue', headers=self.bad_secret_key_header,
                        data = payload, content_type = 'application/json')
                    self.assertEqual(response.status_code, 403)

                    response = api.post('/queue',
                                        headers=self.secret_key_header, data = payload, content_type = 'application/json')

                    self.assertEqual(response.status_code, 200)
                    data = json.loads(response.data)
                    self.assertEqual(data['message'], 'OK')

                    error = 'Target key has not been set'
                    queue_service.validate_user_model.side_effect = QueueException(error)
                    response = api.post('/queue',
                                        headers=self.secret_key_header, data = payload, content_type = 'application/json')
                    self.assertEqual(response.status_code, 200)
                    data = json.loads(response.data)
                    self.assertEqual(data['message'], error)

                    queue_service.validate_user_model.side_effect = Exception()
                    response = api.post('/queue',
                                        headers=self.secret_key_header, data = payload, content_type = 'application/json')
                    self.assertEqual(response.status_code, 400)

    def test_get_leaderboard(self):
        leaderboard_item = self.leaderboard_items[0]
        leaderboard_item['_id'] = str(leaderboard_item['_id'])
        l_id = leaderboard_item['_id']
        pid = 'd78716583729252263f383d9'
        uid = '52263f383d9d787165837292'
        with self.app as api:
            with patch('MMApp.api.ProjectService') as MockProjectService:
                mock_ps = MockProjectService.return_value
                mock_ps.get_leaderboard_item.return_value = leaderboard_item
                # Act
                response = api.get('/leaderboard/{0}/{1}/{2}'.format(l_id, pid, uid), headers=self.secret_key_header)
                # Assert
                self.assertEqual(response.status_code, 200)
                response_leaderboard_item = json.loads(response.data)
                self.assertDictContainsSubset(leaderboard_item, response_leaderboard_item)
                mock_ps.get_leaderboard_item.assert_called_once_with(l_id)

                mock_ps.get_leaderboard_item.side_effect = Exception()
                response = api.get('/leaderboard/{0}/{1}/{2}'.format(l_id, pid, uid), headers=self.secret_key_header)
                self.assertEqual(response.status_code, 400)

                response = api.get('/leaderboard/{0}/{1}/{2}'.format(l_id, pid, uid), headers=self.bad_secret_key_header)
                self.assertEqual(response.status_code, 403)

    def test_get_predictions(self):
        leaderboard_item = self.leaderboard_items[0]
        leaderboard_item['_id'] = str(leaderboard_item['_id'])
        l_id = leaderboard_item['_id']
        pid = 'd78716583729252263f383d9'
        uid = '52263f383d9d787165837292'
        part = '1'
        with self.app as api:
            with patch('MMApp.api.ProjectService') as MockProjectService:
                mock_ps = MockProjectService.return_value
                mock_ps.get_leaderboard_item.return_value = leaderboard_item
                mock_ps.get_predictions_by_part.return_value = "predictions"
                # Act
                response = api.get('/predictions/{0}/{1}/{2}/{3}'.format(l_id, pid, uid, part), headers=self.secret_key_header)
                # Assert
                self.assertEqual(response.status_code, 200)
                response_predictions = json.loads(response.data)

                self.assertEqual(response_predictions, mock_ps.get_predictions_by_part.return_value)

                mock_ps.get_predictions_by_part.assert_called_once_with(leaderboard_item, part)

                response = api.get('/predictions/{0}/{1}/{2}/{3}'.format(l_id, pid, uid, part), headers=self.bad_secret_key_header)
                self.assertEqual(response.status_code, 403)

                mock_ps.get_leaderboard_item.side_effect = Exception()
                response = api.get('/predictions/{0}/{1}/{2}/{3}'.format(l_id, pid, uid, part), headers=self.secret_key_header)
                self.assertEqual(response.status_code, 400)

    def test_get_dataset_list(self):
        pid = 'd78716583729252263f383d9'
        uid = '52263f383d9d787165837292'

        with self.app as api:
            # Act
            with patch('MMApp.api.FeaturelistService') as MockFeaturelistService:
                service = MockFeaturelistService.return_value
                service.get_all_datasets.return_value = []
                response = api.get('/project/{0}/{1}/dataset'.format(pid, uid), headers=self.secret_key_header)
                # Assert
                self.assertEqual(response.status_code, 200)
                service.get_all_datasets.assert_called_once_with(pid)

                response = api.get('/project/{0}/{1}/dataset'.format(pid, uid), headers=self.bad_secret_key_header)
                self.assertEqual(response.status_code, 403)

    def test_get_dataset_names(self):
        pid = 'd78716583729252263f383d9'
        uid = '52263f383d9d787165837292'

        with self.app as api:
            # Act
            with patch('MMApp.api.FeaturelistService') as MockFeaturelistService:
                service = MockFeaturelistService.return_value
                service.get_all_datasets.return_value = [{'name':'list1'},{'name':'universe'}]
                response = api.get('/project/{0}/{1}/datasets'.format(pid, uid), headers=self.secret_key_header)
                # Assert
                self.assertEqual(response.status_code, 200)
                self.assertEqual(json.loads(response.data),['list1'])
                service.get_all_datasets.assert_called_once_with(pid)

                response = api.get('/project/{0}/{1}/datasets'.format(pid, uid), headers=self.bad_secret_key_header)
                self.assertEqual(response.status_code, 403)

    def test_get_dataset_info(self):
        pid = 'd78716583729252263f383d9'
        uid = '52263f383d9d787165837292'

        with self.app as api:
            # Act
            with patch('MMApp.api.FeaturelistService') as MockFeaturelistService:
                service = MockFeaturelistService.return_value
                service.get_feature_names.return_value = []
                response = api.get('/project/{0}/{1}/dataset_info'.format(pid, uid), query_string={'name':'x'}, headers=self.secret_key_header)
                # Assert
                self.assertEqual(response.status_code, 200)
                service.get_feature_names.assert_called_once_with('x')

                response = api.get('/project/{0}/{1}/dataset_info'.format(pid, uid), query_string={'name':'x'}, headers=self.bad_secret_key_header)
                self.assertEqual(response.status_code, 403)

    def test_create_dataset(self):
        pid = 'd78716583729252263f383d9'
        uid = '52263f383d9d787165837292'

        with self.app as api:
            # Act
            with patch('MMApp.api.FeaturelistService') as MockFeaturelistService:
                service = MockFeaturelistService.return_value
                service.get_all_datasets.return_value = []
                request = {
                    'name': 'blah',
                    'columns': [],
                    'key': self.secret_key_header['web-api-key'],
                }
                response = api.post('/project/{0}/{1}/dataset'.format(pid, uid),
                    data=json.dumps(request),
                    headers=self.secret_key_header,
                    content_type = 'application/json')
                # Assert
                self.assertEqual(response.status_code, 200)
                service.create_dataset.assert_called_once_with('blah',[])

                request['key'] = self.bad_secret_key_header['web-api-key']
                response = api.post('/project/{0}/{1}/dataset'.format(pid, uid),
                    data=json.dumps(request),
                    headers=self.bad_secret_key_header,
                    content_type = 'application/json')
                self.assertEqual(response.status_code, 403)

    def test_pong(self):
        with self.app as api:
            with patch('MMApp.api.AdminTools') as MockAdminTools:
                admin_tools_service = MockAdminTools.return_value
                sender = 'worker'
                token = '12345'
                pong = {'sender' : sender, 'token': token}
                payload = json.dumps(pong)

                response = api.post('/pong', data = payload, content_type = 'application/json')

                self.assertEqual(response.status_code, 200)
                admin_tools_service.write_pong.assert_called_once_with(sender, token)

    def test_check_permissions(self):
        with self.app as api:
            with patch('MMApp.api.ProjectService') as MockProjectService:
                mock_ps = MockProjectService.return_value
                mock_ps.get_auth_data.return_value = {'data': "auth_data"}
                pid = str(ObjectId())
                payload = json.dumps({'uid': str(ObjectId())})

                response = api.post('/auth/{0}'.format(pid), data = payload, content_type = 'application/json')

                self.assertEqual(response.status_code, 200)
                self.assertEqual(json.loads(response.data), {'status': 'OK', 'data': "auth_data"})

                mock_ps.get_auth_data.side_effect = Exception()
                response = api.post('/auth/{0}'.format(pid), data = payload, content_type = 'application/json')
                self.assertEqual(response.status_code, 200)
                self.assertEqual(json.loads(response.data)['status'], 'FAIL')

    def test_get_task_code_from_id_list(self):
        with self.app as api:
            with patch('MMApp.api.UserTasks') as MockUserTasks:
                usertasks = MockUserTasks.return_value
                pid = str(ObjectId())
                payload = json.dumps({'uid': str(ObjectId()), 'task_version_ids': None})

                #authentication failure
                response = api.post('/get_task_code', data = payload,
                    content_type = 'application/json', headers=self.bad_secret_key_header)
                self.assertEqual(response.status_code, 403)

                #No task ids
                response = api.post('/get_task_code', data = payload,
                    content_type = 'application/json', headers=self.secret_key_header)
                self.assertEqual(response.status_code, 400)

                #get task access exception
                usertasks.get_tasks_by_ids.side_effect = TaskAccessError()
                payload = json.dumps({'uid': str(ObjectId()), 'task_version_ids': [1]})
                response = api.post('/get_task_code', data = payload,
                    content_type = 'application/json', headers=self.secret_key_header)
                self.assertEqual(response.status_code, 403)

                #exception
                usertasks.get_tasks_by_ids.side_effect = Exception()
                payload = json.dumps({'uid': str(ObjectId()), 'task_version_ids': [1]})
                response = api.post('/get_task_code', data = payload,
                    content_type = 'application/json', headers=self.secret_key_header)
                self.assertEqual(response.status_code, 400)

                #success
                usertasks.get_tasks_by_ids.side_effect = None
                usertasks.get_tasks_by_ids.return_value = ""
                payload = json.dumps({'uid': str(ObjectId()), 'task_version_ids': [1]})
                response = api.post('/get_task_code', data = payload,
                    content_type = 'application/json', headers=self.secret_key_header)
                self.assertEqual(response.status_code, 200)


    @patch('MMApp.api.MMClient')
    @patch('MMApp.api.DatasetService')
    def test_upload_data(self, MockDatasetService, MockClient):
        class fake_uploaded_file():
                dataset_id = ""

        with self.app as api:
            service = MockDatasetService.return_value
            service.process_file_upload.return_value = fake_uploaded_file()
            service.store_uploaded_file.return_value = True
            pid = str(ObjectId())
            uid = str(ObjectId())
            key = "x"
            feature_list_name = "None"
            csv = StringIO("x")
            payload = {'uid': uid, 'key': key, 'feature_list_name': feature_list_name, 'file': (csv, 'filename')}
            MockClient.return_value.add_data.return_value = BrokerServiceResponse()

            response = api.post('/project/{0}/data'.format(pid), data = payload,
                headers=self.secret_key_header,
                content_type = 'multipart/form-data')
            self.assertEqual(response.status_code, 200)
            self.assertTrue(MockClient.return_value.add_data.called)

            csv = StringIO("x")
            payload = {'uid': uid, 'key': key, 'feature_list_name': feature_list_name, 'file': (csv, 'filename')}
            response = api.post('/project/{0}/data'.format(pid), data = payload,
                headers=self.bad_secret_key_header,
                content_type = 'multipart/form-data')
            self.assertEqual(response.status_code, 403)

            payload = {'uid': uid, 'key': key, 'feature_list_name': feature_list_name}
            response = api.post('/project/{0}/data'.format(pid), data = payload,
                headers=self.secret_key_header,
                content_type = 'multipart/form-data')
            self.assertEqual(response.status_code, 404)

            service.store_uploaded_file.return_value = False
            csv = StringIO("x")
            payload = {'uid': uid, 'key': key, 'feature_list_name': feature_list_name, 'file': (csv, 'filename')}
            response = api.post('/project/{0}/data'.format(pid), data = payload,
                headers=self.secret_key_header,
                content_type = 'multipart/form-data')
            self.assertEqual(response.status_code, 404)

            service.process_file_upload.side_effect = Exception()
            csv = StringIO("x")
            payload = {'uid': uid, 'key': key, 'feature_list_name': feature_list_name, 'file': (csv, 'filename')}
            response = api.post('/project/{0}/data'.format(pid), data = payload,
                headers=self.secret_key_header,
                content_type = 'multipart/form-data')
            self.assertEqual(response.status_code, 404)

    def test_get_ide_setup_status(self):
        pid = 'd78716583729252263f383d9'
        uid = '52263f383d9d787165837292'

        with self.app as api:
            with patch('MMApp.api.IdeService') as MockIdeService:
                MockIdeService.return_value.get_status.return_value.to_dict.return_value = {'status': 'value'}
                response = api.get('/ide/{0}/{1}/status'.format(uid, pid), headers=self.secret_key_header)
                self.assertEqual(response.status_code, 200)

                response = api.get('/ide/{0}/{1}/status'.format(uid, pid), headers=self.bad_secret_key_header)
                self.assertEqual(response.status_code, 403)

                MockIdeService.return_value.get_status.side_effect = Exception()
                response = api.get('/ide/{0}/{1}/status'.format(uid, pid), headers=self.secret_key_header)
                self.assertEqual(response.status_code, 400)

    @patch('MMApp.api.FileTransaction')
    @patch('MMApp.api.DatasetService')
    def test_save_ide(self, MockDatasetService, MockFileObject):
        with self.app as api:
            service = MockDatasetService.return_value
            fobj = MockFileObject.return_value
            service.create_server_filename.return_value.local_path = "/dev/null"

            pid = str(ObjectId())
            uid = str(ObjectId())
            payload = {}

            response = api.post('/ide/{0}/{1}/environment'.format(uid, pid), data = payload,
                headers=self.secret_key_header,
                content_type = 'multipart/form-data')
            self.assertEqual(response.status_code, 400)

            response = api.post('/ide/{0}/{1}/environment'.format(uid, pid), data = payload,
                headers=self.bad_secret_key_header,
                content_type = 'multipart/form-data')
            self.assertEqual(response.status_code, 403)

            tar = StringIO("x")
            payload = {'file': (tar, 'filename')}
            response = api.post('/ide/{0}/{1}/environment'.format(uid, pid), data = payload,
                headers=self.secret_key_header,
                content_type = 'multipart/form-data')
            self.assertEqual(response.status_code, 200)

            service.create_server_filename.side_effect = Exception()
            tar = StringIO("x")
            payload = {'file': (tar, 'filename')}
            response = api.post('/ide/{0}/{1}/environment'.format(uid, pid), data = payload,
                headers=self.secret_key_header,
                content_type = 'multipart/form-data')
            self.assertEqual(response.status_code, 400)

    def test_get_ide_url(self):
        pid = 'd78716583729252263f383d9'
        uid = '52263f383d9d787165837292'

        with self.app as api:
            with patch('MMApp.api.FileTransaction') as MockFileObject:
                MockFileObject.return_value.url.return_value = "url"
                response = api.get('/ide/{0}/{1}/environment'.format(uid, pid), headers=self.secret_key_header)
                self.assertEqual(response.status_code, 200)

                response = api.get('/ide/{0}/{1}/environment'.format(uid, pid), headers=self.bad_secret_key_header)
                self.assertEqual(response.status_code, 403)

                MockFileObject.return_value.url.return_value = False
                response = api.get('/ide/{0}/{1}/environment'.format(uid, pid), headers=self.secret_key_header)
                self.assertEqual(response.status_code, 404)

    @patch('MMApp.api.QueueService')
    def test_accept_job(self, MockQueueService):
        pid = 'd78716583729252263f383d9'
        uid = '52263f383d9d787165837292'
        qid = '1'

        with self.app as api:
            queue_service = MockQueueService.return_value
            queue_service.tempstore.read.return_value = 'status'
            queue_service.inprogress.REQUEST_STATUS_OPEN = 'not_status'

            #authentication failure
            response = api.get('/accept_job/{0}/{1}?wid=1'.format(pid, qid), headers=self.bad_secret_key_header)
            self.assertEqual(response.status_code, 403)

            #missing worker id
            response = api.get('/accept_job/{0}/{1}?missing_wid=1'.format(pid, qid), headers=self.secret_key_header)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(json.loads(response.data), {'accepted': False})

            #status read error
            queue_service.tempstore.read.side_effect = Exception()
            response = api.get('/accept_job/{0}/{1}?wid=1'.format(pid, qid), headers=self.secret_key_header)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(json.loads(response.data), {'accepted': False})
            queue_service.tempstore.read.side_effect = None

            #request status not open
            response = api.get('/accept_job/{0}/{1}?wid=1'.format(pid, qid), headers=self.secret_key_header)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(json.loads(response.data), {'accepted': False})

            #database write failure
            queue_service.tempstore.create.side_effect = Exception()
            queue_service.inprogress.REQUEST_STATUS_OPEN = queue_service.tempstore.read.return_value
            response = api.get('/accept_job/{0}/{1}?wid=1'.format(pid, qid), headers=self.secret_key_header)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(json.loads(response.data), {'accepted': False})
            queue_service.tempstore.create.side_effect = None

            #accepted
            response = api.get('/accept_job/{0}/{1}?wid=1'.format(pid, qid), headers=self.secret_key_header)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(json.loads(response.data), {'accepted': True})

    @patch('MMApp.api.IdeService')
    def test_accept_ide_job(self, MockIdeService):
        pid = 'd78716583729252263f383d9'
        uid = '52263f383d9d787165837292'

        with self.app as api:
            ide = MockIdeService.return_value
            ide.confirm_setup_request.return_value = False
            ide.confirm_remove_request.return_value = False

            #authentication failure
            response = api.get('/accept_job/{0}/{1}/ide'.format(uid, pid), headers=self.bad_secret_key_header)
            self.assertEqual(response.status_code, 403)

            #missing arugements
            response = api.get('/accept_job/{0}/{1}/ide'.format(uid, pid), headers=self.secret_key_header)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(json.loads(response.data), {'accepted': False})

            #not confirmed
            response = api.get('/accept_job/{0}/{1}/ide?wid=1&command=ide_setup'.format(uid, pid), headers=self.secret_key_header)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(json.loads(response.data), {'accepted': False})

            #not confirmed
            response = api.get('/accept_job/{0}/{1}/ide?wid=1&command=ide_remove'.format(uid, pid), headers=self.secret_key_header)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(json.loads(response.data), {'accepted': False})

            #confirmed
            ide.confirm_remove_request.return_value = True
            response = api.get('/accept_job/{0}/{1}/ide?wid=1&command=ide_remove'.format(uid, pid), headers=self.secret_key_header)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(json.loads(response.data), {'accepted': True})

    @patch('MMApp.api.ProjectService')
    @patch('MMApp.api.DatasetService')
    def test_save_predictions(self,*args,**kwargs):
        lid = ObjectId(None)
        data= {'pid': str(ObjectId), 'uid': str(ObjectId), 'dataset_id': str(ObjectId(None)), 'meta':{'asdf':1}}
        with self.app as api:
            response = api.post('/predictions/%s'%lid, headers=self.bad_secret_key_header,
                data=json.dumps(data), content_type= 'application/json')
            self.assertEqual(response.status_code, 403)

            response = api.post('/predictions/%s'%lid, headers=self.secret_key_header,
                data=json.dumps({'missing_keys': 'value'}), content_type= 'application/json')
            self.assertEqual(response.status_code, 400)

            response = api.post('/predictions/%s'%lid, headers=self.secret_key_header,
                data=json.dumps(data), content_type= 'application/json')
            self.assertEqual(response.status_code, 200)

    @patch('MMApp.api.DatasetService')
    def test_notify_client(self,*args,**kwargs):
        data = {'result_id':'asdfasdfa', 'predictions':'asdasdfasdf'}
        with self.app as api:
            response = api.post('/notify_predictionapi_client', data=json.dumps(data), content_type= 'application/json')
            self.assertEqual(response.status_code, 200)

    @patch('MMApp.api.DatasetService')
    def test_notify_parallelcv(self, MockDatasetService):
        data = {'data': "value"}
        with self.app as api:
            MockDatasetService.return_value.notify_parallelcv.return_value = "response"
            response = api.post('/notify_parallelcv', data=json.dumps(data), content_type= 'application/json')
            self.assertEqual(response.status_code, 200)

    @patch('MMApp.api.IdeService')
    def test_register_worker(self, MockIdeService):
        with self.app as api:
            uid = ObjectId()
            pid = ObjectId()
            MockIdeService.register_worker.return_value = '1'

            data = {'type': 'ide', 'resources': {'ide': 1}}
            payload = json.dumps(data)

            #authentication failure
            response = api.post('/worker',
                            headers=self.bad_secret_key_header, data = payload, content_type = 'application/json')
            self.assertEqual(response.status_code, 403)

            #no type
            data = {'resources': {'ide': 1}}
            payload = json.dumps(data)
            response = api.post('/worker',
                            headers=self.secret_key_header, data = payload, content_type = 'application/json')
            self.assertEqual(response.status_code, 400)

            #success
            data = {'type': 'ide', 'resources': {'ide': 1}}
            payload = json.dumps(data)
            response = api.post('/worker',
                            headers=self.secret_key_header, data = payload, content_type = 'application/json')
            self.assertEqual(response.status_code, 200)

            #unsupported type
            data = {'type': 'unsupported', 'resources': {'ide': 1}}
            payload = json.dumps(data)
            response = api.post('/worker',
                            headers=self.secret_key_header, data = payload, content_type = 'application/json')
            self.assertEqual(response.status_code, 400)

            #registration failed
            MockIdeService.register_worker.return_value = None
            data = {'type': 'unsupported', 'resources': {'ide': 1}}
            payload = json.dumps(data)
            response = api.post('/worker',
                            headers=self.secret_key_header, data = payload, content_type = 'application/json')
            self.assertEqual(response.status_code, 400)

    @patch('MMApp.api.IdeService')
    def test_worker_resources(self, MockIdeService):
        with self.app as api:
            uid = ObjectId()
            pid = ObjectId()

            data = {'worker_type': 'ide', 'resources': None}
            payload = json.dumps(data)

            #authentication failure
            response = api.post('/worker/worker_id/resources',
                            headers=self.bad_secret_key_header, data = payload, content_type = 'application/json')
            self.assertEqual(response.status_code, 403)

            #no type
            data = {'resources': None}
            payload = json.dumps(data)
            response = api.post('/worker/worker_id/resources',
                            headers=self.secret_key_header, data = payload, content_type = 'application/json')
            self.assertEqual(response.status_code, 400)

            #success
            data = {'worker_type': 'ide', 'resources': None}
            payload = json.dumps(data)
            response = api.post('/worker/worker_id/resources',
                            headers=self.secret_key_header, data = payload, content_type = 'application/json')
            self.assertEqual(response.status_code, 200)

            #unsupported type
            data = {'worker_type': 'unsupported', 'resources': None}
            payload = json.dumps(data)
            response = api.post('/worker/worker_id/resources',
                            headers=self.secret_key_header, data = payload, content_type = 'application/json')
            self.assertEqual(response.status_code, 400)

            #ide service failed
            MockIdeService.remove_worker_resources.side_effect = Exception()
            data = {'worker_type': 'ide', 'resources': None}
            payload = json.dumps(data)
            response = api.post('/worker/worker_id/resources',
                            headers=self.secret_key_header, data = payload, content_type = 'application/json')
            self.assertEqual(response.status_code, 400)

    @patch('MMApp.api.IdeService')
    def test_worker_shutdown(self, MockIdeService):
        with self.app as api:
            uid = ObjectId()
            pid = ObjectId()

            data = {'worker_type': 'ide'}
            payload = json.dumps(data)

            #authentication failure
            response = api.post('/worker/worker_id/shutdown',
                            headers=self.bad_secret_key_header, data = payload, content_type = 'application/json')
            self.assertEqual(response.status_code, 403)

            #no type
            data = {}
            payload = json.dumps(data)
            response = api.post('/worker/worker_id/shutdown',
                            headers=self.secret_key_header, data = payload, content_type = 'application/json')
            self.assertEqual(response.status_code, 400)

            #success
            data = {'worker_type': 'ide'}
            payload = json.dumps(data)
            response = api.post('/worker/worker_id/shutdown',
                            headers=self.secret_key_header, data = payload, content_type = 'application/json')
            self.assertEqual(response.status_code, 200)

            #unsupported type
            data = {'worker_type': 'unsupported'}
            payload = json.dumps(data)
            response = api.post('/worker/worker_id/shutdown',
                            headers=self.secret_key_header, data = payload, content_type = 'application/json')
            self.assertEqual(response.status_code, 400)

            #ide service failed
            MockIdeService.remove_all_users_from_worker.side_effect = Exception()
            data = {'worker_type': 'ide'}
            payload = json.dumps(data)
            response = api.post('/worker/worker_id/shutdown',
                            headers=self.secret_key_header, data = payload, content_type = 'application/json')
            self.assertEqual(response.status_code, 400)


if __name__ == '__main__':
    unittest.main()


