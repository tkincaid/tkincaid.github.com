import unittest
import json
from mock import patch, mock_open
from urlparse import urljoin
from bson import ObjectId
import os

from common.api.api_client import APIClient, ApiError

class TestApiClient(unittest.TestCase):

    def setUp(self):
        self.addCleanup(self.stopPatching)
        self.api = APIClient(urljoin('http://127.0.0.1', 'api/v0/'))
        self.json_header = {'content-type': 'application/json'}

        lb_item_file = os.path.join(os.path.dirname(__file__),'../testdata/fixtures/leaderboard_item.json')

        with open(lb_item_file) as f:
            self.leaderboard_item = json.loads(f.read())


        self.patchers = []
        requests_patch = patch('common.api.api_client.requests')
        self.RequestsMock = requests_patch.start()
        self.patchers.append(requests_patch)


    def stopPatching(self):
        super(TestApiClient, self).tearDown()
        for patcher in self.patchers:
            if patcher:
                patcher.stop()

    def tearDown(self):
        pass

    def test_create_leaderboard_item(self):
        #Create

        mock_response = self.RequestsMock.post.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = self.leaderboard_item

        returned_leaderboard_item = self.api.create_leaderboard_item(self.leaderboard_item)

        self.assertItemsEqual(self.leaderboard_item, returned_leaderboard_item)
        mock_response.json.assert_called_once_with()
        headers = self.api.headers
        headers.update(self.json_header)
        self.RequestsMock.post.assert_called_once_with(self.api.host + 'leaderboard',
            headers=headers,
            data=json.dumps(self.leaderboard_item))

        mock_response.status_code = 403
        self.assertRaises(ApiError, self.api.create_leaderboard_item, self.leaderboard_item)

    def test_save_leaderboard_item(self):
        #update
        l_id = '12345'
        mock_response = self.RequestsMock.put.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = self.leaderboard_item

        self.assertTrue(self.api.save_leaderboard_item(l_id, self.leaderboard_item))
        headers = self.api.headers
        headers.update(self.json_header)
        self.RequestsMock.put.assert_called_once_with(self.api.host + 'leaderboard/%s' % l_id,
            headers=headers,
            data=json.dumps(self.leaderboard_item))

        mock_response.status_code = 403
        self.assertRaises(ApiError, self.api.save_leaderboard_item, l_id, self.leaderboard_item)

    def test_report_error(self):
        mock_response = self.RequestsMock.post.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = self.leaderboard_item

        pid = 12345
        qid = 67890
        error = 'ugly error'
        data = {'qid': qid, 'error': error, 'pid': pid}
        self.assertTrue(self.api.report_error(data))
        headers = self.api.headers
        headers.update(self.json_header)
        self.RequestsMock.post.assert_called_once_with(self.api.host + 'report/error',
            headers=headers,
            data=json.dumps(data))

    def test_report_complete(self):
        mock_response = self.RequestsMock.post.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = self.leaderboard_item

        pid = 12345
        qid = 67890
        data = {'qid': qid, 'pid': pid, 'worker_id': "1"}
        self.assertTrue(self.api.report_complete(data))
        headers = self.api.headers
        headers.update(self.json_header)
        self.RequestsMock.post.assert_called_once_with(self.api.host + 'report/complete',
            headers=headers,
            data=json.dumps(data))

    def test_report_msg(self):
        mock_response = self.RequestsMock.post.return_value
        mock_response.status_code = 200

        url = "url"
        msg_dict = {'msg': 'value'}
        self.assertTrue(self.api._report_msg(msg_dict, url))
        headers = self.api.headers
        headers.update(self.json_header)
        self.RequestsMock.post.assert_called_once_with(self.api.host + url,
            headers=headers,
            data=json.dumps(msg_dict))

        mock_response.status_code = 403
        self.assertRaises(ApiError, self.api._report_msg, msg_dict, url)

    def test_report_started(self):
        mock_response = self.RequestsMock.post.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = self.leaderboard_item

        pid = 12345
        qid = 67890
        data = {'qid': qid, 'pid': pid}
        self.assertTrue(self.api.report_started(data))
        headers = self.api.headers
        headers.update(self.json_header)
        self.RequestsMock.post.assert_called_once_with(self.api.host + 'report/started',
            headers=headers,
            data=json.dumps(data))

    def test_create_queue_item(self):
        #Arrange
        mock_response = self.RequestsMock.post.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = {'message':'OK'}

        user_model = {
        "key": "1",
        "model_type": "user model 1",
        "modelfit": "function(response,data) {\n  library(gbm);\n  datasub = data[,c(\"VehYear\",\"VehBCost\")];\n  gbm.fit(datasub,response,n.trees=500, interaction.depth=10,shrinkage=0.1,bag.fraction=0.5,keep.data=FALSE, verbose=FALSE);\n}\n",
        "modelpredict": "function(model,data) {\n  library(gbm);\n  datasub = data[,c(\"VehYear\",\"VehBCost\")];\n  predict.gbm(model,datasub,n.trees=500,type=\"response\");\n}\n",
        "pid": "523c4443637aba150b8e7654",
        "uid": "523c4443637aba1443c83403"
        }

        #Act
        self.assertTrue(self.api.create_queue_item(user_model))

        #Assert
        headers = self.api.headers
        headers.update(self.json_header)
        self.RequestsMock.post.assert_called_once_with(self.api.host + 'queue',
            headers=headers,
            data=json.dumps(user_model))

        mock_response.status_code = 403
        self.assertRaises(ApiError, self.api.create_queue_item, user_model)

        try:
            self.api.create_queue_item(user_model)
        except ApiError as e:
            self.assertTrue(isinstance(str(e), str))

    def test_get_data_url(self):
        mock_response = self.RequestsMock.get.return_value
        mock_response.status_code = 200
        filename = '009e6c39-046c-42f1-ad5e-c174faac9634'
        mock_response.json.return_value = 'file:///home/user/workspace/DataRobot/local_file_storage/refactor-workspace/%s' % filename

        pid = "1"
        uid = str(ObjectId())
        r = self.api.get_data_url(uid, pid, filename)

        self.json_header['web-api-key'] = self.api.headers['web-api-key']
        self.RequestsMock.get.assert_called_once_with(self.api.host +
                                                      'project/%s/%s/download' % (pid, uid),
                                                      params={'filename': filename},
                                                      headers = self.json_header)

        mock_response.status_code = 403
        self.assertRaises(ApiError, self.api.get_data_url, pid, uid, filename)

    def test_user_is_online(self):
        uid = str(ObjectId())
        mock_response = self.RequestsMock.get.return_value

        mock_response.status_code = 200
        r = self.api.user_is_online(uid)
        self.RequestsMock.get.assert_called_once_with(self.api.host + 'online/%s' % uid,
            headers = self.api.headers)
        self.assertTrue(r)

        mock_response.status_code = 403
        self.assertRaises(ApiError, self.api.user_is_online, uid)

    def test_user_is_offline(self):
        uid = str(ObjectId())
        mock_response = self.RequestsMock.get.return_value

        mock_response.status_code = 404
        r = self.api.user_is_online(uid)
        self.RequestsMock.get.assert_called_once_with(self.api.host + 'online/%s' % uid,
            headers = self.api.headers)
        self.assertFalse(r)

    def test_get_leaderboard_item(self):
        l_id = '12345'
        uid = str(ObjectId())
        pid = str(ObjectId())
        mock_response = self.RequestsMock.get.return_value
        mock_response.status_code = 200
        dataset_id = '009e6c39-046c-42f1-ad5e-c174faac9634'
        mock_response.json.return_value = 'leaderboard_item'

        r = self.api.get_leaderboard_item(l_id, pid, uid)

        self.RequestsMock.get.assert_called_once_with(self.api.host + 'leaderboard/%s/%s/%s' % (l_id, pid, uid),
            headers = self.api.headers)

        mock_response.status_code = 403
        self.assertRaises(ApiError, self.api.get_leaderboard_item, l_id, pid, uid)

    def test_get_task_code_without_version(self):
        uid = str(ObjectId())
        task_id = '1'
        mock_response = self.RequestsMock.get.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = 'task'

        r = self.api.get_task_code(uid, task_id)

        self.RequestsMock.get.assert_called_once_with(self.api.host + 'task_code/%s/%s' % (uid, task_id),
            headers = self.api.headers)
        self.assertEqual(r, mock_response.json.return_value)

        mock_response.status_code = 403
        self.assertRaises(ApiError, self.api.get_task_code, uid, task_id)

    def test_get_task_code_with_version(self):
        uid = str(ObjectId())
        task_id = '1'
        version_id = '1'
        mock_response = self.RequestsMock.get.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = 'task'

        r = self.api.get_task_code(uid, task_id, version_id)

        self.RequestsMock.get.assert_called_once_with(self.api.host + 'task_code/%s/%s/%s' % (uid, task_id, version_id),
            headers = self.api.headers)
        self.assertEqual(r, mock_response.json.return_value)

        mock_response.status_code = 403
        self.assertRaises(ApiError, self.api.get_task_code, uid, task_id, version_id)

    def test_get_predictions(self):
        lid = '12345'
        uid = str(ObjectId())
        pid = str(ObjectId())
        part = "1"
        mock_response = self.RequestsMock.get.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = 'predictions'

        r = self.api.get_predictions(lid, pid, uid, part)

        self.RequestsMock.get.assert_called_once_with(self.api.host + 'predictions/%s/%s/%s/%s' % (lid, pid, uid, part),
            headers = self.api.headers)

        mock_response.status_code = 403
        self.assertRaises(ApiError, self.api.get_predictions, lid, pid, uid, part)

    def test_report_log(self):
        self.assertTrue(self.api.report_log("log_message"))

    def test_accept_job(self):
        mock_response = self.RequestsMock.get.return_value
        mock_response.status_code = 200
        pid = '52263f383d9d787165837292'
        qid = '35'
        wid = '1'
        mock_response.json.return_value = {'accepted': True}

        r = self.api.accept_job(pid, qid, wid)

        self.RequestsMock.get.assert_called_once_with(self.api.host + 'accept_job/%s/%s' % (pid, qid),
            headers = self.api.headers, params={'wid': wid})

        mock_response.status_code = 403
        self.assertRaises(ApiError, self.api.accept_job, pid, qid, wid)

    def test_accept_ide_job(self):
        mock_response = self.RequestsMock.get.return_value
        mock_response.status_code = 200
        command = 'command'
        uid = str(ObjectId())
        pid = '52263f383d9d787165837292'
        wid = '1'
        mock_response.json.return_value = {'accepted': True}

        self.assertTrue(self.api.accept_ide_job(command, uid, pid, wid))

        self.RequestsMock.get.assert_called_once_with(self.api.host + 'accept_job/%s/%s/ide' % (uid, pid),
            headers = self.api.headers, params={'command': command, 'wid': wid})

        mock_response.status_code = 403
        self.assertRaises(ApiError, self.api.accept_ide_job, command, uid, pid, wid)

    def test_save_ide(self):
        mock_response = self.RequestsMock.post.return_value
        mock_response.status_code = 200

        local_filepath = 'file/created/by/worker/therefore/does/not/exist/here/DataRobot/temp_storage/file_name'
        pid = '52263f383d9d787165837292'
        uid = str(ObjectId())

        fake_open = mock_open()
        with patch('common.api.api_client.open', fake_open, create=True):
            self.assertTrue(self.api.save_ide(local_filepath, uid, pid))

            headers = self.api.headers
            self.RequestsMock.post.assert_called_once_with(self.api.host + 'ide/%s/%s/environment' % (uid, pid),
                headers=headers, files = {'file': fake_open.return_value})

            mock_response.status_code = 403
            self.assertRaises(ApiError, self.api.save_ide, local_filepath, uid, pid)

    def test_get_ide_url(self):
        uid = str(ObjectId())
        pid = str(ObjectId())
        mock_response = self.RequestsMock.get.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = 'ide_url'
        self.json_header['web-api-key'] = self.api.headers['web-api-key']

        self.assertEqual(self.api.get_ide_url(uid, pid), mock_response.json.return_value)

        self.RequestsMock.get.assert_called_once_with(self.api.host + 'ide/%s/%s/environment' % (uid, pid),
            headers = self.json_header)

        mock_response.status_code = 403
        self.assertRaises(ApiError, self.api.get_ide_url, uid, pid)

    def test_get_ide_url_not_found(self):
        uid = str(ObjectId())
        pid = str(ObjectId())
        mock_response = self.RequestsMock.get.return_value
        mock_response.status_code = 404
        mock_response.json.return_value = 'ide_url'
        self.json_header['web-api-key'] = self.api.headers['web-api-key']

        self.assertIsNone(self.api.get_ide_url(uid, pid))

        self.RequestsMock.get.assert_called_once_with(self.api.host + 'ide/%s/%s/environment' % (uid, pid),
            headers = self.json_header)

    def test_ide_setup_status(self):
        mock_response = self.RequestsMock.post.return_value
        mock_response.status_code = 200

        #Setup
        ide_status = {'action' : 'SETUP', 'status' : 'COMPLETED', 'credentials' : ('user', 'password'), 'location': '192.168.1.1:1364'}
        uid = '83729252263f383d9d787165'
        pid = '52263f383d9d787165837292'

        r = self.api.ide_setup_status(uid, pid, ide_status)

        headers = self.api.headers
        headers.update(self.json_header)
        self.RequestsMock.post.assert_called_once_with(self.api.host + 'ide/{0}/{1}/status'.format(uid, pid),
            data=json.dumps(ide_status),
            headers=headers)

        mock_response.status_code = 403
        self.assertRaises(ApiError, self.api.ide_setup_status, uid, pid, ide_status)

    def test_get_ide_setup_status(self):
        uid = str(ObjectId())
        pid = str(ObjectId())
        mock_response = self.RequestsMock.get.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = 'ide_status'
        self.json_header['web-api-key'] = self.api.headers['web-api-key']

        self.assertEqual(self.api.get_ide_setup_status(uid, pid), mock_response.json.return_value)

        self.RequestsMock.get.assert_called_once_with(self.api.host + 'ide/%s/%s/status' % (uid, pid),
            headers = self.json_header)

        mock_response.status_code = 403
        self.assertRaises(ApiError, self.api.get_ide_setup_status, uid, pid)

    def test_get_project(self):
        uid = str(ObjectId())
        pid = str(ObjectId())
        mock_response = self.RequestsMock.get.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = 'project'

        self.assertEqual(self.api.get_project(pid, uid), mock_response.json.return_value)

        self.RequestsMock.get.assert_called_once_with(self.api.host + 'project/%s/%s' % (pid, uid),
            headers = self.api.headers)

        mock_response.status_code = 403
        self.assertRaises(ApiError, self.api.get_project, pid, uid)

    def test_get_metadata(self):
        uid = str(ObjectId())
        pid = str(ObjectId())
        mock_response = self.RequestsMock.get.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = 'metadata'

        self.assertEqual(self.api.get_metadata(pid, uid), mock_response.json.return_value)

        self.RequestsMock.get.assert_called_once_with(self.api.host + 'project/%s/%s/dataset' % (pid, uid),
            headers = self.api.headers)

        mock_response.status_code = 403
        self.assertRaises(ApiError, self.api.get_metadata, pid, uid)

    def test_save_predictions(self):
        report = {'lid': "1"}
        mock_response = self.RequestsMock.post.return_value
        mock_response.status_code = 200

        self.assertTrue(self.api.save_predictions(report))

        headers = self.api.headers
        self.RequestsMock.post.assert_called_once_with(self.api.host + 'predictions/%s' % "1",
            headers=headers, data=json.dumps({}))

        mock_response.status_code = 403
        self.assertRaises(ApiError, self.api.save_predictions, {'lid': "1"})

    def test_notify_client(self):
        report = "report"
        mock_response = self.RequestsMock.post.return_value
        mock_response.status_code = 200

        self.assertIsNone(self.api.notify_client(report))

        headers = self.api.headers
        self.RequestsMock.post.assert_called_once_with(self.api.host + 'notify_predictionapi_client',
            headers=headers, data=json.dumps(report))

        mock_response.status_code = 403
        self.assertRaises(ApiError, self.api.notify_client, report)

    def test_notify_parallelcv(self):
        report = "report"
        mock_response = self.RequestsMock.post.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = {'complete': 'value'}

        self.assertEqual(self.api.notify_parallelcv(report), 'value')

        headers = self.api.headers
        self.RequestsMock.post.assert_called_once_with(self.api.host + 'notify_parallelcv',
            headers=headers, data=json.dumps(report))

        mock_response.status_code = 403
        self.assertRaises(ApiError, self.api.notify_parallelcv, report)

    def test_ide_logout_status(self):
        mock_response = self.RequestsMock.post.return_value
        mock_response.status_code = 200

        #Logout
        ide_status = {'action' : 'REMOVE', 'status' : 'COMPLETED'}
        uid = '83729252263f383d9d787165'
        pid = '52263f383d9d787165837292'

        r = self.api.ide_setup_status(uid, pid, ide_status)

        headers = self.api.headers
        headers.update(self.json_header)
        self.RequestsMock.post.assert_called_once_with(self.api.host + 'ide/{0}/{1}/status'.format(uid, pid),
            data=json.dumps(ide_status),
            headers=headers)

    def test_ping(self):
        mock_response = self.RequestsMock.post.return_value
        mock_response.status_code = 200

        #Logout
        sender = 'ide_worker'
        token = 'abc'
        pong = {'sender' : sender, 'token': token}

        r = self.api.pong(sender, token)

        headers = self.api.headers
        headers.update(self.json_header)
        self.RequestsMock.post.assert_called_once_with(self.api.host + 'pong',
            data=json.dumps(pong),
            headers=headers)

        mock_response.status_code = 403
        self.assertRaises(ApiError, self.api.pong, sender, token)

    def test_get_task_code_from_id_list(self):
        mock_response = self.RequestsMock.post.return_value
        mock_response.status_code = 200

        uid = '83729252263f383d9d787165'
        task_version_ids = set([1,2,3])
        r = self.api.get_task_code_from_id_list(uid, task_version_ids)

        mock_response.status_code = 403
        self.assertRaises(ApiError, self.api.get_task_code_from_id_list, uid, task_version_ids)

    def test_register_worker(self):
        mock_response = self.RequestsMock.post.return_value
        mock_response.json.return_value = {'worker_id': '1'}
        mock_response.status_code = 200

        #Setup
        worker_info = {'type': 'ide', 'resources': {'ide': 1}}
        uid = '83729252263f383d9d787165'
        pid = '52263f383d9d787165837292'

        self.assertEqual(self.api.register_worker(worker_info), '1')

        headers = self.api.headers
        self.RequestsMock.post.assert_called_once_with(self.api.host + 'worker',
            data=json.dumps(worker_info),
            headers=headers)

        mock_response.status_code = 403
        self.assertRaises(ApiError, self.api.register_worker, worker_info)

    def test_set_worker_resources(self):
        mock_response = self.RequestsMock.post.return_value
        mock_response.status_code = 200

        data = {'worker_type': 'ide', 'resources': None}

        self.assertTrue(self.api.set_worker_resources("worker_id", data['worker_type']))

        headers = self.api.headers
        headers.update(self.json_header)
        self.RequestsMock.post.assert_called_once_with(self.api.host + 'worker/worker_id/resources',
            data=json.dumps(data),
            headers=headers)

        mock_response.status_code = 403
        self.assertRaises(ApiError, self.api.set_worker_resources, "worker_id", "ide")

    def test_worker_shutdown(self):
        mock_response = self.RequestsMock.post.return_value
        mock_response.status_code = 200

        data = {'worker_type': 'ide'}

        self.assertTrue(self.api.worker_shutdown("worker_id", data['worker_type']))

        headers = self.api.headers
        headers.update(self.json_header)
        self.RequestsMock.post.assert_called_once_with(self.api.host + 'worker/worker_id/shutdown',
            data=json.dumps(data),
            headers=headers)

        mock_response.status_code = 403
        self.assertRaises(ApiError, self.api.worker_shutdown, "worker_id", "ide")

    def test_api_error_is_pickable(self):
        import cPickle as pickle

        error = ApiError('Could not POST', 400, 'http://datarobot-app.com/does-not-matter', 'POST')
        serialized_error = pickle.dumps(error)
        desirialized_error = pickle.loads(serialized_error)
        self.assertEqual(error.msg, desirialized_error.msg)
        self.assertEqual(error.status_code, desirialized_error.status_code)
        self.assertEqual(error.url, desirialized_error.url)
        self.assertEqual(error.method, desirialized_error.method)

if __name__ == '__main__':
    unittest.main()
