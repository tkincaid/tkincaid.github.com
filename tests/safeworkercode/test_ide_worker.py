import unittest
import os
import pandas
from bson import ObjectId

from mock import Mock, patch, call, DEFAULT, mock_open, sentinel

from safeworkercode.ide_worker import IDEWorker, ApiError
from safeworkercode.user_config import EngConfig

class fake_namedtempfile():
    name = "filename"
    def close(self):
        pass
    def write(self, x):
        pass

class fake_status():
    python_location = ""
    rstudio_location = ""

class fake_read(object):
    def __init__(self):
        self.i = True

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self, x):
        if self.i == True:
            self.i = False
            return True
        return self.i

class TestIDEWorker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        pipe = Mock()

        self.worker = IDEWorker(1, {'uid': str(ObjectId()), 'pid': str(ObjectId()), 'key': 'x'}, pipe)

    def test_validate_request(self):
        self.worker.req_dict = dict.fromkeys(['pid', 'dataset_id'], 'x')
        self.assertRaisesRegexp(ValueError, 'Invalid request', self.worker.validate_request)

        self.worker.req_dict = dict.fromkeys(['uid', 'dataset_id'], 'x')
        self.assertRaisesRegexp(ValueError, 'Invalid request', self.worker.validate_request)

        self.worker.req_dict = dict.fromkeys(['uid', 'pid'], 'x')
        self.worker.validate_request()

    @patch('safeworkercode.ide_worker.shutil.rmtree')
    @patch('safeworkercode.ide_worker.passwd')
    def test_new_ide_setup(self, fake_passwd, fake_rmtree):
        with patch.multiple(self.worker,
                validate_request = DEFAULT,
                set_ide_setup_result = DEFAULT,
                create_clean_dir = DEFAULT,
                restore_environment = DEFAULT,
                make_dirs = DEFAULT,
                get_available_port = DEFAULT,
                create_password = DEFAULT,
                get_python_url = DEFAULT,
                create_default_files = DEFAULT,
                create_r_workspace_in_lxc = DEFAULT,
                create_python_workspace_in_lxc = DEFAULT,
                launch_ipython = DEFAULT,
                launch_rstudio = DEFAULT,
                get_location = DEFAULT,
                transfer_files_from_storage = DEFAULT,
                remove_containers = DEFAULT,
                try_dir_remove=DEFAULT,
                clean_host=DEFAULT) as mocks:

            # Arrange
            credentials = (self.worker.req_dict['uid'], 'password')
            python_url = 'python'
            port = 8889
            location = 'ip:8889'
            mocks['remove_containers'].return_value = False
            mocks['restore_environment'].return_value = False
            mocks['create_password'].return_value = credentials[1]
            mocks['get_available_port'].return_value = port
            mocks['get_python_url'].return_value = python_url
            mocks['get_location'].return_value = location
            fake_passwd.return_value = 'HASHPASS'

            # Act
            res = self.worker.ide_setup({})
            self.assertIsNone(res)

            template_data = {'PASSWORD': credentials[1],
                             'PASSWORD_HASH': fake_passwd(),
                             'WEB_API_LOCATION': EngConfig['WEB_API_LOCATION'],
                             'QUEUE_WEB_API_URL': EngConfig['IDE_WORKER_QUEUE_WEB_API'],
                             'TASK_WEB_API_URL': EngConfig['IDE_WORKER_TASK_WEB_API'],
                             'UID': self.worker.req_dict['uid'],
                             'PID': self.worker.req_dict['pid'],
                             'KEY': 'x',
                             'PYTHON_URL': python_url,
                             'MOUNT_DIR': EngConfig['IDE_MOUNT_DIR'],
                             'HOST_EUID': self.worker.euid,
                             'RSTUDIO_INIT_FILE': os.path.join(EngConfig['IDE_MOUNT_DIR'], EngConfig['RSTUDIO_INIT_FILE']),
                             'IPYTHON_INIT_FILE': os.path.join(EngConfig['IDE_MOUNT_DIR'], EngConfig['IPYTHON_INIT_FILE'])}

            # Assert
            self.assertTrue(mocks['validate_request'].called)
            mocks['create_default_files'].assert_called_once_with(template_data)
            mocks['create_r_workspace_in_lxc'].assert_called_once_with(template_data)
            mocks['create_python_workspace_in_lxc'].assert_called_once_with(template_data)
            mocks['launch_ipython'].assert_called_once_with(port)
            mocks['launch_rstudio'].assert_called_once_with(port)

            mocks['set_ide_setup_result'].assert_has_calls([
                call(credentials=credentials, rstudio_location=location, python_location=location, status = self.worker.COMPLETED)
                    ])

    def test_get_location(self):
        port = '1324'
        ip = '192.168.5.113'
        with patch.dict(EngConfig, {'LOCAL_IP': ip}, clear=True):
            location = self.worker.get_location(port)

            self.assertEqual(location, '%s:%s' % (ip, port) )

    def test_get_valid_r_identifier(self):

        # Multiple non-word chars collapse to one underscore
        filename = 'home/ulises/workspace/DataRobot/tests/testdata/bad chars !#$#^$#%&^ and spaces'
        actual = self.worker.get_valid_r_identifier(filename)
        expected = 'home_ulises_workspace_DataRobot_tests_testdata_bad_chars_and_spaces'
        self.assertEqual(expected, actual)

        #Only the first n chars are taken (The R language guide does not mention the max length ....)
        filename = 'x' * 500
        actual = self.worker.get_valid_r_identifier(filename)
        expected = 'x' * self.worker.r_max_identifier_length
        self.assertEqual(expected, actual)

        #Ensure only valid identifiers are produced (from the R language reference, 10.3.2 Identifiers):
        # Identifiers consist of a sequence of letters, digits, the period ('.') and the underscore. They must
        # not start with a digit nor underscore, nor with a period followed by a digit.

        filename = '_' + ('x' * 500)
        actual = self.worker.get_valid_r_identifier(filename)
        expected = 'x' * self.worker.r_max_identifier_length
        self.assertEqual(expected, actual)

        filename = '.' + ('x' * 500)
        actual = self.worker.get_valid_r_identifier(filename)
        expected = 'x' * self.worker.r_max_identifier_length
        self.assertEqual(expected, actual)

        filename = '8' + ('x' * 500)
        actual = self.worker.get_valid_r_identifier(filename)
        expected = 'x' * self.worker.r_max_identifier_length
        self.assertEqual(expected, actual)

    @patch('safeworkercode.ide_worker.shutil.copyfile')
    @patch('safeworkercode.ide_worker.os.path.join')
    def test_create_r_workspace_in_lxc(self, mock_os, mock_copy):
        with patch.multiple(self.worker, create_file_from_template=DEFAULT) as mocks:
            self.worker.create_r_workspace_in_lxc({})
            self.assertTrue(mocks['create_file_from_template'].called)
            self.assertTrue(mock_copy.called)

    def test_clean_host(self):
        with patch.multiple(self.worker, remove_orphan_sessions=DEFAULT, remove_orphan_lxc_contexts=DEFAULT) as mocks:
            self.worker.clean_host()
            self.assertTrue(mocks['remove_orphan_sessions'].called)
            self.assertTrue(mocks['remove_orphan_lxc_contexts'].called)

    @patch('safeworkercode.ide_worker.os.path.join')
    def test_create_python_workspace_in_lxc(self, mock_os):
        with patch.multiple(self.worker, create_file_from_template=DEFAULT) as mocks:
            self.worker.create_python_workspace_in_lxc({})
            self.assertTrue(mocks['create_file_from_template'].called)

    def test_set_ide_setup_failed_result(self):
        with patch.object(self.worker, 'api') as mock_api:
            self.worker.set_ide_setup_result(status = IDEWorker.FAILED)

            expected_ide_status = {'command':'ide_setup','status': 'FAILED'}
            mock_api.ide_setup_status.assert_called_once_with(self.worker.req_dict['uid'], self.worker.req_dict['pid'], expected_ide_status)

    def test_set_ide_setup_successful_result(self):
        credentials = ('u', 'p')
        location = '127.0.0.1:1354'
        python_location = '127.0.0.1:1355'
        with patch.object(self.worker, 'api') as mock_api:
            self.worker.set_ide_setup_result(status = IDEWorker.COMPLETED,
                                             credentials = credentials, rstudio_location = location,
                                             python_location = python_location)

            expected_ide_status = {'command':'ide_setup', 'status': 'COMPLETED', 'username' : credentials[0], 'password': credentials[1],
                                   'rstudio_location':location, 'python_location': python_location}
            mock_api.ide_setup_status.assert_called_once_with(self.worker.req_dict['uid'], self.worker.req_dict['pid'], expected_ide_status)

    @patch('safeworkercode.ide_worker.sys.exit')
    def test_user_signal(self, mock_exit):
        self.worker.user_signal("signum", "stack_frame")
        self.assertTrue(mock_exit.called)

    def test_accept_job(self):
        with patch.object(self.worker, 'api') as mock_api:
            mock_api.accept_ide_job.return_value = True
            self.worker.req_dict['command'] = "ping"
            self.assertTrue(self.worker.accept_job())
            self.worker.req_dict['command'] = "shutdown"
            self.assertTrue(self.worker.accept_job())
            self.worker.req_dict['command'] = "not ping"
            self.assertTrue(self.worker.accept_job())

    def test_ping(self):
        with patch.object(self.worker, 'api') as mock_api:
            self.worker.ping({'token': 1})
            self.assertTrue(mock_api.pong.called)

    def test_invalid_setup_request(self):
        with patch.multiple(self.worker,
                            validate_request=DEFAULT,
                            set_ide_setup_result=DEFAULT,
                            create_clean_dir=DEFAULT,
                            remove_containers=DEFAULT,
                            clean_host=DEFAULT) as mocks:
            mocks['validate_request'].side_effect = ValueError()
            mocks['create_clean_dir'].side_effect = Exception()
            self.worker.ide_setup({})
            mocks['set_ide_setup_result'].assert_called_once_with(status=self.worker.FAILED)

    @patch('safeworkercode.ide_worker.shutil.rmtree')
    def test_setup_error(self, mock_rmtree):
        with patch.multiple(self.worker,
                            validate_request=DEFAULT,
                            set_ide_setup_result=DEFAULT,
                            create_clean_dir=DEFAULT,
                            remove_containers=DEFAULT,
                            try_dir_remove=DEFAULT,
                            clean_host=DEFAULT) as mocks:
            mocks['validate_request'].return_value = True
            mocks['create_clean_dir'].side_effect = Exception()
            self.worker.ide_setup({})
            self.assertTrue(mocks['remove_containers'].called)
            self.assertTrue(mocks['try_dir_remove'].called)
            mocks['set_ide_setup_result'].assert_called_once_with(status=self.worker.FAILED)

    @patch('safeworkercode.ide_worker.shutil.rmtree')
    def test_setup_error_rm_fail(self, mock_rmtree):
        with patch.multiple(self.worker,
                            validate_request=DEFAULT,
                            set_ide_setup_result=DEFAULT,
                            create_clean_dir=DEFAULT,
                            remove_containers=DEFAULT,
                            clean_host=DEFAULT) as mocks:
            mocks['validate_request'].return_value = True
            mocks['create_clean_dir'].side_effect = Exception()
            mock_rmtree.side_effect = Exception()
            self.worker.ide_setup({})
            self.assertTrue(mocks['remove_containers'].called)
            mocks['set_ide_setup_result'].assert_called_once_with(status=self.worker.FAILED)

    @patch('safeworkercode.ide_worker.shutil.rmtree')
    def test_invalid_remove_request(self, mock_rmtree):
        with patch.multiple(self.worker,
                            validate_request=DEFAULT,
                            set_ide_remove_result=DEFAULT,
                            wait_for_setup=DEFAULT,
                            remove_containers=DEFAULT,
                            ide_save=DEFAULT,
                            try_dir_remove=DEFAULT) as mocks:
            mocks['validate_request'].side_effect = ValueError()
            mocks['wait_for_setup'].return_value = False
            self.worker.ide_remove({})
            mocks['set_ide_remove_result'].assert_called_once_with(status=self.worker.FAILED)

    @patch('safeworkercode.ide_worker.os.path.isdir')
    @patch('safeworkercode.ide_worker.shutil.rmtree')
    def test_remove_wait_for_setup(self, mock_rmtree, mock_isdir):
        with patch.multiple(self.worker,
                            validate_request=DEFAULT,
                            set_ide_remove_result=DEFAULT,
                            wait_for_setup=DEFAULT,
                            remove_containers=DEFAULT,
                            ide_save=DEFAULT,
                            try_dir_remove=DEFAULT) as mocks:
            mocks['validate_request'].return_value = True
            mocks['wait_for_setup'].return_value = False
            self.worker.ide_remove({})
            mocks['set_ide_remove_result'].assert_called_once_with(self.worker.FAILED)
            self.assertFalse(mocks['remove_containers'].called)

    @patch('safeworkercode.ide_worker.os.path.isdir')
    @patch('safeworkercode.ide_worker.shutil.rmtree')
    def test_remove_save_error(self, mock_rmtree, mock_isdir):
        with patch.multiple(self.worker,
                            validate_request=DEFAULT,
                            set_ide_remove_result=DEFAULT,
                            wait_for_setup=DEFAULT,
                            remove_containers=DEFAULT,
                            ide_save=DEFAULT,
                            try_dir_remove=DEFAULT) as mocks:
            mocks['validate_request'].return_value = True
            mocks['wait_for_setup'].return_value = True
            mocks['ide_save'].side_effect = Exception()
            mock_isdir.return_value = True
            self.worker.req_dict['save'] = True
            self.worker.ide_remove({})
            mocks['try_dir_remove'].assert_called_once_with(self.worker.lxc_context)
            mocks['set_ide_remove_result'].assert_called_once_with(self.worker.FAILED)

    @patch('safeworkercode.ide_worker.os.path.isdir')
    @patch('safeworkercode.ide_worker.shutil.rmtree')
    def test_remove_save_fail(self, mock_rmtree, mock_isdir):
        with patch.multiple(self.worker,
                            validate_request=DEFAULT,
                            set_ide_remove_result=DEFAULT,
                            wait_for_setup=DEFAULT,
                            remove_containers=DEFAULT,
                            ide_save=DEFAULT,
                            try_dir_remove=DEFAULT) as mocks:
            mocks['validate_request'].return_value = True
            mocks['wait_for_setup'].return_value = True
            mocks['ide_save'].return_value = False
            mock_isdir.return_value = True
            self.worker.req_dict['save'] = True
            self.worker.ide_remove({})
            mocks['try_dir_remove'].assert_called_once_with(self.worker.lxc_context)
            mocks['set_ide_remove_result'].assert_called_once_with(self.worker.COMPLETED)

    @patch('safeworkercode.ide_worker.os.path.isdir')
    @patch('safeworkercode.ide_worker.shutil.rmtree')
    def test_remove_cleanup_error(self, mock_rmtree, mock_isdir):
        with patch.multiple(self.worker,
                            validate_request=DEFAULT,
                            set_ide_remove_result=DEFAULT,
                            wait_for_setup=DEFAULT,
                            remove_containers=DEFAULT,
                            ide_save=DEFAULT,
                            try_dir_remove=DEFAULT) as mocks:
            mocks['validate_request'].return_value = True
            mocks['wait_for_setup'].return_value = True
            mock_isdir.return_value = True
            self.worker.req_dict['save'] = True
            mocks['try_dir_remove'].return_value = False
            self.worker.ide_remove({})
            mocks['set_ide_remove_result'].assert_called_once_with(self.worker.FAILED)

    @patch('safeworkercode.ide_worker.time')
    def test_wait_for_setup(self, mock_time):
        with patch.multiple(self.worker,
                            get_ide_setup_status=DEFAULT) as mocks:
            mock_time.time = (i for i in xrange(100)).next
            self.worker.WAIT_FOR_SETUP_TIMEOUT = 2
            mocks['get_ide_setup_status'].return_value = {'status': self.worker.COMPLETED}
            self.assertTrue(self.worker.wait_for_setup())
            #FAILED is not actually a status that can be returned
            mocks['get_ide_setup_status'].return_value = {'status': self.worker.FAILED}
            self.assertFalse(self.worker.wait_for_setup())
            mocks['get_ide_setup_status'].return_value = {'status': self.worker.IN_PROCESS}
            self.assertFalse(self.worker.wait_for_setup())
            self.assertTrue(mock_time.sleep.called)

    @patch('safeworkercode.ide_worker.requests')
    def test_can_connect(self, mock_requests):
        mock_requests.get.return_value.status_code = 200
        self.assertTrue(self.worker.can_connect(fake_status()))
        mock_requests.get.return_value.status_code = 403
        self.assertFalse(self.worker.can_connect(fake_status()))

    @patch('safeworkercode.ide_worker.os')
    @patch('safeworkercode.ide_worker.tarfile')
    @patch('safeworkercode.ide_worker.shutil.rmtree')
    def test_ide_save(self, fake_rmtree, fake_tarfile, fake_os):
        with patch.multiple(self.worker,  api=DEFAULT, try_file_remove=DEFAULT) as mocks:
            self.assertTrue(self.worker.ide_save())

            fake_os.walk.return_value = [("", "", [""])]
            fake_os.path.getsize.return_value = 12*1024*1024
            fake_os.remove.return_value = True
            self.assertTrue(self.worker.ide_save())
            self.assertTrue(mocks['try_file_remove'].called)

            fake_os.remove.side_effect = OSError()
            self.assertFalse(self.worker.ide_save())

    @patch('safeworkercode.ide_worker.os')
    @patch('safeworkercode.ide_worker.tarfile')
    @patch('safeworkercode.ide_worker.shutil.rmtree')
    def test_ide_save_error(self, fake_rmtree, fake_tarfile, fake_os):
        with patch.multiple(self.worker,  api=DEFAULT, try_file_remove=DEFAULT, try_dir_remove=DEFAULT) as mocks:
            mocks['api'].save_ide.side_effect = Exception()
            fake_os.walk.return_value = [("", "", [""])]
            fake_os.path.getsize.return_value = 2*1024*1024
            fake_os.remove.return_value = True
            self.assertFalse(self.worker.ide_save())
            self.assertTrue(mocks['try_file_remove'].called)

    @patch('safeworkercode.ide_worker.os')
    @patch('safeworkercode.ide_worker.tarfile')
    @patch('safeworkercode.ide_worker.shutil.rmtree')
    def test_ide_save_not_exist(self, fake_rmtree, fake_tarfile, fake_os):
        with patch.multiple(self.worker,  api=DEFAULT, try_file_remove=DEFAULT, try_dir_remove=DEFAULT) as mocks:
            fake_os.path.exists.return_value = False
            self.assertFalse(self.worker.ide_save())

    @patch('safeworkercode.ide_worker.os')
    @patch('safeworkercode.ide_worker.tarfile')
    @patch('safeworkercode.ide_worker.shutil.rmtree')
    def test_ide_save_dir_remove_fail(self, fake_rmtree, fake_tarfile, fake_os):
        with patch.multiple(self.worker,  api=DEFAULT, try_file_remove=DEFAULT, try_dir_remove=DEFAULT) as mocks:
            mocks['try_dir_remove'].return_value = False
            self.assertFalse(self.worker.ide_save())

    @patch('safeworkercode.ide_worker.os')
    def test_get_all_lxc_context_names(self, fake_os):
        fake_os.listdir.return_value = ['1']
        self.assertEqual(self.worker.get_all_lxc_context_names(), fake_os.listdir.return_value)

        fake_os.path.isdir.return_value = False
        self.assertEqual(self.worker.get_all_lxc_context_names(), [])


    @patch('safeworkercode.ide_worker.tarfile')
    def test_restore_environment(self, fake_tarfile):
        with patch.multiple(self.worker,  get_saved_ide=DEFAULT, try_file_remove=DEFAULT) as mocks:
            mocks['get_saved_ide'].return_value = ""
            self.assertFalse(self.worker.restore_environment())
            mocks['get_saved_ide'].return_value = "file"
            self.assertTrue(self.worker.restore_environment())
            self.assertTrue(mocks['try_file_remove'].called)
            mocks['get_saved_ide'].side_effect = Exception()
            self.assertFalse(self.worker.restore_environment())

    def test_remove_containers(self):
        with patch.multiple(self.worker,  find_container=DEFAULT, docker_client=DEFAULT) as mocks:
            self.assertTrue(self.worker.remove_containers())
            mocks['docker_client'].stop.side_effect = Exception()
            self.assertFalse(self.worker.remove_containers())

    def test_get_saved_ide(self):
        with patch.multiple(self.worker,  api=DEFAULT, download_as_tempfile=DEFAULT) as mocks:
            mocks['download_as_tempfile'].return_value = "file"
            mocks['api'].get_ide_url.return_value = None
            self.assertIsNone(self.worker.get_saved_ide())
            mocks['api'].get_ide_url.return_value = "url"
            self.assertEqual(self.worker.get_saved_ide(), mocks['download_as_tempfile'].return_value)
            mocks['api'].get_ide_url.side_effect = ApiError("")
            self.assertIsNone(self.worker.get_saved_ide())

    def test_get_ide_setup_status(self):
        with patch.multiple(self.worker,  api=DEFAULT) as mocks:
            mocks['api'].get_ide_setup_status.return_value = "status"
            self.assertEqual(self.worker.get_ide_setup_status(), mocks['api'].get_ide_setup_status.return_value)

    def test_set_ide_remove_result(self):
        with patch.multiple(self.worker,  api=DEFAULT) as mocks:
            mocks['api'].ide_setup_status.return_value = True
            self.assertTrue(self.worker.set_ide_remove_result("status"))
            mocks['api'].ide_setup_status.assert_called_once_with(self.worker.req_dict['uid'],
                                                                  self.worker.req_dict['pid'],
                                                                  {'command': 'ide_remove', 'status': "status"})

    @patch('safeworkercode.ide_worker.string')
    @patch('__builtin__.open')
    def test_create_file_from_template(self, fake_open, fake_string):
        fake_string.Template.return_value.substitute.return_value = "file_content"
        self.assertEqual(self.worker.create_file_from_template("","",""),
                         fake_string.Template.return_value.substitute.return_value)

    def test_get_lxc_conf(self):
        self.assertTrue(isinstance(self.worker.get_lxc_conf(), list))

    def test_parse_lxc_context_name(self):
        uid = str(ObjectId())
        pid = str(ObjectId())
        context_name = "ide-%s-%s" % (uid, pid)
        self.assertEqual(self.worker.parse_lxc_context_name(context_name), (uid,pid))
        self.assertRaises(ValueError, self.worker.parse_lxc_context_name, "invalid")


    def test_parse_container_name(self):
        uid = str(ObjectId())
        pid = str(ObjectId())
        container_name = "/ide-%s-%s-label" % (uid, pid)
        self.assertEqual(self.worker.parse_container_name(container_name), (uid,pid,"label"))
        container_name = "worker-%s-%s-label" % (uid, pid)
        self.assertRaises(ValueError, self.worker.parse_container_name, container_name)

    @patch('safeworkercode.ide_worker.tempfile')
    @patch('safeworkercode.ide_worker.urllib2')
    def test_download_data_files(self, fake_urllib2, fake_tempfile):
        with patch.multiple(self.worker,  api=DEFAULT) as mocks:
            uid = str(ObjectId())
            pid = str(ObjectId())
            mocks['api'].get_data_url.return_value = "url"
            fake_urllib2.urlopen.return_value.read.return_value = None
            fake_tempfile.NamedTemporaryFile.return_value = fake_namedtempfile()
            self.assertEqual(self.worker.download_data_files(uid, pid, ["filename"]), ["filename"])
            fake_urllib2.urlopen.return_value.read = fake_read().next
            self.assertEqual(self.worker.download_data_files(uid, pid, ["filename"]), ["filename"])

    @patch('safeworkercode.ide_worker.tempfile')
    def test_new_tempfile(self, fake_tempfile):
        fake_tempfile.NamedTemporaryFile.return_value = fake_namedtempfile()
        self.assertEqual(self.worker.new_tempfile(), "filename")

    def test_download_as_tempfile(self):
        with patch.multiple(self.worker, new_tempfile=DEFAULT, download=DEFAULT) as mocks:
            mocks['new_tempfile'].return_value = "filename"
            self.assertEqual(self.worker.download_as_tempfile("url"), mocks['new_tempfile'].return_value)

    @patch('safeworkercode.ide_worker.urllib2')
    @patch('__builtin__.open')
    def test_download(self, fake_open, fake_urllib2):
        fake_urllib2.urlopen.return_value.read.return_value = None
        self.assertIsNone(self.worker.download("url", "local_file"))
        fake_urllib2.urlopen.return_value.read = fake_read().next
        self.assertIsNone(self.worker.download("url", "local_file"))

    @patch('safeworkercode.ide_worker.os.makedirs')
    @patch('safeworkercode.ide_worker.os.path.isdir')
    @patch('safeworkercode.ide_worker.shutil.rmtree')
    def test_create_clean_dir(self, mock_rmtree, mock_isdir, mock_makedirs):
        self.assertIsNone(self.worker.create_clean_dir("path"))

    @patch('safeworkercode.ide_worker.os.remove')
    def test_try_file_remove(self, mock_remove):
        self.assertIsNone(self.worker.try_file_remove("filename"))
        mock_remove.side_effect = OSError()
        self.assertIsNone(self.worker.try_file_remove("filename"))

    @patch('safeworkercode.ide_worker.os.path.isdir')
    @patch('safeworkercode.ide_worker.shutil.rmtree')
    def test_try_dir_remove(self, mock_rmtree, mock_isdir):
        self.assertTrue(self.worker.try_dir_remove("dir"))
        mock_rmtree.side_effect = OSError()
        self.assertFalse(self.worker.try_dir_remove("dir"))

    def test_get_python_url(self):
        with patch.dict(EngConfig, {'LOCAL_IP': '0.0.0.0'}):
            self.assertEqual(self.worker.get_python_url(1), "ipython/0_0_0_1")

    @patch('safeworkercode.ide_worker.socket')
    def test_get_available_port(self, fake_socket):
        fake_socket.socket.return_value.getsockname.return_value = ("addr", "port")
        self.assertEqual(self.worker.get_available_port(), "port")

    @patch('safeworkercode.ide_worker.os.path.join')
    @patch('safeworkercode.ide_worker.shutil.copyfile')
    def test_transfer_files_from_storage(self, mock_copyfile, mock_join):
        with patch.multiple(self.worker, download_data_files=DEFAULT, try_file_remove=DEFAULT) as mocks:
            #project data is required
            self.worker.req_dict['files'] = None
            self.assertFalse(self.worker.transfer_files_from_storage())

            self.worker.req_dict['files'] = ["file"]
            self.assertTrue(self.worker.transfer_files_from_storage())
            mocks['download_data_files'].side_effect = Exception()
            self.assertFalse(self.worker.transfer_files_from_storage())
            self.assertFalse(mocks['try_file_remove'].called)
            mocks['download_data_files'].side_effect = 'foobar.csv'
            mock_copyfile.side_effect = Exception()
            self.assertFalse(self.worker.transfer_files_from_storage())
            self.assertTrue(mocks['try_file_remove'].called)

    def test_get_memory_limit(self):
        with patch.dict(EngConfig, {'IDE_WORKER_MEMORY_LIMIT': 1}):
            self.assertEqual(self.worker.get_memory_limit(), 1)

    @patch('safeworkercode.ide_worker.os.makedirs')
    def test_make_dirs(self, fake_makedirs):
        self.assertIsNone(self.worker.make_dirs())
        fake_makedirs.side_effect = OSError()
        fake_makedirs.side_effect.errno = 17
        self.assertIsNone(self.worker.make_dirs())
        fake_makedirs.side_effect.errno = 1
        self.assertRaises(OSError, self.worker.make_dirs)

    @patch('safeworkercode.ide_worker.shutil.copyfile')
    @patch('safeworkercode.ide_worker.os.path.join')
    def test_create_default_files(self, fake_join, mock_copyfile):
        with patch.multiple(self.worker, create_file_from_template=DEFAULT) as mocks:
            self.assertIsNone(self.worker.create_default_files("template_data"))

    @patch('safeworkercode.ide_worker.os.path.join')
    def test_launch_rstudio(self, fake_join):
        with patch.multiple(self.worker,
                            docker_client=DEFAULT,
                            get_memory_limit=DEFAULT,
                            get_lxc_context_name=DEFAULT,
                            get_lxc_conf=DEFAULT) as mocks:
            mocks['docker_client'].port.return_value = "port"
            self.assertEqual(self.worker.launch_rstudio("port"), mocks['docker_client'].port.return_value)
            self.assertRaises(ValueError, self.worker.launch_rstudio, "different_port")

    @patch('safeworkercode.ide_worker.os.path.join')
    def test_launch_ipython(self, fake_join):
        with patch.multiple(self.worker,
                            docker_client=DEFAULT,
                            get_memory_limit=DEFAULT,
                            get_lxc_context_name=DEFAULT,
                            get_lxc_conf=DEFAULT) as mocks:
            mocks['docker_client'].port.return_value = "port"
            self.assertEqual(self.worker.launch_ipython("port"), mocks['docker_client'].port.return_value)
            self.assertRaises(ValueError, self.worker.launch_ipython, "different_port")

    def test_run(self):
        with patch.multiple(self.worker, accept_job=DEFAULT, pipe=DEFAULT) as mocks:
            with patch.dict(self.worker.WORK_ROUTER, {'command': lambda x: x}):
                self.worker.req_dict['command'] = 'command'
                self.assertEqual(self.worker.run(), self.worker.req_dict)
                mocks['accept_job'].return_value = False
                self.assertIsNone(self.worker.run())

    def test_run_interrupt(self):
        with patch.multiple(self.worker, accept_job=DEFAULT, pipe=DEFAULT) as mocks:
            with patch.dict(self.worker.WORK_ROUTER, {'command': lambda x: x}):
                mocks['accept_job'].side_effect = SystemExit
                self.worker.req_dict['command'] = 'command'
                self.assertIsNone(self.worker.run())

    @patch('safeworkercode.ide_worker.os.geteuid')
    def test_init(self, fake_geteuid):
        fake_geteuid.return_value = 1
        pipe = Mock()
        self.assertRaises(RuntimeError, IDEWorker, 1, {'uid': str(ObjectId()), 'pid': str(ObjectId()), 'key': 'x'}, pipe)

    def test_find_container(self):
        uid = "52668f10637aba7d5104b7ae"
        pid = "26374b768f1065aea7d510ab"
        containers = [{u'Command': u'/bin/bash start_rstudio.sh /bin/bash',
          u'Created': 1382454054,
          u'Id': u'6a885bbbecd1cc0b2ab4dd99fcf2dbd074826df282ff9bd7d49cf52ab10cdb51',
          u'Image': u'rstudio-ide-52668f10637aba7d5104b7ae-26374b768f1065aea7d510ab:latest',
          u'Ports': u'49192->8787',
          u'SizeRootFs': 0,
          u'SizeRw': 0,
          u'Status': u'Up 9 seconds',
          u'Names': [u'/ide-%s-%s-rstudio' % (uid, pid)]}]

        #wrong uid
        with patch.object(self.worker.docker_client, 'containers') as mock_container:
            mock_container.return_value = containers
            container = self.worker.find_container("wrong_uid", pid, "rstudio")
            self.assertIsNone(container)

        #no containers
        with patch.object(self.worker.docker_client, 'containers') as mock_container:
            mock_container.return_value = []
            container = self.worker.find_container(uid, pid, "rstudio")
            self.assertIsNone(container)

        #valid
        with patch.object(self.worker.docker_client, 'containers') as mock_container:
            mock_container.return_value = containers
            container = self.worker.find_container(uid, pid, "rstudio")
            self.assertIsNotNone(container)

    def test_create_credentials(self):
        request = dict.fromkeys(['uid', 'pid', 'dataset_id', 'original_filename'], 'x')
        password = self.worker.create_password()
        credentials = ('x', password)

        self.assertEqual(len(credentials), 2)
        self.assertEqual(request['uid'], credentials[0])
        self.assertEqual(self.worker.PASSWORD_LENGTH, len(credentials[1]))

    def test_get_all_sessions(self):
        with patch.multiple(self.worker,  docker_client=DEFAULT) as mocks:
            mocks['docker_client'].get_container_names.return_value = ['/ide-1-2-label', '/worker-container']
            self.assertEqual(self.worker.get_all_sessions(), set([('1','2')]))

    def test_remove_orphan_sessions(self):
        with patch.multiple(self.worker,
                            get_all_sessions=DEFAULT,
                            api=DEFAULT,
                            remove_containers=DEFAULT,
                            try_dir_remove=DEFAULT) as mocks:
            mocks['get_all_sessions'].return_value = set([('uid', 'pid')])
            self.assertIsNotNone(self.worker.worker_id)
            mocks['api'].get_ide_setup_status.return_value = {}
            self.assertIsNone(self.worker.remove_orphan_sessions())
            self.assertTrue(mocks['remove_containers'].called)
            self.assertTrue(mocks['try_dir_remove'].called)

    def test_remove_orphan_sessions_api_error(self):
        with patch.multiple(self.worker,
                            get_all_sessions=DEFAULT,
                            api=DEFAULT,
                            remove_containers=DEFAULT,
                            try_dir_remove=DEFAULT) as mocks:
            mocks['get_all_sessions'].return_value = set([('uid', 'pid')])
            self.assertIsNotNone(self.worker.worker_id)
            mocks['api'].get_ide_setup_status.side_effect = ApiError("")
            self.assertIsNone(self.worker.remove_orphan_sessions())

    def test_remove_orphan_lxc_contexts(self):
        with patch.multiple(self.worker,
                            get_all_sessions=DEFAULT,
                            get_all_lxc_context_names=DEFAULT,
                            try_dir_remove=DEFAULT) as mocks:
            mocks['get_all_sessions'].return_value = set([('uid', 'pid')])
            mocks['get_all_lxc_context_names'].return_value = ['ide-1-2', 'invalid']
            self.assertIsNone(self.worker.remove_orphan_lxc_contexts())
            self.assertTrue(mocks['try_dir_remove'].called)

    def test_remove_all_sessions(self):
        with patch.multiple(self.worker,
                            get_all_sessions=DEFAULT,
                            api=DEFAULT,
                            remove_containers=DEFAULT,
                            try_dir_remove=DEFAULT) as mocks:
            mocks['get_all_sessions'].return_value = set([('uid', 'pid')])
            self.assertIsNotNone(self.worker.worker_id)
            self.assertIsNone(self.worker.remove_all_sessions())
            self.assertTrue(mocks['remove_containers'].called)
            self.assertTrue(mocks['try_dir_remove'].called)

    @patch('safeworkercode.ide_worker.os')
    def test_remove_all_lxc_contexts(self, fake_os):
        with patch.multiple(self.worker,
                            try_dir_remove=DEFAULT) as mocks:
            fake_os.listdir.return_value = ["dir"]
            fake_os.path.join.return_value = "full_path"
            self.assertIsNone(self.worker.remove_all_lxc_contexts())
            self.assertTrue(mocks['try_dir_remove'].called)

