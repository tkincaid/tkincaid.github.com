import unittest
import os
import urllib2

from mock import Mock, patch, call, DEFAULT, mock_open

from safeworkercode.user_worker import UserWorker
from safeworkercode.user_config import EngConfig
from tests.safeworkercode.test_usermodel import UserModelTestBase

class TestUserWorker(UserModelTestBase):

    @classmethod
    def setUpClass(self):
        self.zmq_patcher = patch('safeworkercode.user_worker.zmq')
        self.zmq_patcher.start()

    @classmethod
    def tearDownClass(self):
        self.zmq_patcher.stop()

    def setUp(self):
        pipe = Mock()
        self.worker = UserWorker({}, pipe)
        self.container = {u'Command': u'/bin/bash start_rstudio.sh /bin/bash',
             u'Created': 1382531566,
             u'Id': u'5eee68572c4cc1d8147455e0be62986f442921cfb12be91da798cd0766ef954f',
             u'Image': u'rstudio-ide-5267ba65637aba15b13f35d5:latest',
             u'Ports': u'49202->8787',
             u'SizeRootFs': 0,
             u'SizeRw': 0,
             u'Status': u'Up 12 seconds'}

    def test_fit_valid_request(self):

        # Arrange
        request = self.generate_one_request()
        local_file_name = 'local_file_name'
        with patch.multiple(self.worker, download_file=DEFAULT,
                            get_dataframes=DEFAULT,
                            isolate_and_execute_user_model=DEFAULT,
                            api=DEFAULT, wait_and_clean=DEFAULT,
                            pickle_data=DEFAULT,
                            update_metrics=DEFAULT) as mocks:

               report = {}
               report.update(request)
               report['new_key'] = 'new_value'
               req_dict = {'job':'value'}
               self.worker.req_dict = req_dict
               class temp:
                    samplesize = 999
               self.worker.parts = temp()

               mocks['download_file'].return_value = local_file_name
               mocks['get_dataframes'].return_value = ['ATrainDataFrame', 'ATestDataFrame']
               mocks['pickle_data'].side_effect = ['TrainFilePath', 'TestFilePath']

               # Act
               r = self.worker.fit(request)
               # Assert
               mocks['api'].report_started.assert_called_once_with({
                'pid': request['pid'], 'qid': request['qid']})
               mocks['download_file'].assert_called_once_with('FAKE')
               mocks['isolate_and_execute_user_model'].assert_called_once_with(request,
                       'TrainFilePath', 'TestFilePath')

               #The worker notified the manager through the pipe what the worker was working on
               #self.worker.pipe.send.assert_has_calls([call(req_dict)])

               self.assertItemsEqual(r, {'message':'OK'})

    def test_fit_report_error(self):
      request = self.generate_one_request()
      dataset_id = '009e6c39-046c-42f1-ad5e-c174faac9634'
      request['project'] = {'default_dataset_id' : dataset_id,
                            'target': {
                                'name': 'y',
                                'type': 'Binary',
                            },
                            "metadata": {'files': [dataset_id]},
                            'holdout_pct': 20}

      with patch.multiple(self.worker,
                          download_file = DEFAULT,
                          isolate_and_execute_user_model = DEFAULT,
                          get_dataframes=DEFAULT) as mocks:
            with patch.object(self.worker, 'api') as mock_api:
                mocks['isolate_and_execute_user_model'].side_effect = Exception('Boom!')
                mocks['get_dataframes'].return_value = ['TrainDF', 'TestDF']

                class temp:
                     samplesize = 999
                self.worker.parts = temp()
                r = self.worker.fit(request)

                mock_api.report_error.assert_called_once_with({
                  'qid': request['qid'], 'pid': request['pid'], 'error': 'Boom!'})

    def test_fit_accepts_valid_py_request(self):
        request = self.generate_one_request()
        dataset_id = '009e6c39-046c-42f1-ad5e-c174faac9634'
        local_file_name = 'local_file_name'
        request['project']['default_dataset_id'] = dataset_id
        request['project']['metadata'] =  {'files': [dataset_id]}
        with patch.multiple(self.worker, download_file = DEFAULT,
                isolate_and_execute_user_model = DEFAULT,
                get_dataframes=DEFAULT, wait_and_clean=DEFAULT,
                pickle_data=DEFAULT) as mocks:
              with patch.object(self.worker, 'api') as mock_api:
                mocks['download_file'].return_value = local_file_name
                mocks['get_dataframes'].return_value = ['TrainDF', 'TestDF']
                mocks['pickle_data'].side_effect = ['/train/path', '/test/path']

                class temp:
                     samplesize = 999
                self.worker.parts = temp()
                r = self.worker.fit(request)
                mocks['isolate_and_execute_user_model'].assert_called_once_with(
                        request,'/train/path', '/test/path')

    def test_download_file(self):
        fake_open = mock_open()
        with patch.object(urllib2, 'urlopen', fake_open):
            with patch.object(self.worker, 'api') as mock_api:
                dataset_id = '0a20e1e3-13c2-4e81-a671-f30f5fa15d95'
                url = 'http://domain.com/%s' % dataset_id
                mock_api.get_data_url.return_value = url

                tests = os.path.dirname(os.path.abspath(__file__))
                filename = os.path.join(tests, 'testdata', dataset_id)

                r = self.worker.download_file(dataset_id)
                self.assertIsNotNone(r)
                mock_api.get_data_url.assert_called_once_with(dataset_id)
                fake_open.assert_called_once_with(url)

    def test_fit_invalid_request(self):
        r = self.worker.fit({})
        self.assertIsNone(r)

        # Missing project
        data = dict.fromkeys(['qid', 'pid', 'modelfit', 'modelpredict', 'lid'], 'x')
        r = self.worker.fit(data)
        self.assertIsNone(r)

    def test_isolate_and_execute_user_model(self):
        # ARRANGE
        request = self.generate_one_request()
        uid = '12345'
        request['uid'] = uid
        data_file_name = 'data-file-name.txt'
        data_file_path = os.path.join('/path/to/worskpace/in/container/', data_file_name)

        config_mock = Mock()
        config_mock.return_value = {}
        with patch.multiple(self.worker, create_user_model_config = config_mock,
            create_file_from_template = DEFAULT,
            launch_lxc = DEFAULT,
            create_lxc_context = DEFAULT,
            copy_data_to_lxc = DEFAULT,
            copy_application_files = DEFAULT,
            publish_resources = DEFAULT) as mocks:

            context_name = self.worker.get_lxc_context_name(request)
            context_host_dir = 'path/to/context/in/host'
            lxc_context_location =  os.path.join(context_host_dir, context_name)
            mocks['create_lxc_context'].return_value = lxc_context_location
            mocks['copy_data_to_lxc'].return_value = data_file_name
            mocks['launch_lxc'].return_value = self.container
            file_contents = 'file contents'
            fake_open = mock_open(read_data = file_contents )

            with patch('safeworkercode.user_worker.open', fake_open, create=True):
                # ACT

                self.worker.isolate_and_execute_user_model(request, data_file_path, data_file_path)
                # ASSERT
                config_mock.assert_called_once_with(request, data_file_name, data_file_name)
                data={
                  'workspace': EngConfig['ISOLATED_WORKER_WORKSPACE'],
                  'isolated_user' : EngConfig['ISOLATED_WORKER_USERNAME'],
                  'repository': EngConfig['LXC_REPOSITORY']
                }
                build_file_path = os.path.join(lxc_context_location, 'Dockerfile')

                mocks['create_file_from_template'].assert_called_once_with(EngConfig['LXC_USER_WORKER_FILE_TEMPLATE'], data, build_file_path)
                mocks['copy_application_files'].assert_called_once_with(lxc_context_location)

                mocks['create_lxc_context'].assert_called_once_with(context_name)


                mocks['launch_lxc'].assert_called_once_with(tag = context_name,
                    build_file_path = lxc_context_location)

                mocks['publish_resources'].assert_called_once_with(request, self.container, 1)

                config_path = os.path.join(lxc_context_location, EngConfig['ISOLATED_WORKER_CONFIG_NAME'])
                template_file_name = EngConfig['LXC_USER_WORKER_FILE_TEMPLATE']


                fake_open.assert_has_calls([
                    call(config_path, 'w')
                ], any_order = True)

    def test_create_user_model_config(self):
        request = dict.fromkeys(['qid', 'pid', 'modelfit', 'modelpredict', 'lid'], 'x')
        train_file_name = '/tmp/train'
        test_file_name = '/tmp/test'

        user_model_config = self.worker.create_user_model_config(request,
                train_file_name, test_file_name)

        keys = ['data_dir', 'web_api_location', 'request', 'train_file_name',
                'test_file_name']
        self.assertTrue(all(k in user_model_config for k in keys))

    def test_create_file_from_template(self):

        template = "RUN mkdir -p $data_dir"

        fake_open = mock_open(read_data = template)

        template_file_path = '/path/to/workspace/in/container/'
        some_var = 'value'
        data = {'data_dir' : some_var}
        destination_path = '/new/path/file_name.ext'

        with patch('safeworkercode.safe_worker_base.open', fake_open, create=True):
            file_content = self.worker.create_file_from_template(template_file_path, data, destination_path)

        fake_open.assert_has_calls([
                    call(template_file_path),
                    call(destination_path, 'w')
                ], any_order = True)

        self.assertEqual(file_content, template.replace('$data_dir', some_var))

    def test_publish_resources(self):
        request = dict.fromkeys(['qid', 'pid', 'modelfit', 'modelpredict', 'lid'], 'x')
        process_id = 10285
        container_info = {u'Args': [u'start_rstudio.sh', u'/bin/bash'],
             u'Config': {u'AttachStderr': True,
              u'AttachStdin': False,
              u'AttachStdout': True,
              u'Cmd': [u'/bin/bash'],
              u'CpuShares': 0,
              u'Dns': None,
              u'Domainname': u'',
              u'Entrypoint': [u'/bin/bash', u'start_rstudio.sh'],
              u'Env': [u'HOME=/',
               u'PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'],
              u'Hostname': u'5eee68572c4c',
              u'Image': u'rstudio-ide-5267ba65637aba15b13f35d5',
              u'Memory': 0,
              u'MemorySwap': 0,
              u'NetworkDisabled': False,
              u'OpenStdin': False,
              u'PortSpecs': [u'8787'],
              u'Privileged': False,
              u'StdinOnce': False,
              u'Tty': False,
              u'User': u'',
              u'Volumes': None,
              u'VolumesFrom': u'',
              u'WorkingDir': u''},
             u'Created': u'2013-10-23T08:32:46.010611735-04:00',
             u'HostnamePath': u'/var/lib/docker/containers/5eee68572c4cc1d8147455e0be62986f442921cfb12be91da798cd0766ef954f/hostname',
             u'HostsPath': u'/var/lib/docker/containers/5eee68572c4cc1d8147455e0be62986f442921cfb12be91da798cd0766ef954f/hosts',
             u'ID': u'5eee68572c4cc1d8147455e0be62986f442921cfb12be91da798cd0766ef954f',
             u'Image': u'9299863e63cef3eabae14daf3adb3fec7f6b5c3f4aaf668d21c6ae9c740e8f2f',
             u'NetworkSettings': {u'Bridge': u'docker0',
              u'Gateway': u'172.17.42.1',
              u'IPAddress': u'172.17.0.51',
              u'IPPrefixLen': 16,
              u'PortMapping': {u'Tcp': {u'8787': u'49202'}, u'Udp': {}}},
             u'Path': u'/bin/bash',
             u'ResolvConfPath': u'/var/lib/docker/containers/5eee68572c4cc1d8147455e0be62986f442921cfb12be91da798cd0766ef954f/resolv.conf',
             u'State': {u'ExitCode': 0,
              u'FinishedAt': u'0001-01-01T00:00:00Z',
              u'Ghost': False,
              u'Pid': process_id,
              u'Running': True,
              u'StartedAt': u'2013-10-23T08:32:46.055974577-04:00'},
             u'SysInitPath': u'/usr/bin/docker',
             u'Volumes': {},
             u'VolumesRW': {}}

        with patch.object(self.worker, 'resource_usage_socket') as mock_usage_socket:
            with patch.object(self.worker, 'docker_client') as mock_docker_client:
                mock_docker_client.inspect_container.return_value = container_info
                flag = 0

                self.worker.publish_resources(request, self.container, flag)

                mock_docker_client.inspect_container.assert_called_once_with(self.container)
                mock_usage_socket.send_multipart.assert_called_once_with([request['pid'], request['qid'], str(process_id), str(flag)])

    def test_py_execution(self):
        request = dict.fromkeys(['qid', 'pid', 'modelsource', 'lid'], 'x')
        context_name = 'PID-QID'
        lxc_context = '/home/datarobot/{}'.format(context_name)
        data = {'use':'this', 'for':'that', 'in':'substitution'}

        with patch.multiple(self.worker, create_file_from_template = DEFAULT,
                launch_lxc = DEFAULT,
                copy_application_files = DEFAULT,
                publish_resources = DEFAULT,
                write_user_module = DEFAULT) as worker_mocks:

            # make tempfile here

            self.worker.execute_and_return_py_container(request, context_name,
                    lxc_context, data)

            worker_mocks['launch_lxc'].calledWith(tag=context_name,
                    build_file_path = lxc_context)


if __name__ == '__main__':
    unittest.main()
