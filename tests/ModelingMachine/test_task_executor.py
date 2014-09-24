import unittest
import os
import mock

from config.engine import EngConfig
from ModelingMachine.engine.task_executor import TaskExecutor
from common.exceptions import UserTaskError
from ModelingMachine.engine.monitor import FakeMonitor

class TestTaskExecutor(unittest.TestCase):

    def setUp(self):
        self.mock_vertex = mock.Mock()
        self.mock_vertex_type = 'UserVertex'
        self.mock_vertex.__class__.__name__ = self.mock_vertex_type

        self.patchers = []
        config_patch = mock.patch.dict(EngConfig, {}, clear = False)
        config_patch.start()
        monitor_patch = mock.patch('ModelingMachine.engine.task_executor.Monitor', FakeMonitor)
        monitor_patch.start()
        self.patchers.append(config_patch)
        self.patchers.append(monitor_patch)
        self.addCleanup(self.stopPatching)

    def stopPatching(self):
        for patcher in self.patchers:
            if patcher:
                patcher.stop()

    def test_get_lxc_context_name(self):
        executor = TaskExecutor({'task_map':'whatever'})
        executor.pid = 'pid'
        executor.qid = 'qid'

        lxc_name = executor.get_lxc_context_name()
        self.assertIn(executor.pid, lxc_name)
        self.assertIn(executor.qid, lxc_name)

    # FIXME: Identical function in IntegrationTestBase, create test utility class or common base
    def path_to_test_file(self, file_name):
        tests = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(tests, 'testdata', file_name)

    def get_lxc_error_output_example(self):
        file_path = self.path_to_test_file('lxc-error-output-example')
        with open(file_path, 'r') as f:
            return f.read()

    def test_get_user_traceback(self):
        executor = TaskExecutor({'task_map':'whatever'})
        lxc_error_logs = self.get_lxc_error_output_example()
        logs = executor.get_user_traceback(lxc_error_logs)
        datarobot_code_keywords = ['ModelingMachine', 'user_vertex.py', 'usermodule.py']
        for k in datarobot_code_keywords:
            self.assertNotIn(k, logs, '{} was found in the logs, get_user_traceback '.format(k) +
                'should only show the user model traceback')

    @mock.patch('ModelingMachine.engine.task_executor.os', autospec = True)
    def test_user_error_exception(self, MockOs):
        executor = TaskExecutor({'task_map':'whatever'})
        with mock.patch.object(executor, 'docker_client', return_value = mock.DEFAULT) as mock_docker:
            mock_docker.read_lxc_log.return_value = self.get_lxc_error_output_example()

            self.assertRaises(UserTaskError, executor.launch_lxc_and_execute,
                'fake-lxc-volume-path')
