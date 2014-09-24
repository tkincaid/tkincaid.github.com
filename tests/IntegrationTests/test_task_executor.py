import os
import cPickle
import json
import shutil
import numpy as np
from scipy.io import savemat,loadmat
import scipy.sparse as sp
import tempfile


from mock import patch, Mock, DEFAULT

from config.engine import EngConfig
from ModelingMachine.engine.task_executor import TaskExecutor
from ModelingMachine.engine.mocks import RequestData, VertexFactory, Executor
from ModelingMachine.engine.blueprint_interpreter import BlueprintInterpreter
from ModelingMachine.engine.worker_request import WorkerRequest

from tests.IntegrationTests.storage_test_base import StorageTestBase
from tests.ModelingMachine.blueprint_interpreter_test_helper import BlueprintInterpreterTestHelper

class TestTaskExecutor(StorageTestBase):

    @classmethod
    def setUpClass(self):
        super(TestTaskExecutor, self).setUpClass()
        # Call to method in StorageTestBase
        self.test_directory, self.datasets = self.create_test_files()

        self.lxc_context_name = 'sw-pid-uid'
        self.lxc_shared_volume = os.path.join(self.test_directory, self.lxc_context_name)

    @classmethod
    def tearDownClass(self):
        super(TestTaskExecutor, self).tearDownClass()

    def setUp(self):
        super(TestTaskExecutor, self).setUp()

        self.mock_vertex = {'task_list':['USERTASK id=asdf'], 'id':'asdf', 'stored_files':{}, 'task_map':{'asdf':None}}

    def test_create_lxc_context(self):
        executor = TaskExecutor(self.mock_vertex)

        if os.path.isdir(self.lxc_shared_volume):
            shutil.rmtree(self.lxc_shared_volume)

        self.assertFalse(os.path.isdir(self.lxc_shared_volume))

        actual_location = executor.create_lxc_context(self.lxc_shared_volume)

        self.assertEqual(self.lxc_shared_volume, actual_location)
        self.assertTrue(os.path.isdir(self.lxc_shared_volume))

    def test_save_lxc_input(self):
        executor = TaskExecutor(self.mock_vertex)

        executor.data = {'key':'data'}
        executor.vertex = {'key':'vertex'}

        def assert_files_exist(exist):
            for filename in ['data', 'vertex']:
                expected_file = os.path.join(self.lxc_shared_volume, EngConfig['SECURE_WORKER_DATA_INPUT'], filename)
                file_exists = os.path.isfile(expected_file)
                self.assertEqual(file_exists, exist)


        assert_files_exist(False)
        executor.save_lxc_input(executor.data, self.lxc_shared_volume)
        assert_files_exist(True)

    def test_launch_lxc_and_execute_can_write_to_and_read_from_volume(self):
        executor = TaskExecutor(self.mock_vertex)
        #FIXME: don't pull an image from the public docker registry inside a test
        executor.docker_client.pull('busybox')
        mount_dest = '/task_workspace'
        test_filename = 'shared-file'
        dest_test_file_path = os.path.join(
            mount_dest,
            EngConfig['SECURE_WORKER_DATA_INPUT'],
            test_filename
        )
        # Create the mount origin here. Otherwise docker will create it with root
        mount_origin = os.path.join(self.lxc_shared_volume, EngConfig['SECURE_WORKER_DATA_INPUT'])
        os.makedirs(mount_origin)
        origin_test_filepath = os.path.join(mount_origin, test_filename)


        command = ['cp', dest_test_file_path, dest_test_file_path + '-copy']

        with patch.multiple(executor, get_command = DEFAULT, get_repository = DEFAULT) as mocks:
            mocks['get_command'].return_value = command
            mocks['get_repository'].return_value = 'busybox'

            with open(origin_test_filepath, 'w'):
                container = executor.launch_lxc_and_execute(self.lxc_shared_volume)
                exitcode = executor.docker_client.wait(container)
                self.assertEqual(exitcode, 0)

            os.unlink(origin_test_filepath)

            copy = origin_test_filepath + '-copy'
            output_was_found = os.path.isfile(copy)
            self.assertTrue(output_was_found)

            os.unlink(copy)

    def test_collect_output_data(self):
        output_dir = os.path.join(self.lxc_shared_volume, EngConfig['SECURE_WORKER_DATA_OUTPUT'])
        os.makedirs(output_dir)

        vertex_file = os.path.join(output_dir, 'vertex')
        tf = tempfile.NamedTemporaryFile(mode='w',delete=False,dir=output_dir)
        tf.close()
        vertex = {(0,-1):tf.name}
        data_file = os.path.join(output_dir, 'data')
        data = {'type':'data'}
        stack_file = os.path.join(output_dir, 'stack')
        stack = {'type':'data'}
        log_file = os.path.join(output_dir, 'logs')
        log = 'test\n'
        report_file = os.path.join(output_dir, 'report')
        report = {'0,-1':'report test'}
        info_file = os.path.join(output_dir, 'info')
        info = {'test':'info test'}

        with open(vertex_file,'w') as file_out:
            json.dump({'0,-1':tf.name},file_out)
        with open(data_file,'w') as file_out:
            cPickle.dump(data, file_out, protocol=cPickle.HIGHEST_PROTOCOL)
        with open(stack_file,'w') as file_out:
            cPickle.dump(stack, file_out, protocol=cPickle.HIGHEST_PROTOCOL)
        with open(log_file,'w') as file_out:
            file_out.write(log)
        with open(report_file,'w') as file_out:
            json.dump(report,file_out)
        with open(info_file,'w') as file_out:
            json.dump(info,file_out)
        np.savez(data_file+'.npz',
            colnames_sa=np.array([1,2,3]),
            colnames_ss=np.array([1,2,3]),
            coltypes_sa=np.array([1,2,3]),
            coltypes_ss=np.array([1,2,3]))
        savemat(data_file+'.mat',{'x':sp.csr_matrix([1,2,3])})
        np.savez(stack_file+'.npz',
            colnames_sa=np.array([1,2,3]),
            colnames_ss=np.array([1,2,3]),
            coltypes_sa=np.array([1,2,3]),
            coltypes_ss=np.array([1,2,3]))
        savemat(stack_file+'.mat',{'x':sp.csr_matrix([1,2,3])})

        executor = TaskExecutor(self.mock_vertex)
        result = executor.collect_output_data(self.lxc_shared_volume)

        self.assertEqual(result.colnames(), [1,2,3,1,2,3])
        self.assertEqual(executor.task_report.values(), report.values())
        self.assertEqual(executor.task_info, info)


    def test_datarobot_task_execution(self):
        """
            Executes datarobot task directly.
            In the near feature we will use containers for DataRobot tasks too

            Refer to TestRunUserTask for tests that execute user tasks inside containers
        """

        self.bp_helper = BlueprintInterpreterTestHelper(
            BlueprintInterpreter,
            WorkerRequest,
            RequestData,
            VertexFactory
        )

        bp1 = {}
        bp1['1'] = (['NUM'],['NI'],'T')
        bp1['2'] = (['1'],['GLMB'],'P')

        bp3 = {}
        bp3['1'] = (['NUM'],['NI'],'T')
        bp3['2'] = (['1'],['RFC nt=10;ls=5'],'P')


        blueprints = [bp1, bp3]
        request_data = self.bp_helper.get_requestdata(self.test_directory, self.datasets[0])
        request_data.partition['total_size'] = 200

        with patch('ModelingMachine.engine.blueprint_interpreter.Executor', Executor):
            result_mock = self.bp_helper.execute_blueprints(blueprints, request_data)

        with patch('ModelingMachine.engine.blueprint_interpreter.Executor', TaskExecutor):
            result_executor = self.bp_helper.execute_blueprints(blueprints, request_data)

        self.assertItemsEqual(result_mock['results'], result_executor['results'])

    def test_json_save_load(self):
        obj = { 'files': { (0,-1): 'filename', (1,-1): 'filename' }, (2,-1): 'test' }
        tf = tempfile.NamedTemporaryFile(mode='w',delete=True)
        executor = TaskExecutor(self.mock_vertex)
        executor.save_to_json(obj,tf.name)
        obj2 = executor.read_json_object(tf.name)
        self.assertEqual(obj,obj2)

