import unittest
import os

import pandas
import cPickle
from numpy.random import randn

from ModelingMachine.engine.user_vertex import UserVertex
from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.blueprint_interpreter import OutputData
from ModelingMachine.secure_worker_lxc.workspace import run_user_task
from ModelingMachine.engine.task_executor import TaskExecutor
from ModelingMachine.engine.container import Container
from mock import patch

class TestRunUserTask(unittest.TestCase):
    """
    Tests the same script that runs within the docker container with a direct call
    """

    @classmethod
    def setUpClass(self):
        task_map = {
            '12345p': {
                'modeltype': 'Python',
                'classname': 'CustomModel',
                'modelsource': '''
import numpy as np
class CustomModel(object):
    def fit(self, X, Y):
        return self
    def predict(self, X):
        return np.ones(len(X))
                '''
            }
        }

        self.python_vertex = { 'task_list': ['USERTASK id=12345p'], 'task_map': task_map, 'id':'a11b31','stored_files':{} }
        self.p_xdata = pandas.DataFrame({'x1':randn(350), 'x2':randn(350)})
        self.p_ydata = pandas.DataFrame({'y':randn(350)+2000})

        self.Z = Partition(350, total_size=350)
        self.Z.set(max_folds=0, max_reps=5)

        self.data = OutputData('_fit_and_act',
            {'X':self.p_xdata,'Y':self.p_ydata,'Z':self.Z, 'method':'predict'},
            {'vertex_index':1,'pid':'123','qid':'456'})

        self.input_dir = '/tmp/input'
        self.output_dir = '/tmp/output'

    @classmethod
    def tearDownClass(self):
        if os.path.isfile('usermodule.py'):
            os.unlink('usermodule.py')

    def test_run_user_task_directly(self):

        run_user_task.save_to_file(
            self.python_vertex,
            os.path.join(self.input_dir, run_user_task.VERTEX_FILE_NAME)
        )

        with open(os.path.join(self.input_dir, run_user_task.DATA_FILE_NAME),'wb') as out_file:
            cPickle.dump(self.data, out_file, protocol=cPickle.HIGHEST_PROTOCOL)

        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        with patch('ModelingMachine.engine.user_vertex.UserVertex.location', self.input_dir):
            run_user_task.run(self.input_dir, self.output_dir)

        #result = run_user_task.read_pickled_object(
        #    os.path.join(self.output_dir, run_user_task.DATA_FILE_NAME)
        #)
        result = Container()
        result.load(os.path.join(self.output_dir, run_user_task.DATA_FILE_NAME))

        python_vertex = run_user_task.read_json_object(
            os.path.join(self.output_dir, run_user_task.VERTEX_FILE_NAME)
        )

        self.verify_test_output(result, python_vertex)

    def verify_test_output(self, result, vertex):
        for p in result:
            self.assertEqual(sum(result(**p)),350)
        self.assertIsNotNone(vertex)


    def test_run_user_task_inside_container(self):

        executor = TaskExecutor(self.python_vertex)
        task_execution_output = executor.run(self.data)

        self.verify_test_output(
            task_execution_output.data,
            task_execution_output.vertex_output
        )
