############################################################################
#
#       unit test for blueprint interpreter
#
#       Author: Tom DeGodoy
#
#       Copyright DataRobot, Inc. 2013
#
###########################################################################

import numpy as np
import sys
import unittest
import os
import logging
from copy import deepcopy
import json
from bson.objectid import ObjectId
import pytest

from mock import patch, Mock, PropertyMock

from config.engine import EngConfig
from config.test_config import db_config
from common.engine.progress import Progress
from MMApp.entities.db_conn import DBConnections
from ModelingMachine.engine.mocks import RequestData, VertexFactory, Executor
from ModelingMachine.engine import blueprint_interpreter
from ModelingMachine.engine import mocks
from ModelingMachine.engine.blueprint_interpreter import BuildData, ReportOutput
from ModelingMachine.engine.worker_request import WorkerRequest, VertexDefinition, BlueprintIterator, BlenderIterator
from ModelingMachine.engine.vertex import Vertex
import ModelingMachine.engine.user_vertex
from ModelingMachine.engine.monitor import FakeMonitor
from tests.ModelingMachine.blueprint_interpreter_test_helper import BlueprintInterpreterTestHelper
from tests.IntegrationTests.storage_test_base import StorageTestBase
import ModelingMachine


mm = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(mm,'../..'))

class FakeRequest(dict):
    def __init__(self,d):
        self.update(d)
        self.input_types='NUM'
    def vertex_inputs(self,x):
        return None


class TestBlueprintInterpreter(StorageTestBase):
    def np_equal(self,a,b):
        """ assert two numpy arrays are equal (even if they have nan's)
        """
        try:
            np.testing.assert_equal(a,b)
        except AssertionError:
            return False
        return True

    @classmethod
    def setUpClass(cls):
        super(TestBlueprintInterpreter, cls).setUpClass()

        cls.Executor = blueprint_interpreter.Executor
        blueprint_interpreter.Executor = mocks.Executor
        cls.dbs = DBConnections()

        cls.bp1 = {}
        cls.bp1['1'] = (['NUM'],['NI'],'T')
        cls.bp1['2'] = (['1'],['GLMB'],'P')

        cls.bp2 = {}
        cls.bp2['1'] = (['NUM'],['NI'],'T')
        cls.bp2['2'] = (['CAT'],['DM'],'T')
        cls.bp2['3'] = (['1','2'],['GLMB'],'P')

        cls.bp3 = {}
        cls.bp3['1'] = (['NUM'],['NI'],'T')
        cls.bp3['2'] = (['1'],['LR1'],'S')
        cls.bp3['3'] = (['2'],['RFC nt=10;ls=5'],'P')

        cls.bp_helper = BlueprintInterpreterTestHelper(
            blueprint_interpreter.BlueprintInterpreter,
            WorkerRequest,
            RequestData,
            VertexFactory
        )

        # Call to method in StorageTestBase
        cls.test_directory, cls.datasets = cls.create_test_files()

    @classmethod
    def tearDownClass(cls):
        super(TestBlueprintInterpreter, cls).tearDownClass()
        cls.dbs.destroy_database()
        blueprint_interpreter.Executor = cls.Executor

    def setUp(self):
        super(TestBlueprintInterpreter, self).setUp()
        self.dbs.destroy_database()

    def tearDown(self):
        pass

    def test_blender_iterator(self):
        req = self.bp_helper.create_blender_request(None, None, self.bp1, self.bp3)
        req = WorkerRequest(req)
        blender = BlenderIterator(req)
        for item in blender:
            self.assertNotEqual(item,req) #make sure not to overwrite req
            self.assertEqual(set(item.keys()),set(req.keys()))

    def test_report_output(self):
        x = {'blueprint_id':1234,'pid':ObjectId(),'test':{'b':1,'c':2},'unlocked_holdout':False}
        r = ReportOutput(x)
        check = {'blueprint_id':1234,'pid':str(x['pid']),'test':{'b':1,'c':2}}
        self.assertEqual(r, check)
        self.assertEqual(json.loads(r.json), check)

    def test_build(self):

        blueprints = [self.bp1, self.bp3]
        request_data = self.bp_helper.get_requestdata(self.test_directory, self.datasets[0])
        request_data.partition = { 'total_size': 50, 'holdout_pct': 20, 'version': 1 }

        result = self.bp_helper.execute_blueprints(blueprints, request_data)

        bi = result['bi']
        bi.vertex_factory.clear_cache()
        print bi
        with self.assertRaises(KeyError):
            req2 = result['requests'][1]
            out = bi._build(req2, subset='holdout')

    def test_fit(self):
        req = WorkerRequest(self.bp_helper.create_request(None, None, self.bp1))
        request_data = self.bp_helper.get_requestdata(self.test_directory, self.datasets[0])
        request_data.partition = { 'total_size': 200, 'holdout_pct': 20, 'version': 1 }
        vertex_factory = VertexFactory()
        bi = blueprint_interpreter.BlueprintInterpreter( vertex_factory, request_data, req )

        # Tests that any empty-initialized data will still end up in mongo, in case
        # a job errors the key will still be there
        self.assertIn('partition_stats', blueprint_interpreter.mongodict(
            bi.report()))

        out = bi.fit()
        self.assertTrue(out)

        uireport = json.loads(bi.report().json)
        self.assertLessEqual(set(uireport.keys()), set(ReportOutput.ui_allowed_keys))

        print vertex_factory
        self.assertEqual(len(vertex_factory),len(req.blueprint))


    def test_user_model(self, *args, **kwargs):
        req = WorkerRequest(self.bp_helper.create_request(None, None, self.bp_helper.user_bp))
        request_data = self.bp_helper.get_requestdata(self.test_directory, self.datasets[2])
        request_data.partition = { 'total_size': 200, 'holdout_pct': 20, 'version': 1 }
        vertex_factory = VertexFactory(request_data.usertasks)
        bi = blueprint_interpreter.BlueprintInterpreter( vertex_factory, request_data, req )
        with patch('ModelingMachine.engine.user_vertex.UserVertex.location', self.test_directory):
            out = bi.fit()
        self.assertTrue(out)
        self.assertTrue(bi.report())

    @patch.object(ModelingMachine.engine.blueprint_interpreter.BuildData,'_getX')
    @patch.object(ModelingMachine.engine.blueprint_interpreter.BuildData,'_create_partition')
    @patch.object(ModelingMachine.engine.blueprint_interpreter.BlueprintInterpreter,'_score')
    @patch.object(ModelingMachine.engine.blueprint_interpreter.BlueprintInterpreter,'_prediction_report')
    @patch.object(ModelingMachine.engine.blueprint_interpreter.BlueprintInterpreter,'_fit_prediction_report')
    def test_vertex_fit_calls(self,*args,**kwargs):
        req = WorkerRequest(self.bp_helper.create_request(None, None, self.bp1))
        req['partitions'] = [(i,-1) for i in range(5)]
        request_data = self.bp_helper.get_requestdata(self.test_directory, self.datasets[0])
        vertex_factory = VertexFactory()
        request_data.partition = { 'total_size': 200, 'holdout_pct': 20, 'reps': 5 }
        bi = blueprint_interpreter.BlueprintInterpreter( vertex_factory, request_data, req )

        with patch('ModelingMachine.engine.vertex.Vertex._fit_and_act') as mock_fit:
            with patch('ModelingMachine.engine.vertex.Vertex._act') as mock_act:
                with patch('ModelingMachine.engine.vertex.Vertex.dirty_parts',set(req.partitions)):
                    #mocking dirty_parts so vertex.save() will create files eventhough fit was mocked
                    bi._build(req)
                    #act is not called
                    self.assertEqual(mock_act.call_count,0)
                    #fit is called
                    self.assertGreater(mock_fit.call_count,0)

        bi = blueprint_interpreter.BlueprintInterpreter( vertex_factory, request_data, req )
        with patch('ModelingMachine.engine.vertex.Vertex._fit_and_act') as mock_fit:
            with patch('ModelingMachine.engine.vertex.Vertex._act') as mock_act:
                bi.fit()
                #act is not called
                self.assertEqual(mock_act.call_count,0)
                #fit is called
                self.assertGreater(mock_fit.call_count,0)

        bi = blueprint_interpreter.BlueprintInterpreter( vertex_factory, request_data, req )
        with patch('ModelingMachine.engine.vertex.Vertex._fit_and_act') as mock_fit:
            with patch('ModelingMachine.engine.vertex.Vertex._act') as mock_act:
                bi._build(req, strict_cache=True)
                #fit is not called
                self.assertEqual(mock_fit.call_count,0)
                #act is called
                self.assertGreater(mock_act.call_count,0)

        print bi

        with patch('ModelingMachine.engine.vertex.Vertex._fit_and_act') as mock_fit:
            with patch('ModelingMachine.engine.vertex.Vertex._act') as mock_act:
                with patch('ModelingMachine.engine.mocks.VertexFactory.clear_cache') as mock_cc:
                    #compute_final_score clears the in RAM cache in order to force loading from storage.
                    #but the mock of vertex factory has no access to storage, only the in RAM cache,
                    #so clear_cache had to be mocked
                    bi.compute_final_score()
                    #clear_cache is called once:
                    self.assertEqual(mock_cc.call_count,1)
                    #fit is not called
                    self.assertEqual(mock_fit.call_count,0)
                    #act is called
                    self.assertGreater(mock_act.call_count,0)

        with patch('ModelingMachine.engine.vertex.Vertex._fit_and_act') as mock_fit:
            with patch('ModelingMachine.engine.vertex.Vertex._act') as mock_act:
                bi.request['command'] = 'predict'
                bi.predict()
                #fit is not called
                self.assertEqual(mock_fit.call_count,0)
                #act is called
                self.assertGreater(mock_act.call_count,0)

    @patch('common.io.query.subset_data_with_na')
    @patch('ModelingMachine.engine.blueprint_interpreter.Container')
    @patch('ModelingMachine.engine.blueprint_interpreter.BlueprintInterpreter.metrics',{})
    def test_score(self,*args,**kwargs):
        req = WorkerRequest(self.bp_helper.create_request(None, None, self.bp1))
        req['partitions'] = [(i,-1) for i in range(5)]
        request_data = self.bp_helper.get_requestdata(self.test_directory, self.datasets[0])
        request_data.partition = { 'total_size': 200, 'holdout_pct': 20 }
        vertex_factory = VertexFactory()

        self.assertEqual(req.partitions,[(i,-1) for i in range(5)])
        expected = req.copy()
        expected['partitions'] = [(0,-1)]

        with patch('ModelingMachine.engine.blueprint_interpreter.BlueprintInterpreter._build') as mock_build:
            with patch('ModelingMachine.engine.blueprint_interpreter.BuildData') as mock_build_data:
                bi = blueprint_interpreter.BlueprintInterpreter( vertex_factory, request_data, req )
                bi._score('holdout')
                bi._build.assert_called_with(expected,subset='holdout',strict_cache=True)
                self.assertEqual(mock_build_data.call_count, 1)

        expected['partitions'] = [(-1,-1)]
        expected['samplepct'] = 80
        with patch('ModelingMachine.engine.blueprint_interpreter.BlueprintInterpreter._build') as mock_build:
            with patch('ModelingMachine.engine.blueprint_interpreter.BuildData') as mock_build_data:
                bi = blueprint_interpreter.BlueprintInterpreter( vertex_factory, request_data, expected )
                bi._score('holdout')
                bi._build.assert_called_with(expected,subset='holdout',strict_cache=True)
                self.assertEqual(mock_build_data.call_count, 1)

        expected['partitions'] = [(-1,-1)]
        expected['samplepct'] = 100
        with patch('ModelingMachine.engine.blueprint_interpreter.BlueprintInterpreter._build') as mock_build:
            with patch('ModelingMachine.engine.blueprint_interpreter.BuildData') as mock_build_data:
                bi = blueprint_interpreter.BlueprintInterpreter( vertex_factory, request_data, expected )
                bi._fit_data = Mock()
                bi._score('holdout')
                bi._build.assert_called_with(expected,subset='holdout',strict_cache=True)
                self.assertEqual(mock_build_data.call_count, 3) #called more times to get stacked predictions

    def test_output_cache(self):
        blueprints = [self.bp1, self.bp3]
        request_data = self.bp_helper.get_requestdata(self.test_directory, self.datasets[0])
        request_data.partition = { 'total_size': 200, 'holdout_pct': 20 }

        bp1 = blueprints[0]
        bp2 = blueprints[1]
        pid = None
        dataset_id = None

        # build a simple blueprint
        req = self.bp_helper.WorkerRequest(self.bp_helper.create_request(pid, dataset_id, bp1))
        vertex_factory = self.bp_helper.VertexFactory()
        bi = self.bp_helper.BlueprintInterpreter( vertex_factory, request_data )
        out = bi._build(req)
        # 2 vertices in cache
        self.assertEqual(len(bi.output_cache.cache.keys()),2)

        # build the same blueprint
        req = self.bp_helper.WorkerRequest(self.bp_helper.create_request(pid, dataset_id, bp1))
        out = bi._build(req)
        # still 2 vertices in cache
        self.assertEqual(len(bi.output_cache.cache.keys()),2)

        # build the same blueprint with stack method
        bp1a = deepcopy(bp1)
        bp1a['2'] = bp1a['2'][0:2] + ('S',)
        req = self.bp_helper.WorkerRequest(self.bp_helper.create_request(pid, dataset_id, bp1a))
        out = bi._build(req)
        # should add 1 vertex to cache
        self.assertEqual(len(bi.output_cache.cache.keys()),3)

    def test_update_grid(self):
        # update task with no args
        request = {
            'blueprint': {'1': (['NUM'],['NI'],'T'), '2': (['1'],['GLM'],'P') },
            'new_grid': {'2': {'method': 'Dumb', 'grid': {'p1': '1', 'p2': '3,4' } } },
        }
        vd = VertexDefinition(FakeRequest(request), '2')
        vd.tasks = ['GLM']
        task_list = vd.get_new_grid()
        self.assertEqual(task_list,(['GLM p1=1;p2=[3,4];t_a=0']))

        # update task with args
        request = {
            'blueprint': {'1': (['NUM'],['NI'],'T'), '2': (['1'],['GLM p1=[3,4];p2=5'],'P') },
            'new_grid': {'2': {'method': 'Dumb', 'grid': {'p1': '1', 'p2': '3,4' } } },
        }
        vd = VertexDefinition(FakeRequest(request), '2')
        vd.tasks = ['GLM p1=[3,4];p2=5']
        task_list = vd.get_new_grid()
        self.assertEqual(task_list,(['GLM p1=[3,4,1];p2=[5,3,4];t_a=0']))

class TestReporting(unittest.TestCase):

    def test_base_report_has_test_key_even_if_no_validation(self):
        request = {u'blueprint': {u'1': [[u'NUM'], [u'NI'], u'T'], u'2': [[u'1'], [u'RULEFITC n=200;ml=5;s=auto;a=auto;v=0;mr=200;t_m=LogLoss'], u'P']}, u'lid': u'53c4aae08bd88f293a08e4cb', u'samplepct': 65, u'uid': u'5359d6cb8bd88f5cddefd3a8', u'blueprint_id': u'e811db9d5a2efb2e8f1a4802228f7871', u'total_size': 1, u'qid': 7, u'icons': [0], u'pid': u'53c4a15b8bd88f47ec115ad0', u'runs': 1, u'max_reps': 1, u's': 0, u'command': u'fit', u'features': [u'Missing Values Imputed', u'RuleFit Classifier'], u'model_type': u'RuleFit Classifier', u'bp': 13, u'max_folds': 0, u'new_lid': True, u'dataset_id': u'53c4a15d8bd88f1e48fc6839', u'partitions': [[-1, -1]]}
        worker_request = WorkerRequest(request)
        mock_request_data = RequestData(target_name='target',
                                        target_type='Regression',
                                        target_vector={'main': range(10),
                                                       'holdout': range(10)})
        vertices = VertexFactory(mock_request_data.usertasks)
        progress_inst = Progress.sink()

        model = blueprint_interpreter.BlueprintInterpreter(
            vertices, mock_request_data, worker_request, progress_inst,
            FakeMonitor())

        base_report = model.report()
        self.assertIn('test', base_report)
        for metric_name in base_report['test']['metrics']:
            self.assertIsNone(base_report['test'][metric_name][0])


if __name__ == '__main__':
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logger.addHandler(console)

    unittest.main()
