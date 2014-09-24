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
import time
from bson.objectid import ObjectId

from mock import patch

from config.engine import EngConfig
from config.test_config import db_config
from common.wrappers import database

from ModelingMachine.engine.mocks import VertexFactory
import ModelingMachine.engine.vertex
import ModelingMachine.engine.user_vertex
from ModelingMachine.engine import blueprint_interpreter
from ModelingMachine.engine.monitor import FakeMonitor
from ModelingMachine.engine import mocks
from ModelingMachine.engine.workspace import Workspace
from ModelingMachine.engine.worker_request import WorkerRequest
from ModelingMachine.engine.data_processor import RequestData
from ModelingMachine.engine.pandas_data_utils import varTypeString
from tests.ModelingMachine.blueprint_interpreter_test_helper import BlueprintInterpreterTestHelper
from tests.IntegrationTests.storage_test_base import StorageTestBase

mm = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(mm,'../..'))

class TestRequestData(StorageTestBase):
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
        super(TestRequestData, cls).setUpClass()
        cls.vertex_monitor_patch = patch('ModelingMachine.engine.vertex.Monitor', ModelingMachine.engine.monitor.FakeMonitor)
        cls.vertex_monitor_mock = cls.vertex_monitor_patch.start()

        cls.Executor = blueprint_interpreter.Executor
        blueprint_interpreter.Executor = mocks.Executor
        cls.persistent = database.new('persistent')
        cls.tempstore = database.new('tempstore')

        cls.bp1 = {}
        cls.bp1['1'] = (['NUM'],['NI'],'T')
        cls.bp1['2'] = (['1'],['GLMB'],'P')

        cls.bp2 = {}
        cls.bp2['1'] = (['NUM'],['NI'],'T')
        cls.bp2['2'] = (['CAT'],['DM'],'T')
        cls.bp2['3'] = (['1','2'],['GLMB'],'P')

        cls.bp3 = {}
        cls.bp3['1'] = (['NUM'],['NI'],'T')
        cls.bp3['2'] = (['1'],['RFC nt=10;ls=5'],'P')

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
        super(TestRequestData, cls).tearDownClass()
        cls.vertex_monitor_patch.stop()
        cls.tempstore.conn.flushdb()
        cls.persistent.db_connection.drop_database(db_config["persistent"]["dbname"])
        blueprint_interpreter.Executor = cls.Executor

    def setUp(self):
        super(TestRequestData, self).setUp()
        self.tempstore.conn.flushdb()
        self.persistent.db_connection.drop_database(db_config["persistent"]["dbname"])

    def tearDown(self):
        pass

    def create_project(self, filename, targetname):
        workspace = Workspace()

        folds = 5
        reps = 5
        holdout_pct = 20
        mode = None
        missing_maps_to = None
        positive_class = None
        metric = None

        workspace.init_project({'filename': 'projects/'+str(self.pid)+'/raw/'+filename, 'uid':'1', 'pid': str(self.pid), 'active':1, 'stage':'eda:', 'originalName':filename, 'created':time.time()})
        workspace.init_dataset("universe", ['projects/'+str(self.pid)+'/raw/'+filename])
        vartypes, typeConvert = varTypeString(workspace.get_dataframe())
        workspace._update_metadata({'varTypeString': vartypes}, workspace.get_dataset_id_by_name("universe"))
        workspace.set_partition(folds, reps, holdout_pct=holdout_pct,total_size=160)
        workspace.set_target(targetname, missing_maps_to=missing_maps_to, positive_class=positive_class, metric=metric)

        return workspace

    @patch('ModelingMachine.engine.data_processor.APIClient')
    def get_requestdata(self, request, api_client_mock):
        """ create a fake request data to go with request1 """

        project = self.persistent.read(table='project',
                                     condition={'_id': ObjectId(request['pid'])},
                                     limit=(0,0),
                                     result=[])

        api_client_mock.return_value.get_project.return_value = project[0]

        #metadata_table = self.persistent.read(table='metadata', result=[])

        metadata = self.persistent.read(table='metadata',
                                     condition={'pid': ObjectId(request['pid']), 'newdata': {'$ne': True}},
                                     result=[])

        api_client_mock.return_value.get_metadata.return_value = metadata

        api_client_mock.return_value.get_task_code_from_id_list.return_value = self.bp_helper.user_task

        requestobj = WorkerRequest(request)
        rd = RequestData(requestobj)
        # print "RD datasets:", rd.datasets.keys()
        # print "RD target name:", rd.target_name
        # print "RD target type:", rd.target_type
        # print "RD target vector main:", type(rd.target_vector['main'])
        # print "RD partition:", rd.partition
        return rd

    def test_requestdata(self):
        ws = self.create_project(filename=self.datasets[0]['filename'], targetname=self.datasets[0]['target'][0])
        pid = str(ws.pid)
        dataset_id = ws.get_active_dataset_id()
        request = self.bp_helper.create_request(pid, dataset_id, self.bp1)
        request_data = self.get_requestdata(request)
        self.bp_helper.execute_blueprints([self.bp1, self.bp3], request_data, pid, dataset_id)

if __name__ == '__main__':
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logger.addHandler(console)

    unittest.main()
