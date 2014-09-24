############################################################################
#
#       unit test for Worker
#
#       Author: Tom DeGodoy
#
#       Copyright DataRobot, Inc. 2013
#
###########################################################################

import pandas
import numpy as np
import unittest
import hashlib
from bson.objectid import ObjectId
import os
import time
import datetime
import psutil
import signal
import json
import copy
import logging
from mock import Mock, patch, DEFAULT
import common.services.eda
from datetime import datetime
import pytest

from config.engine import EngConfig
from config.test_config import db_config
from common.wrappers import database

import ModelingMachine
from ModelingMachine.engine.pandas_data_utils import getX, getY, varTypes, varTypeString
from ModelingMachine.metablueprint import oldmb
from ModelingMachine import worker
from ModelingMachine.worker import ProjectStage

from MMApp.entities.dataset import DatasetService
from MMApp.entities.db_conn import DBConnections
from common.entities.job import blender_inputs
from common.services.queue_service_base import QueueServiceBase
from common.services.project import ProjectServiceBase as ProjectService
import common.services.autopilot as autopilot
import common.io as io
from common.services.flippers import FLIPPERS
from tests.IntegrationTests.storage_test_base import StorageTestBase


class MongoEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


class Metablueprint(oldmb.OldMetablueprint):
    pass

worker.Metablueprint = Metablueprint

class TestMMWorker(StorageTestBase):
    """
    """
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
        super(TestMMWorker, cls).setUpClass()
        # Call to method in StorageTestBase
        cls.test_directory, cls.datasets = cls.create_test_files()

        cls.dbs = DBConnections()
        cls.get_collection = cls.dbs.get_collection
        cls.redis_conn = cls.dbs.get_redis_connection()

        cls.tempstore = database.new('tempstore')
        cls.persistent = database.new('persistent')

        cls.bp1 = {}
        cls.bp1['1'] = (['NUM'],['NI'],'T')
        cls.bp1['2'] = (['1'],['GLMB'],'P')

        cls.bp2 = {}
        cls.bp2['1'] = (['NUM'],['NI'],'T')
        cls.bp2['2'] = (['CAT'],['DM'],'T')
        cls.bp2['3'] = (['1','2'],['ST','LR1'],'P')

        cls.bp3 = {}
        cls.bp3['1'] = (['NUM'],['NI'],'T')
        cls.bp3['2'] = (['CAT'],['DM'],'T')
        cls.bp3['3'] = (['TXT'],['TM'],'T')
        cls.bp3['4'] = (['2'],['LR1'],'T')
        cls.bp3['5'] = (['3'],['LR1'],'P')
        cls.bp3['6'] = (['1','4','5'],['RFI'],'P')

        workspace_patcher = patch.dict('ModelingMachine.engine.workspace.EngConfig', {'DATA_DIR': cls.test_directory}, clear = False)
        workspace_patcher.start()
        cls.patchers.append(workspace_patcher)

    @classmethod
    def tearDownClass(cls):
        super(TestMMWorker, cls).tearDownClass()
        cls.redis_conn.flushdb()
        cls.dbs.destroy_database()

    def setUp(self):
        super(TestMMWorker, self).setUp()
        self.redis_conn.flushdb()
        self.dbs.destroy_database()
        time.sleep(0.5)

        self.worker = worker.Worker(worker_id="1", request={}, pipe=None, connect=False)
        self.worker.pid = 1
        self.worker.pipe = Mock()
        self.worker.ctx = Mock()

    def tearDown(self):
        self.worker = None
        #kill child processes
        children = psutil.Process(os.getpid()).get_children(recursive=True)
        if len(children)>0:
            for child in children:
                try:
                    os.kill(child.pid,signal.SIGUSR1)
                except:
                    continue
            time.sleep(0.5) #some time for process kill signals to work

    def wait_for_eda(self, pid, ncols):
        #wait for eda to complete with a timeout of 10 seconds:
        t = time.time()
        while time.time()-t < 10:
            eda = self.get_eda_list(pid)
            if len(eda) == 1:
                eda = eda[0]
                if len(eda.get('eda', {}))==ncols:
                    break
            time.sleep(1)

        time.sleep(1) #give it a sec (jenkins server is slow)
        return eda.get('eda', {})

    def fetch_stage(self, pid):
        return self.redis_conn.get('stage:'+str(pid))

    def wait_wsready(self, pid, timeout=10):
        '''I think that we have already moved away from using wsready as a
        signal... but let's refactor first, then change functionality
        '''
        return self.redis_conn.blpop('wsready:'+str(pid), timeout)

    def current_redis_keys(self):
        return self.redis_conn.keys()

    def get_eda_list(self, pid):
        query_cursor = self.get_collection('eda').find({'pid':ObjectId(pid)})
        query = [i for i in query_cursor]
        return query

    def get_metadata_list(self, pid, dataset_id):
        query_cursor = self.get_collection('metadata').find({'pid':ObjectId(pid), '_id': ObjectId(dataset_id)})
        query = [i for i in query_cursor]
        return query

    def get_project_data(self, pid):
        query = self.get_collection('project').find_one({'_id':ObjectId(pid)})
        return query

    def set_to_full_auto(self, pid):
        self.redis_conn.hset('queue_settings:'+str(pid), 'mode', autopilot.AUTO)

    def read_userkeys(self, uid):
        return self.redis_conn.lrange('userkeys:'+str(uid), 0, -1)

    def read_top_of_queue(self, pid):
        return self.redis_conn.lrange('queue:'+str(pid), 0, 1)[0]

    def update_metadata_columns(self, pid, dataset_id, columns):
        self.get_collection('metadata').update({'pid':ObjectId(pid), '_id': ObjectId(dataset_id)},
                                               {'$set': {'columns': columns}})

    def get_metablueprint_data(self):
        return self.get_collection('metablueprint').find_one(
            {'pid':ObjectId(self.pid)})


    def create_project(self, file_id, pid, uid, ds, controls):
        upload_filename = 'projects/{}/raw/{}'.format(str(pid), file_id)

        if FLIPPERS.metadata_created_at_upload:
            ds_service = DatasetService(pid=self.pid, uid=uid,
                                        tempstore=self.tempstore,
                                        persistent=self.persistent)
            dataset_id = ds_service.create_metadata('universe',
                [upload_filename], controls, ds)
            p_service = ProjectService(pid=self.pid, uid=uid,
                                       tempstore=self.tempstore,
                                       persistent=self.persistent)
            p_service.init_project(upload_filename, file_id, dataset_id)


        request = {'originalName': file_id, 'dataset_id': upload_filename,
                   'uid': str(uid), 'pid': str(pid)}
        response = self.worker.create_project(request)

        return response


    @patch('ModelingMachine.worker.DatasetReader', autospec=True)
    @patch('ModelingMachine.worker.DatasetService', autospec=True)
    @patch('ModelingMachine.worker.ProjectService', autospec=True)
    def test_aim_failes_and_clears_submitted_target(self, MockProjectService,
                                                    MockDatsetService,
                                                    MockDatasetReader):
        uid = ObjectId()
        pid = ObjectId()
        request = {'uid':str(uid),'pid':str(pid),'target':'targetname','folds':5,'reps':5,'holdout_pct':20}

        project_service = MockProjectService.return_value
        workspace = Mock()
        project_service.validate_target.side_effect = ValueError('BOOM!')

        with patch.object(self.worker, 'tempstore') as mock_tempstore:
            mock_tempstore.read.return_value = ProjectStage.AIM

            result = self.worker.aim(request, workspace=workspace)
            self.assertIsNone(result)

            self.assertFalse(workspace.set_partition.called)
            self.assertTrue(project_service.clear_submitted_target.called)

    @patch('ModelingMachine.worker.DatasetReader', autospec=True)
    @patch('ModelingMachine.worker.DatasetService', autospec=True)
    @patch('ModelingMachine.worker.ProjectService', autospec=True)
    @patch('ModelingMachine.worker.EdaService', autospec=True)
    @patch('ModelingMachine.metablueprint.dummy_mb.DummyBlueprint', autospec=True)
    @patch('ModelingMachine.worker.MetablueprintService', autospec=True)
    def test_aim_sets_metrics_list(self, MockMbService, MockMb,
                                   MockEdaService,
                                   MockProjectService,
                                   MockDatasetService,
                                   MockDatasetReader,
                                   ):

        uid = ObjectId()
        pid = ObjectId()
        request = {'uid':str(uid),'pid':str(pid),'target':'targetname','folds':5,'reps':5,'holdout_pct':20}

        ps = ProjectService(str(pid), str(uid))
        mockps = MockProjectService.return_value
        ps.data = {'_id': str(pid), 'metric': None}
        mockps.create_metrics_list.side_effect = ps.create_metrics_list

        MockMb = MockMb.return_value
        MockMb.addMetadata.return_value = True
        MockMb._data = {'classname': 'DummyBlueprint'}
        mockeda = MockEdaService.return_value
        mockeda.get_target_metrics_list.return_value = ['AUC', 'Gini', 'LogLoss']
        workspace = Mock()
        workspace.get_metadata.return_value = {'shape': (1000, 1000)}
        workspace.get_target.return_value = {'type': 'Binary'}
        workspace.data = {'partition': 1}

        with patch.object(self.worker, 'tempstore') as mock_tempstore:
            mock_tempstore.read.return_value = ProjectStage.AIM

            result = self.worker.aim(request, workspace=workspace)
            self.assertIsNone(result)

            project = self.persistent.read(table='project', condition={'_id': pid}, result={})
            metric_detail = project['metric_detail']

            self.assertIn({'name': 'AUC', 'ascending': True} , metric_detail)


    @patch.object(ModelingMachine.worker.ProjectService, 'assert_has_permission')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    def test_plot_prediction(self,*args,**kwargs):
        uid = ObjectId()
        lid = ObjectId()
        pid = self.pid

        dataset = self.datasets[2]
        file_id = dataset['filename']
        targetname = dataset['target'][0]
        ds, controls = io.inspect_uploaded_file(
            os.path.join(self.test_directory,file_id), file_id)

        response = self.create_project(file_id, pid, uid, ds, controls)

        request = {'uid':str(uid),'pid':pid,'target':targetname,'folds':5,'reps':5,'holdout_pct':20}
        response = self.worker.aim(request)

        pred = [float(i)/float(ds.shape[0]) for i in range(ds.shape[0])]
        part = [1] * len(pred)

        #FIXME: need a better mock of the prediction object
        self.worker.persistent.create( {
            'dataset_id':'1234',
            'pid':ObjectId(pid),
            'lid':lid,
            'predicted-0':pred,
            'Full Model 100%':pred+pred, #some longer vector (including holdout)
            'partition':part,
            'row_index':range(ds.shape[0])},
                table='predictions')

        request = {'pid':str(pid), 'uid':str(uid), 'lid':str(lid)}
        with patch.object(self.worker, 'get_leaderboard_item') as mock1:
            with patch.object(ModelingMachine.worker.EdaService, 'column_importances') as mock2:
                mock1.return_value = {'dataset_id':'1234', 'samplepct':40}
                mock2.return_value = [(i,j) for j,i in enumerate(ds.columns)]
                out = self.worker.plot_prediction(request)

        nrows = []
        for key,value in out.items():
            #print
            #print key
            #print value
            self.assertNotEqual(key, targetname)
            self.assertIn(key, ds.columns)
            self.assertIsInstance(value, dict)
            self.assertEqual(set(value.keys()), set(['rows','act','pred','labels','order','position', 'weight']))
            #check that all lists in value have same length
            self.assertEqual(len(set([len(i) for i in value.values() if isinstance(i,list)])), 1)
            nrows += [sum(value['rows'])]
        self.assertEqual(len(set(nrows)),1) #all plots have the same total number of rows

        #test on 100% samplepct

        request = {'pid':str(pid), 'uid':str(uid), 'lid':str(lid)}
        with patch.object(self.worker, 'get_leaderboard_item') as mock1:
            with patch.object(ModelingMachine.worker.EdaService, 'column_importances') as mock2:
                mock1.return_value = {'dataset_id':'1234', 'samplepct':100}
                mock2.return_value = [(i,j) for j,i in enumerate(ds.columns)]
                out = self.worker.plot_prediction(request)

        nrows = []
        for key,value in out.items():
            #print
            #print key
            #print value
            self.assertNotEqual(key, targetname)
            self.assertIn(key, ds.columns)
            self.assertIsInstance(value, dict)
            self.assertEqual(set(value.keys()), set(['rows','act','pred','labels','order','position','weight']))
            #check that all lists in value have same length
            self.assertEqual(len(set([len(i) for i in value.values() if isinstance(i,list)])), 1)
            nrows += [sum(value['rows'])]
        self.assertEqual(len(set(nrows)),1) #all plots have the same total number of rows

    def assert_proj_data_equivalent(self, reference, computed, is_target_set=False):
        '''This is a test that data stored in the project is basically
        unchanged

        Parameters
        ----------
        reference : dict
            The reference data fixture
        computed : dict
            The newly computed data fixture
        target_set : boolean
            If the target has been set, there are additional details stored
            with the project that we need to keep track of
        '''
        self.assertEqual(set(reference.keys()), set(computed.keys()))
        should_equal_init = ['holdout_unlocked', 'originalName', 'stage', 'active']
        for key in should_equal_init:
            self.assertEqual(reference[key], computed[key])

        # These are assertions about the stored types of ObjectIds
        self.assertIsInstance(computed['_id'], ObjectId)
        self.assertIsInstance(computed['default_dataset_id'], basestring)
        self.assertTrue(ObjectId(computed['default_dataset_id']))

        self.assertIsInstance(computed['uid'], basestring)
        self.assertTrue(ObjectId(computed['uid']))

        if not is_target_set:
            return

        should_equal_now = ['holdout_pct', 'holdout_unlocked', 'metric',
                            'partition', 'target', 'target_options',
                            'originalName', 'stage']
        for key in should_equal_now:
            self.assertEqual(reference[key], computed[key])

        self.assertIsInstance(computed['created'], float)
        self.assertIsInstance(computed['default_dataset_id'], basestring)
        self.assertIsInstance(computed['_id'], ObjectId)
        self.assertIsInstance(computed['uid'], basestring)

    def assert_eda_equal(self, reference, computed):
        self.assertEqual(set(reference.keys()), set(computed.keys()))
        for key in reference.keys():
            self.assertEqual(set(reference[key].keys()),
                             set(computed[key].keys()))
            for subkey in reference[key].keys():
                if subkey == 'profile':
                    self.assert_eda_profile_equal(reference[key][subkey],
                                                  computed[key][subkey])
                elif subkey == 'metric_options':
                    # non deterministic ordering due to dict iteration
                    a = sorted(reference[key][subkey])
                    b = sorted(computed[key][subkey])
                    msg_if_fail = ('{}.{} differs:\n\tReference: {}'
                        '\n\tComputed: {}').format(
                            key, subkey, a, b)
                    self.assertEqual(a, b, msg_if_fail)
                else:
                    msg_if_fail = ('{}.{} differs:\n\tReference: {}'
                        '\n\tComputed: {}').format(
                            key, subkey, reference[key][subkey],
                            computed[key][subkey])
                    self.assertEqual(reference[key][subkey],
                                     computed[key][subkey], msg_if_fail)

    def assert_eda_profile_equal(self, reference, computed):
        numerical_close_keys = ['info', 'raw_info']

        self.assertEqual(set(reference.keys()), set(computed.keys()))
        for key in reference.keys():
            if key in numerical_close_keys:
                np.testing.assert_almost_equal(reference[key], computed[key], 5)
            elif key in ('plot', 'plot2'):
                # FIXME dunno why plot doesnt match teh fixture
                continue
            else:
                self.assertEqual(reference[key], computed[key], key)


    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    def test_eda_output_regression(self,*args):
        """ Test EDA and Project collections for changes.

        If this test fails:
          1. Do not panic
          2. Are these changes that affect schema? We might need to create
             another backwards compatibility test for it.
          3. Are you pretty comfortable that we have tests and logic for
             both your changes and the previous state of the code? Then update
             the fixtures for these tests (see the code below)
        """
        dataset = self.datasets[6]
        self.worker.accept_request = Mock()
        self.worker.accept_request.return_value = True

        file_id = dataset['filename']
        targetname = dataset['target'][0]
        ds, controls = io.inspect_uploaded_file(
            os.path.join(self.test_directory,file_id), file_id)
        colnames = list(ds.columns)
        ncols = len(colnames)

        uid = ObjectId()
        pid = self.pid

        #startCSV request
        response = self.create_project(file_id, pid, uid, ds, controls)
        self.assertNotEqual(response,None)

        #wait for workspace ready signal
        reply = self.wait_wsready(pid)
        self.assertNotEqual(reply,None)
        dataset_id = reply[1]


        eda = self.wait_for_eda(pid, ncols)
        # Use this to generate a new fixture once you are sure you have
        # understood all the effects your changes have made
        #with open('fastiron-eda1.json', 'wb') as out:
        #    json.dump(eda, out)

        project_data = self.get_project_data(pid)
        # Use these lines to generate a new project fixture once you are sure
        # you have understood all the effects your changes have made
        #with open('fastiron-proj1.json', 'wb') as out:
        #   json.dump(project_data, out, cls=MongoEncoder)

        ref_proj_json_path = os.path.join(self.testdatadir, 'fixtures',
                                          'fastiron-proj1.json')
        with open(ref_proj_json_path) as in_fp:
            reference_proj_data = json.load(in_fp)
            self.assert_proj_data_equivalent(reference_proj_data,
                                             project_data)

        # reference_eda_json_path = os.path.join(self.testdatadir, 'fixtures',
        #                                        'fastiron-eda1.json')
        # with open(reference_eda_json_path) as in_fp:
        #     reference_eda = json.load(in_fp)
        #     self.assert_eda_equal(reference_eda, eda)

        # Check EDA results before aim
        original_eda_list = self.get_eda_list(pid)
        self.assertEqual(len(original_eda_list), 1)
        original_eda = original_eda_list[0]
        original_eda_doc = original_eda['eda']

        #aim request
        self.set_to_full_auto(pid)
        request = {'uid':str(uid),'pid':pid,'target':targetname,'folds':5,'reps':5,'holdout_pct':20,'metric':'RMSE'}
        response = self.worker.aim(request)

        t = time.time()
        while time.time()-t < 10:
            stage = self.fetch_stage(pid)
            if stage == 'modeling:':
                break
            time.sleep(1)

        # Check EDA results after aim
        eda_list = self.get_eda_list(pid)
        eda = eda_list[0]['eda']

        project_data = self.get_project_data(pid)
        # Use these lines to generate a new project fixture once you are sure
        # you have understood all the effects your changes have made
        #with open('fastiron-proj2.json', 'wb') as out:
        #   json.dump(project_data, out, cls=MongoEncoder)

        ref_proj_json_path = os.path.join(self.testdatadir, 'fixtures',
                                          'fastiron-proj2.json')
        with open(ref_proj_json_path) as in_fp:
            reference_proj_data = json.load(in_fp)
            self.assert_proj_data_equivalent(reference_proj_data,
                                             project_data,
                                             is_target_set=True)

        # Use the following lines to generate a new fixture and move it into
        # the test directory once you are sure your the effects of your
        # changes are handled
        #with open('fastiron-eda2.json', 'wb') as out:
        #    json.dump(eda, out)

        # reference_eda2_path = os.path.join(self.testdatadir, 'fixtures',
        #                                    'fastiron-eda2.json')
        # with open(reference_eda2_path) as in_fp:
        #     reference_eda = json.load(in_fp)
        #     self.assert_eda_equal(reference_eda, eda)

    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    def test_eda_dates(self,*args):
        """ test EDA handles dates correctly
        """
        dataset = self.datasets[2]
        self.worker.accept_request = Mock()
        self.worker.accept_request.return_value = True

        file_id = dataset['filename']
        targetname = dataset['target'][0]
        ds, controls = io.inspect_uploaded_file(
            os.path.join(self.test_directory,file_id), file_id)
        colnames = list(ds.columns)
        ncols = len(colnames)

        uid = ObjectId()
        pid = self.pid

        #startCSV request
        response = self.create_project(file_id, pid, uid, ds, controls)
        self.assertNotEqual(response,None)

        #wait for workspace ready signal
        reply = self.wait_wsready(pid)
        self.assertNotEqual(reply,None)
        dataset_id = reply[1]

        eda = self.wait_for_eda(pid, ncols)

        # Check EDA results before aim
        original_eda_list = self.get_eda_list(pid)
        self.assertEqual(len(original_eda_list), 1)
        original_eda = original_eda_list[0]
        date_eda = original_eda['eda']['PurchDate']
        # check histogram date formatting
        plot_keys = [i[0] for i in date_eda['profile']['plot']]
        self.assertTrue(all([isinstance(datetime.strptime(i,'%m/%d/%Y').toordinal(),int) for i in plot_keys]))
        plot2_keys = [i[0] for i in date_eda['profile']['plot2']]
        self.assertIn('=All Other=',plot2_keys)
        self.assertTrue(all([i=='=All Other=' or isinstance(datetime.strptime(i,'%m/%d/%Y').toordinal(),int) for i in plot2_keys]))
        self.assertEqual(date_eda['profile']['type'],'D')
        self.assertEqual(date_eda['types']['conversion'], '%m/%d/%Y')
        # check EDA summary
        self.assertEqual(date_eda['summary'], [43, 0, u'04/13/2010', 158.72407810017359, u'02/18/2009', u'03/15/2010', u'12/20/2010'])

        #aim request
        self.set_to_full_auto(pid)
        request = {'uid':str(uid),'pid':pid,'target':targetname,'folds':5,'reps':5,'holdout_pct':20,'metric':'RMSE'}
        response = self.worker.aim(request)

        t = time.time()
        while time.time()-t < 10:
            stage = self.fetch_stage(pid)
            if stage == 'modeling:':
                break
            time.sleep(1)

        # Check EDA results after aim
        eda_list = self.get_eda_list(pid)
        date_eda = eda_list[0]['eda']['PurchDate']
        plot_keys = [i[0] for i in date_eda['profile']['plot']]
        self.assertTrue(all([isinstance(datetime.strptime(i,'%m/%d/%Y').toordinal(),int) for i in plot_keys]))
        plot2_keys = [i[0] for i in date_eda['profile']['plot2']]
        # check histogram date formatting
        self.assertIn('=All Other=',plot2_keys)
        self.assertTrue(all([i=='=All Other=' or isinstance(datetime.strptime(i,'%m/%d/%Y').toordinal(),int) for i in plot2_keys]))
        self.assertEqual(date_eda['profile']['type'],'D')
        self.assertEqual(date_eda['types']['conversion'], '%m/%d/%Y')
        # check EDA summary
        self.assertEqual(date_eda['summary'], [43, 0, u'04/13/2010', 158.72407810017359, u'02/18/2009', u'03/15/2010', u'12/20/2010'])
        # Make sure ACE works
        self.assertAlmostEqual(date_eda['profile']['info'],0.0178711232)

    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    def run_full_test(self,dataset,*args):
        """ runs all worker requests in sequence on a given dataset
        """
        self.worker.accept_request = Mock()
        self.worker.accept_request.return_value = True

        file_id = dataset['filename']
        targetname = dataset['target'][0]
        ds, controls = io.inspect_uploaded_file(
            os.path.join(self.test_directory,file_id), file_id)
        colnames = list(ds.columns)
        ncols = len(colnames)

        self.assertEqual(self.current_redis_keys(),[])

        uid = ObjectId()
        pid = self.pid

        #startCSV request
        response = self.create_project(file_id, pid, uid, ds, controls)
        self.assertNotEqual(response,None)

        #wait for workspace ready signal
        reply = self.wait_wsready(pid)
        self.assertNotEqual(reply,None)
        dataset_id = reply[1]

        query_cursor = self.get_collection('metadata').find({'pid':ObjectId(pid), 'name': "Raw Features"})
        query = [i for i in query_cursor]
        self.assertEqual(len(query), 1)
        raw_dataset_id = str(query[0]['_id'])

        query_cursor = self.get_collection('metadata').find({'pid':ObjectId(pid), 'name': "Informative Features"})
        query = [i for i in query_cursor]
        self.assertEqual(len(query), 1)
        infofeatures_dataset_id = str(query[0]['_id'])

        eda = self.wait_for_eda(pid, ncols)
        #check redis

        original_eda_list = self.get_eda_list(pid)
        self.assertEqual(len(original_eda_list), 1)
        original_eda = original_eda_list[0]

        #check that this doesn't change anything
        self.worker.eda({"pid": pid, "uid": uid})

        eda_list = self.get_eda_list(pid)
        self.assertEqual(len(eda_list), 1)
        new_eda = eda_list[0]

        for key in original_eda["eda"]:
            self.assertEqual(original_eda["eda"][key], new_eda["eda"][key])

        self.assertItemsEqual(original_eda, new_eda)

        original_eda_column_names = [column["name"] for column in original_eda["eda"].values()]

        metadata_list = self.get_metadata_list(pid, dataset_id)
        self.assertEqual(len(metadata_list), 1)
        metadata = metadata_list[0]

        self.assertEqual(len(new_eda["eda"]), len(metadata["columns"]))


        new_columns = copy.deepcopy(metadata["columns"])
        new_columns.append([0, "transform_test1", 10])
        new_columns.append([2, "transform_test2", 11])

        self.update_metadata_columns(pid, dataset_id, new_columns)

        query_cursor = self.get_collection('metadata').find({'pid':ObjectId(pid), '_id': ObjectId(dataset_id)})
        query = [i for i in query_cursor]
        self.assertEqual(len(query), 1)
        self.assertEqual(new_columns, query[0].get("columns"))

        colnames += ["transform_test1", "transform_test2"]
        ncols += 2

        #check that this adds the new eda
        self.worker.eda({"pid": pid, "uid": uid})

        eda = self.wait_for_eda(pid, ncols)

        eda_list = self.get_eda_list(pid)
        self.assertEqual(len(eda_list), 1)
        new_eda = eda_list[0]
        new_eda_column_names = [column["name"] for column in new_eda["eda"].values()]
        new_column_names = [column[1] for column in new_columns]

        self.assertItemsEqual(new_eda_column_names, new_column_names)


        redis_keys = set(['userkeys:'+str(uid), 'stage:'+str(pid)])
        self.assertEqual(set(self.current_redis_keys()),redis_keys)
        #check eda
        self.assertEqual(len(eda), ncols)
        for key in eda:
            #This will fail if any variables have low info
            item = eda[key]
            self.assertIsInstance(item,dict)
            if any(item['low_info'].values()):
                self.assertItemsEqual(item.keys(),['id',
                                                   'summary',
                                                   'raw_variable_index',
                                                   'transform_id',
                                                   'transform_args',
                                                   'name',
                                                   'low_info',
                                                   'types'])
            else:
                self.assertItemsEqual(item.keys(),['id',
                                                   'profile',
                                                   'summary',
                                                   'raw_variable_index',
                                                   'transform_id',
                                                   'transform_args',
                                                   'name',
                                                   'low_info',
                                                   'metric_options',
                                                   'types'])
                self.assertIsInstance(item['profile'],dict)
                self.assertGreaterEqual(set(item['profile'].keys()),set(['plot','name','y','type']))
                self.assertIsInstance(item['profile']['plot'],list)
                self.assertIsInstance(item['profile']['name'],unicode)
                self.assertIsInstance(item['profile']['type'],unicode)
                self.assertEqual(item['profile']['y'],None)
                if item['profile'].get('type')=='N':
                    self.assertIsInstance(item['profile']['miss_count'],float)
                    self.assertIsInstance(item['profile']['plot2'],list)
                    self.assertEqual(item['profile']['miss_ymean'],None)

            self.assertIsInstance(item['summary'],list)
            self.assertLessEqual(len(item['summary']),7)
            for el in item['summary']:
                self.assertIn(type(el),[int,float,unicode])

        #check other redis keys
        userkeys = self.read_userkeys(uid)
        self.assertEqual(userkeys,[str(pid)])
        stage = self.fetch_stage(pid)
        self.assertEqual(stage,'aim:')

        #check workspace created (and stored on Mongo DB)
        query_cursor = self.get_collection('project').find({'_id':ObjectId(pid)})
        query = [i for i in query_cursor]
        self.assertEqual(len(query),1)
        project = query[0]
        #print project
        answer = set([u'_id', u'uid', u'default_dataset_id', u'stage',u'active',u'holdout_unlocked',u'originalName',u'created'])
        self.assertEqual(set(project.keys()),answer)
        self.assertEqual(project['uid'],str(uid))
        self.assertEqual(project['default_dataset_id'],infofeatures_dataset_id)
        self.assertEqual(project['stage'],'aim:')

        #check metadata
        query_cursor = self.get_collection('metadata').find({'pid':ObjectId(pid), 'name': "universe"})
        query = [i for i in query_cursor]
        self.assertEqual(len(query),1)
        thismeta = query[0]
        answer,tc = varTypeString(ds)
        self.assertEqual(thismeta.get('varTypeString'),answer)
        answer = list(ds.shape)
        self.assertEqual(thismeta.get('shape'),answer)
        self.assertEqual([column[1] for column in thismeta.get('columns')],[i.replace('-','_') for i in colnames])
        self.assertEqual(project.get('target'),None) #target not set yet

        #check eda record on mongo DB
        eda_list = self.get_eda_list(pid)
        self.assertEqual(len(eda_list),1)
        rec = eda_list[0]

        self.assertEqual(rec['dataset_id'],"universe") #dataset_id from the request
        self.assertIsInstance(rec['eda'],dict)
        self.assertEqual(set(rec.keys()),set(['eda','pid','dataset_id','_id']))
        self.assertIsInstance(rec['pid'],ObjectId)
        self.assertIsInstance(rec['dataset_id'],unicode)
        self.assertIsInstance(rec['_id'],ObjectId)
        self.assertIsInstance(rec['eda'][colnames[0]],dict)
        if not any(rec['eda'][colnames[0]]['low_info'].keys()):
            self.assertEqual(set(rec['eda'][colnames[0]].keys()),set(['id','summary','profile', 'name',
                                                              'raw_variable_index', 'transform_id',
                                                              'transform_args', 'low_info', 'metric_options','types']))
            self.assertIsInstance(rec['eda'][colnames[0]]['profile'],dict)
        self.assertIsInstance(rec['eda'][colnames[0]]['summary'],list)

        #aim request
        # TODO - The webserver is setting the queue_settings before this code
        # is called; we got around this by just having a default autopilot
        # mode - however, that is not how the app functions at all, and should
        # be changed.  The best course is probably to move the setting of the
        # autopilot mode down into the worker so that it is easier to test
        # and more like real running conditions
        self.set_to_full_auto(pid)
        request = {'uid':str(uid),'pid':pid,'target':targetname,'folds':5,'reps':5,'holdout_pct':20,'metric':'RMSE'}
        response = self.worker.aim(request)

        # TODO - we really need to make worker unit testable
        # Make sure that the newly created metablueprint has the flags we
        # think it should. Those values are stored in the database.
        # If we get more flags, we should factor this assertion
        mb_data = self.get_metablueprint_data()
        mb_flags = mb_data['flags']
        self.assertTrue(mb_flags['submitted_jobs_stored'])

        #check redis
        answer = ['userkeys:'+str(uid), 'qid_counter:'+str(pid),
                'queue:'+str(pid), 'stage:'+str(pid), 'queue_settings:'+str(pid)]

        self.assertGreaterEqual(set(self.current_redis_keys()),set(answer))

        self.worker.eda({"pid": pid, "uid": uid})

        #check mongo
        query_cursor = self.get_collection('project').find({'_id':ObjectId(pid)})
        query = [i for i in query_cursor]
        self.assertEqual(len(query),1)
        doc = query[0]
        #self.assertEqual(doc.get('stage'),'eda2:')

        #check project
        query_cursor = self.get_collection('project').find({'_id':ObjectId(pid)})
        query = [i for i in query_cursor]
        self.assertEqual(len(query),1)
        project = query[0]
        self.assertIsInstance(project.get('target'),dict) #now target should be set

        #univariate request
        # #wait for univariates to complete
        t = time.time()
        while time.time()-t < 10:
            # query = self.get_eda_list(pid)
            # self.assertEqual(len(query),1)
            # column_profiles = ['y' in column.get('profile', {}) for column in query[0].get('eda', {})]
            # if all(column_profiles) and len(column_profiles) == ncols:
            #     break
            stage = self.fetch_stage(pid)
            if stage == 'modeling:':
                break
            time.sleep(1)

        time.sleep(1) #give it a sec (jenkins server is slow)
        #check redis
        answer = ['userkeys:'+str(uid),
                'qid_counter:'+str(pid), 'queue:'+str(pid),
                'stage:'+str(pid), 'queue_settings:'+str(pid)]

        self.assertGreaterEqual(set(self.current_redis_keys()),set(answer))
        stage = self.fetch_stage(pid)
        self.assertEqual(stage,'modeling:')

        eda_list = self.get_eda_list(pid)
        self.assertEqual(len(eda_list),1)
        univar = eda_list[0]['eda']
        for key in univar:
            univar[key] = univar[key].get('profile')

        #check univariates
        self.assertIsInstance(univar,dict)
        self.assertItemsEqual(univar.keys(),colnames)
        for key in univar:
            item = univar[key]
            if item:
                self.assertIsInstance(item['plot'],list)
                self.assertIsInstance(item['name'],unicode)
                if item.get('info', 1) <= 0.005:
                    continue
                self.assertEqual(item['y'],targetname)
                self.assertIsInstance(item['type'],unicode)
                if item['type']=='N':
                    self.assertIsInstance(item['miss_count'],float)
                    self.assertIsInstance(item['plot2'],list)
                    self.assertIsInstance(item['miss_ymean'],float)

        queue_query = self.read_top_of_queue(pid)
        item = json.loads(queue_query)
        answer = set([u'blueprint', u'features', u'icons', u'max_reps',
                      u'samplepct', u'bp', u'pid', u'model_type', u'max_folds',  u'qid',
                      u'dataset_id', u'blueprint_id', u'lid'])
        self.assertGreaterEqual(set(item.keys()),answer)
        self.assertIsInstance(item.get('blueprint'),dict)
        self.assertEqual(item.get('pid'),str(pid))
        self.assertEqual(item.get('dataset_id'),infofeatures_dataset_id)


    def test_on_credit_data(self):
        self.run_full_test(self.datasets[0])


    '''
    def test_on_allstate_data(self):
        self.run_full_test(self.datasets[1])

    def test_on_kickcars_data(self):
        self.run_full_test(self.datasets[2])
        '''

    def dont_test_logmsg(self):
        request = {}
        request['pid'] = '12345'
        request['blueprint'] = 'some blueprint text'
        request['qid'] = '1'
        request['uid'] = '54321'
        request['lid'] = '98765'
        request['dataset_id'] = '56789'
        request['command'] = 'fit'

        worker_instance = worker.Worker(worker_id="1", request=request, pipe = Mock(), connect = False)

        # Arrange: Make it fail
        with patch.multiple(worker_instance, tempstore = DEFAULT, logger = DEFAULT, accept_request = DEFAULT) as mocks:
            mocks['tempstore'].read.side_effect = Exception('Boom!')
            mocks['accept_request'].return_value = True

            #Act
            try:
                worker_instance.fit_blueprint(request)
            except:
                pass

            #Assert:
            self.assertTrue(mocks['logger'].error.called)


@pytest.mark.db
class TestEDAOfOldProject(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tempstore = database.new('tempstore')
        cls.persistent = database.new('persistent')

    def setUp(self):
        self.tempstore.conn.flushdb()
        self.persistent.destroy(table='project')
        self.persistent.destroy(table='metadata')
        self.persistent.destroy(table='metablueprint')
        self.uid = ObjectId(u'5359d6cb8bd88f5cddefd3a8')

        metadata_doc = {
            'columns': [[1, 'IsBadBuy', 0],
                        [4, u'VehYear', 0],
                        [16, u'Size', 0]],
            'varTypeString': u'NNC',
            'name': 'universe',
        }
        self.dataset_id = self.persistent.create(table='metadata',
                                                 values=metadata_doc)
        proj_doc = {
            u'active': 1,
            u'created': 1402071398.928407,
            u'default_dataset_id': str(self.dataset_id),
            u'holdout_pct': 20,
            u'holdout_unlocked': False,
            u'metric': u'LogLoss',
            u'mode': 2,
            u'originalName': u'kickcars-sample-200.csv',
            u'partition': {u'folds': 5,
                u'holdout_pct': 20,
                u'reps': 5,
                u'seed': 0,
                u'total_size': 200},
            u'roles': {str(self.uid): [u'OWNER']},
            u'stage': u'modeling:',
            u'target': {u'name': u'IsBadBuy', u'size': 158.0,
                        u'type': u'Binary'},
            u'target_options': {u'missing_maps_to': None,
                                u'name': u'IsBadBuy',
                                u'positive_class': None},
            u'tokens': {str(self.uid): u'-mWkLYluIYOc_Q=='},
            u'uid': str(self.uid),
            u'version': 1}
        self.pid = self.persistent.create(table='project',
                                          values=proj_doc)

    @patch('ModelingMachine.worker.eda_multi', autospec=True)
    @patch('ModelingMachine.worker.DatasetReader', autospec=True)
    def test_EDA_can_get_a_default_metablueprint_on_old_project(self,
            MockReader, mock_eda):
        '''Before, there was no attribute `classname` for the metablueprint.
        Now, however, we use it to know which metablueprint class to
        instantiate.  We need to be able to fall back on one.
        '''
        request = {'uid': str(self.uid),
                   'pid': str(self.pid),
                   'columns': ['NewGuy']}

        metadata_doc = {
            'columns': [[1, 'IsBadBuy', 0],
                        [4, u'ReviewFreeText', 0]],
            'varTypeString': u'NT',
            'pid': self.pid,
            'name': 'universe',
        }
        test_dataset_id = self.persistent.create(table='metadata',
                                                 values=metadata_doc)
        metablueprint_doc = {
            'pid': self.pid,  # Specifically, classname not set
        }
        faked_metablueprint = self.persistent.create(table='metablueprint',
                                                     values=metablueprint_doc)

        tested_worker = ModelingMachine.worker.Worker(worker_id=0, request=None, pipe=None,
            connect=False, tempstore=self.tempstore, persistent=self.persistent)

        mock_eda.eda_stream.return_value = ({}, [])

        work_request = {'pid': self.pid, 'uid': self.uid, 'columns': ['SalePrice']}
        tested_worker.eda(request)  # Would fail at
                                    # mb_service.get_metablueprint if bug
                                    # still existed


@pytest.mark.db
@patch('ModelingMachine.worker.Worker.acquire_metablueprint')
class TestNextSteps(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tempstore = database.new('tempstore')
        cls.persistent = database.new('persistent')

    def setUp(self):
        self.tempstore.conn.flushdb()
        self.persistent.destroy(table='project')
        self.persistent.destroy(table='metadata')
        self.persistent.destroy(table='metablueprint')
        self.uid = ObjectId(u'5359d6cb8bd88f5cddefd3a8')

        metadata_doc = {
            'columns': [[1, 'IsBadBuy', 0],
                        [4, u'VehYear', 0],
                        [16, u'Size', 0]],
            'varTypeString': u'NNC',
            'name': 'universe',
        }
        self.dataset_id = self.persistent.create(table='metadata',
                                                 values=metadata_doc)
        proj_doc = {
            u'active': 1,
            u'created': 1402071398.928407,
            u'default_dataset_id': str(self.dataset_id),
            u'holdout_pct': 20,
            u'holdout_unlocked': False,
            u'metric': u'LogLoss',
            u'mode': 2,
            u'originalName': u'kickcars-sample-200.csv',
            u'partition': {u'folds': 5,
                u'holdout_pct': 20,
                u'reps': 5,
                u'seed': 0,
                u'total_size': 200},
            u'roles': {str(self.uid): [u'OWNER']},
            u'stage': u'modeling:',
            u'target': {u'name': u'IsBadBuy', u'size': 158.0, u'type': u'Binary'},
            u'target_options': {u'missing_maps_to': None,
                u'name': u'IsBadBuy',
                u'positive_class': None},
            u'tokens': {str(self.uid): u'-mWkLYluIYOc_Q=='},
            u'uid': str(self.uid),
            u'version': 1}
        self.pid = self.persistent.create(table='project',
                                          values=proj_doc)


    def test_next_steps_with_additional_dataset_okay(self, MockMB):
        uploaded_data_meta_doc = {
            u'created': datetime(2014, 6, 6, 16, 18, 0, 575000),
            u'dataset_id': u'projects/5391e9668bd88f74e3431b4c/raw/65530979-8039-4f29-a982-d395cf7b3629',
            u'files': [u'projects/5391e9668bd88f74e3431b4c/raw/65530979-8039-4f29-a982-d395cf7b3629'],
            u'name': u'projects/5391e9668bd88f74e3431b4c/raw/65530979-8039-4f29-a982-d395cf7b3629',
            u'newdata': True,
            u'originalName': u'kickcars-training-sample.csv',
            u'pid': self.pid}

        self.persistent.create(table='metadata',
                               values=uploaded_data_meta_doc)

        tested_worker = ModelingMachine.worker.Worker(worker_id=0, request=None, pipe=None,
            connect=False, tempstore=self.tempstore, persistent=self.persistent)

        work_request = {'pid': self.pid, 'uid': self.uid,
                        'dataset_id': self.dataset_id}
        tested_worker.next_steps(work_request, check_stage=False)
        self.assertTrue(MockMB.return_value.called)

    def test_next_steps_with_text_only_datset_no_reference_models(self, MockMB):
        metadata_doc = {
            'columns': [[1, 'IsBadBuy', 0],
                        [4, u'ReviewFreeText', 0]],
            'varTypeString': u'NT',
            'name': 'universe',
        }
        test_dataset_id = self.persistent.create(table='metadata',
                                                 values=metadata_doc)

        tested_worker = ModelingMachine.worker.Worker(worker_id=0, request=None, pipe=None,
            connect=False, tempstore=self.tempstore, persistent=self.persistent)

        work_request = {'pid': self.pid, 'uid': self.uid,
                        'dataset_id': test_dataset_id}
        tested_worker.next_steps(work_request, check_stage=False)
        MockMB.assert_called_once_with(self.pid, self.uid,
                                       reference_models=False)
        self.assertTrue(MockMB.return_value.called)

if __name__ == '__main__':
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    logfile = logging.FileHandler(__file__+'.log',mode='w')
    logger.addHandler(console)
    logger.addHandler(logfile)

    unittest.main()
