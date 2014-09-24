import unittest
import pytest
from bson.objectid import ObjectId
from mock import patch, Mock
import json
import datetime

from config.engine import EngConfig
from config.test_config import db_config as config
from common.wrappers import database

from common.services.project import ProjectServiceBase as ProjectService
from common.services.autopilot import AutopilotService, DONE, MANUAL, SEMI, AUTO
import common.services.eda
from common.engine.progress import ProgressSink
import common
from common import load_class

from MMApp.entities.user import UserService
from MMApp.entities.roles import RoleProvider
from MMApp.entities.jobqueue import QueueService
from MMApp.entities.permissions import Roles, Permissions

from ModelingMachine.engine.worker_request import WorkerRequest

from common.services.model_refresh import refresh_model_by_lid, ModelBuild, ModelRefreshRequestError

from base_test_services import TestServiceBase

class TestModelRefreshService(TestServiceBase):
    """
    1. test the request function
    2. test each method of ModelRefresh db service
    """
    @classmethod
    def setUpClass(self):
        super(TestModelRefreshService, self).setUpClass()

    @classmethod
    def tearDownClass(self):
        super(TestModelRefreshService, self).tearDownClass()
        for table in ['model_refresh']:
            self.persistent.destroy(table=table)

    def setUp(self):
        super(TestModelRefreshService, self).setUp()
        for table in ['model_refresh']:
            self.persistent.destroy(table=table)

    @patch.dict(EngConfig, {'MODEL_REFRESH_API': True})
    @patch('common.services.model_refresh.ProjectService')
    @patch('common.services.model_refresh.QueueService')
    @patch('common.services.model_refresh.DatasetService')
    def test_request_failures1(self,*args,**kwargs):
        ds, qs, ps = args
        #any input arguments
        pid,lid,file_ids = ObjectId(None), ObjectId(None), None

        #case 1: model not found in leaderboard
        #expect: ModelRefreshRequestError
        ps.return_value.read_leaderboard_item.return_value = {}
        with self.assertRaises(ModelRefreshRequestError):
            out = refresh_model_by_lid(str(pid), str(lid), file_ids)
        ps.return_value.read_leaderboard_item.assert_called_once()

        self.assertEqual(qs.return_value.add.call_count, 0)
        
    @patch.dict(EngConfig, {'MODEL_REFRESH_API': True})
    @patch('common.services.model_refresh.ProjectService')
    @patch('common.services.model_refresh.QueueService')
    @patch('common.services.model_refresh.DatasetService')
    def test_request_failures2(self,*args,**kwargs):
        ds, qs, ps = args
        #any input arguments
        pid,lid,file_ids = ObjectId(None), ObjectId(None), None

        #case 2: model found in leaderboard, BUT no metadata
        #expect: ModelRefreshRequestError
        model = self.fake_leaderboard_item({'_id':lid, 'pid':pid})
        ps.return_value.read_leaderboard_item.return_value = model

        with self.assertRaises(ModelRefreshRequestError):
            out = refresh_model_by_lid(str(pid), str(lid), file_ids)

        ps.return_value.read_leaderboard_item.assert_called_once()
        ds.return_value.query.assert_called_once()

        self.assertEqual(qs.return_value.add.call_count, 0)


    @patch.dict(EngConfig, {'MODEL_REFRESH_API': True})
    @patch('common.services.model_refresh.ProjectService')
    @patch('common.services.model_refresh.QueueService')
    @patch('common.services.model_refresh.DatasetService')
    def test_request_failures3(self,*args,**kwargs):
        ds, qs, ps = args

        pid,lid,file_ids = ObjectId(None), ObjectId(None), ['invalid_id']

        #case 3: model found in leaderboard AND metadata found BUT invalid file_ids
        #expect: ModelRefreshRequestError
        model = self.fake_leaderboard_item({'_id':lid, 'pid':pid})
        ps.return_value.read_leaderboard_item.return_value = model

        metadata = [{'_id':ObjectId(model['dataset_id'])}, {'_id':ObjectId(None), 'newdata':True}]
        ds.return_value.query.return_value = metadata

        with self.assertRaises(ModelRefreshRequestError):
            out = refresh_model_by_lid(str(pid), str(lid), file_ids)

        ps.return_value.read_leaderboard_item.assert_called_once()
        ds.return_value.query.assert_called_once()

        self.assertEqual(qs.return_value.add.call_count, 0)

    @patch.dict(EngConfig, {'MODEL_REFRESH_API': True})
    @patch('common.services.model_refresh.ProjectService')
    @patch('common.services.model_refresh.QueueService')
    @patch('common.services.model_refresh.DatasetService')
    @patch('common.services.model_refresh.ModelBuild')
    def test_request_failures4(self,*args,**kwargs):
        mb, ds, qs, ps = args

        valid_file_id = ObjectId(None)
        pid,lid,file_ids = ObjectId(None), ObjectId(None), [str(valid_file_id)]

        #case 4: model OK, metadata OK, file_ids OK, BUT request already exists (duplicate)
        #expect: ModelRefreshRequestError
        model = self.fake_leaderboard_item({'_id':lid, 'pid':pid})
        ps.return_value.read_leaderboard_item.return_value = model

        metadata = [{'_id':ObjectId(model['dataset_id'])}, 
                    {'_id':ObjectId(None), 'newdata':True},
                    {'_id':valid_file_id, 'newdata':True}]
        ds.return_value.query.return_value = metadata

        mb.return_value.get.return_value = 'whatever'

        with self.assertRaises(ModelRefreshRequestError):
            out = refresh_model_by_lid(str(pid), str(lid), file_ids)

        ps.return_value.read_leaderboard_item.assert_called_once()
        ds.return_value.query.assert_called_once()
        mb.return_value.get.assert_called_once()

        self.assertEqual(qs.return_value.add.call_count, 0)

    @patch.dict(EngConfig, {'MODEL_REFRESH_API': True})
    @patch('common.services.model_refresh.ProjectService')
    @patch('common.services.model_refresh.QueueService')
    @patch('common.services.model_refresh.DatasetService')
    @patch('common.services.model_refresh.ModelBuild')
    def test_request_success(self,*args,**kwargs):
        mb, ds, qs, ps = args

        valid_file_id = ObjectId(None)
        pid,lid,file_ids = ObjectId(None), ObjectId(None), [str(valid_file_id)]

        #case 4: model OK, metadata OK, file_ids OK, request OK, 
        model = self.fake_leaderboard_item({'_id':lid, 'pid':pid})
        ps.return_value.read_leaderboard_item.return_value = model

        metadata = [{'_id':ObjectId(model['dataset_id'])}, 
                    {'_id':ObjectId(None), 'newdata':True},
                    {'_id':valid_file_id, 'newdata':True}]
        ds.return_value.query.return_value = metadata

        mb.return_value.get.return_value = []

        out = refresh_model_by_lid(str(pid), str(lid), file_ids)

        ps.return_value.read_leaderboard_item.assert_called_once()
        ds.return_value.query.assert_called_once()
        mb.return_value.get.assert_called_once()
        mb.return_value.insert.assert_called_once_with(str(lid), file_ids)

        self.assertEqual(qs.return_value.add.call_count, 1)

    @patch.dict(EngConfig, {'MODEL_REFRESH_API': False})
    @patch('common.services.model_refresh.ProjectService')
    @patch('common.services.model_refresh.QueueService')
    @patch('common.services.model_refresh.DatasetService')
    @patch('common.services.model_refresh.ModelBuild')
    def test_flag_off(self,*args,**kwargs):
        mb, ds, qs, ps = args

        valid_file_id = ObjectId(None)
        pid,lid,file_ids = ObjectId(None), ObjectId(None), [str(valid_file_id)]

        #case 4: model OK, metadata OK, file_ids OK, request OK, 
        model = self.fake_leaderboard_item({'_id':lid, 'pid':pid})
        ps.return_value.read_leaderboard_item.return_value = model

        metadata = [{'_id':ObjectId(model['dataset_id'])}, 
                    {'_id':ObjectId(None), 'newdata':True},
                    {'_id':valid_file_id, 'newdata':True}]
        ds.return_value.query.return_value = metadata

        mb.return_value.get.return_value = []

        with self.assertRaises(ModelRefreshRequestError):
            out = refresh_model_by_lid(str(pid), str(lid), file_ids)




    @patch.dict(EngConfig, {'MODEL_REFRESH_API': True})
    def test_ModelBuild_get_insert(self):
        username, uid, pid, dataset_id = self.create_project()
        mb = ModelBuild(pid)
        lid = str(ObjectId(None))
        file_ids = [str(ObjectId(None))]

        mb.insert(lid, file_ids)
        out = mb.get()
        self.assertIsInstance(out, list)
        self.assertEqual(len(out),1)

        out2 = mb.get(lid=ObjectId(lid), file_ids=file_ids)
        self.assertEqual(out,out2)

        lid2 = str(ObjectId(None))
        mb.insert(lid2, file_ids)
        out = mb.get()
        self.assertIsInstance(out, list)
        self.assertEqual(len(out),2)
        self.assertEqual({lid,lid2}, set([str(i['lid']) for i in out]))

        out3 = mb.get(lid=ObjectId(lid), file_ids=file_ids)
        self.assertEqual(out3,out2)
        
    @patch.dict(EngConfig, {'MODEL_REFRESH_API': True})
    def test_ModelBuild_complete(self):
        username, uid, pid, dataset_id = self.create_project()
        mb = ModelBuild(pid)
        lid = str(ObjectId(None))
        file_ids = [str(ObjectId(None))]

        build_id = mb.insert(lid, file_ids)

        out1 = mb.get(lid=ObjectId(lid), file_ids=file_ids)

        mb.complete(lid, build_id, datetime.datetime.utcnow().isoformat())

        out2 = mb.get(lid=ObjectId(lid), file_ids=file_ids)

        self.assertEqual(len(out2), 1)
        self.assertTrue('build_datetime' in out2[0])
        for key,value in out1[0].items():
            self.assertEqual(out2[0][key], value)

    @patch.dict(EngConfig, {'MODEL_REFRESH_API': True})
    def test_model_refresh_service_integration(self):
        # test without mocking services
        username, uid, pid, dataset_id = self.create_project()
        lid = ObjectId(None)
        model = self.fake_leaderboard_item({'_id':lid, 'pid':pid, 'dataset_id':dataset_id, 'uid':uid})
        self.persistent.create(model, table='leaderboard')

        qs = QueueService(pid, ProgressSink(), uid)
        q = qs.get()
        self.assertEqual(len(q),1)

        out = refresh_model_by_lid(str(pid), str(lid))

        qs = QueueService(pid, ProgressSink(), uid)
        q = qs.get()
        self.assertEqual(len(q),2)
        for key,value in out.items():
            self.assertEqual(q[1][key], value)
        
        #test queue processing/routing of the request
        with patch('MMApp.entities.jobqueue.SecureBrokerClient') as mock_compute:
            out = qs.start_new_tasks(2)
            mock_compute.return_value.refresh.assert_called_once()
            self.assertEqual(out,1)

        inprogress = self.tempstore.read(keyname='inprogress', index=str(pid), result={})
        req = json.loads(inprogress['1'])
        self.assertEqual(req['command'], 'refresh')

        #test worker request
        wq = WorkerRequest(req)
        self.assertEqual(wq.command, 'refresh')



        


        








if __name__=='__main__':
    unittest.main()




