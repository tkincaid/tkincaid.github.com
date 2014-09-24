import pandas
import numpy as np
import unittest
import logging
import psutil
import os
import time
import copy
import pytest
import common
import math
import json
from math import log
from mock import Mock, patch, DEFAULT
from bson.objectid import ObjectId
from tests.IntegrationTests.storage_test_base import StorageTestBase
from MMApp.entities.user import UserService
from ModelingMachine import worker
from ModelingMachine.engine.partition import NewPartition,Partition
from ModelingMachine.engine.mocks import DB_APIClient, DelayedStorageClient
from ModelingMachine.engine.data_processor import RequestData
from ModelingMachine.engine.worker_request import VertexDefinition, WorkerRequest
from ModelingMachine.engine.secure_worker import SecureWorker
from MMApp.entities.project import ProjectService
from MMApp.entities.dataset import DatasetService
from common.services.queue_service_base import QueueServiceBase
from MMApp.entities.db_conn import DBConnections
from common.wrappers import database
import common.services.eda
from config.engine import EngConfig
from config.test_config import db_config
config = db_config
import ModelingMachine
import common.io as io

from common.services.flippers import FLIPPERS

logger = logging.getLogger('datarobot')
logger.setLevel(logging.DEBUG)

class TestNewPartition(StorageTestBase):
    @classmethod
    def setUpClass(cls):
        super(TestNewPartition, cls).setUpClass()
        # Call to method in StorageTestBase
        cls.test_directory, cls.datasets = cls.create_test_files()

        cls.dbs = DBConnections()
        cls.get_collection = cls.dbs.get_collection
        cls.redis_conn = cls.dbs.get_redis_connection()
        cls.tempstore = database.new("tempstore")
        cls.persistent = database.new("persistent")
        cls.request =  {
            'blueprint': {'1': (['NUM'], ['NI'], 'T'), '2': (['1'], ['GLMB'], 'P')},
            'lid': '538e3344f9ac6359c38e7d7e',
            'uid': '532329fcbb51f7015cf64c96',
            'blueprint_id': 'd4c06a5c23cf1d917019720bceba32c8',
            'total_size': 200.0,
            'icons': [0],
            'bp': 1,
            'new_lid': True,
            'max_folds': 0}

    @classmethod
    def tearDownClass(cls):
        super(TestNewPartition, cls).tearDownClass()
        cls.clear_tempstore_except_workers()
        cls.dbs.destroy_database()

    @classmethod
    def clear_tempstore_except_workers(self):
        workers = set(self.tempstore.conn.smembers('workers'))
        secure_workers = set(self.tempstore.conn.smembers('secure-workers'))
        ide_workers = set(self.tempstore.conn.smembers('ide-workers'))
        self.tempstore.conn.flushdb()
        if workers:
            self.tempstore.create(keyname='workers', values=workers)
        if secure_workers:
            self.tempstore.create(keyname='secure-workers', values=secure_workers)
        if ide_workers:
            self.tempstore.create(keyname='ide-workers', values=ide_workers)

    def setUp(self):
        super(TestNewPartition, self).setUp()
        self.clear_tempstore_except_workers()
        self.dbs.destroy_database()
        time.sleep(0.5)

        self.worker = worker.Worker(worker_id="1", request={}, pipe=None, connect=False)
        self.worker.pid = 1

        self.worker.pipe = Mock()
        self.worker.ctx = Mock()
        self.worker.accept_request = Mock()
        self.worker.accept_request.return_value = True

        self.username = 'a@asdf.com'
        userservice = UserService(self.username)
        userservice.create_account('asdfasdf')
        self.uid = userservice.uid

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

    def create_project(self, dataset, aim_request):
        file_id = dataset['filename']
        pid = ObjectId(aim_request['pid'])

        # The test creates a project self.pid and loads the files we need
        # We look there, which is equivalent to how we clone a project
        upload_filename = 'projects/{}/raw/{}'.format(self.pid, file_id)
        if FLIPPERS.metadata_created_at_upload:
            filepath = os.path.join(self.test_directory, file_id)
            ds, controls = io.inspect_uploaded_file(filepath, file_id)
            ds_service = DatasetService(pid=pid, uid=self.uid,
                                        tempstore=self.tempstore,
                                        persistent=self.persistent)
            dataset_id = ds_service.create_metadata('universe', [upload_filename], controls, ds)
            p_service = ProjectService(pid=pid, uid=self.uid,
                                       tempstore=self.tempstore,
                                       persistent=self.persistent)
            p_service.init_project(upload_filename, file_id, dataset_id)

        request = {'originalName': file_id, 'dataset_id': upload_filename, 'uid':str(self.uid),
                   'pid' : str(pid), 'metric': dataset['metric']}
        self.worker.create_project(request)

        targetname = dataset['target'][0]
        self.redis_conn.hset('queue_settings:'+str(pid), 'mode', '0')
        response = self.worker.aim(aim_request)
        self.persistent.update(table='project',condition={'_id':pid},values={'version':1.1})
        return str(pid)

    def purge_queue(self,pid,purge_all=False):
        self.redis_conn.delete('queue:'+str(pid))
        if purge_all:
            self.tempstore.destroy(keyname='errors', index=str(pid))
            self.tempstore.destroy(keyname='inprogress', index=str(pid))
            self.tempstore.destroy(keyname='onhold', index=str(pid))
            self.tempstore.destroy(keyname='onhold_reqs', index=str(pid))

    def next_job_from_queue(self,pid,last=False):
        if last:
            queue_query = self.redis_conn.rpop('queue:'+str(pid))
        else:
            queue_query = self.redis_conn.lpop('queue:'+str(pid))
        if not queue_query:
            return {}
        item = json.loads(queue_query)
        self.redis_conn.hmset('inprogress:'+str(pid),{item['qid']: queue_query})
        if item.get('predict'):
            item['command'] = 'predict'
        else:
            item['command'] = 'fit'
        item['uid'] = str(self.uid)
        return item


    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    def test_cv_options(self,*args):
        dataset = self.datasets[0]
        for reps in (2,6):
            for holdout_pct in (0,15,30):
                pid = ModelingMachine.engine.mocks.ProjectService.create_project(self.uid)

                total_size = 200
                aim_request = {'uid': str(self.uid), 'pid': str(pid),  #Not self.pid - equivalent to cloning project
                               'target': 'SeriousDlqin2yrs','metric':'RMSE',
                               'folds': 5, 'reps':reps, 'holdout_pct':holdout_pct,
                               'mode': 0}
                pid = self.create_project(dataset, aim_request)
                dataset_id = self.tempstore.read(keyname='wsready', index=str(pid), result=[])[0]
                q = QueueServiceBase(str(pid), Mock(), uid=self.uid)
                self.purge_queue(pid, purge_all=True)
                for ss in (44,66,80,90,100):
                    request = copy.copy(self.request)
                    request['dataset_id'] = dataset_id
                    request['pid'] = pid
                    request['samplepct'] = ss
                    if ss > (reps-1.0)/reps*(100-holdout_pct):
                        request['partitions'] = [(-1,-1)]
                    else:
                        request['max_reps'] = reps
                    self.persistent.destroy(condition={'pid':ObjectId(pid)},table='leaderboard')
                    q.put([request])
                    k = 0
                    while True:
                        item = self.next_job_from_queue(pid)
                        if not item:
                            break
                        self.assertEqual(item['samplepct'], ss)
                        self.assertEqual(item['total_size'], 200.0)
                        if ss <= (reps-1.0)/reps*(100-holdout_pct):
                            self.assertEqual(item['partitions'], [[k, -1]])
                        else:
                            self.assertEqual(item['partitions'], [[-1, -1]])
                        item['max_folds'] = 1
                        sw = SecureWorker(item, Mock())
                        with patch.object(ModelingMachine.engine.tasks.base_modeler.BaseModeler,'_train') as train_mock:
                            with patch('ModelingMachine.engine.tasks.base_modeler.inject') as inject_mock:
                                train_mock.return_value = None
                                lid = sw.run()
                        xt = train_mock.call_args[0][0]
                        self.assertLess(abs(xt.shape[0]-(total_size * ss/100.0)),2)
                        k += 1
                    if ss <= (reps-1.0)/reps*(100-holdout_pct):
                        self.assertEqual(k,reps)
                    else:
                        self.assertEqual(k,1)

    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    def test_nested_CV(self,*args):
        dataset = self.datasets[0]
        reps = 5
        for gs_options in ('t_n=3;t_s=0','t_f=0.25;t_s=0','t_n=3;t_s=1','t_f=0.25;t_s=1'):
            for holdout_pct in (0,15,30):
                total_size = 200
                pid = ModelingMachine.engine.mocks.ProjectService.create_project(self.uid)
                aim_request = {'uid': str(self.uid), 'pid': str(pid),
                               'target': 'SeriousDlqin2yrs','metric':'RMSE',
                               'folds': 5, 'reps':reps,
                               'holdout_pct':holdout_pct, 'mode': 0}
                pid = self.create_project(dataset,aim_request)
                dataset_id = self.tempstore.read(keyname='wsready', index=str(pid), result=[])[0]
                q = QueueServiceBase(str(pid), Mock(), uid=self.uid)
                self.purge_queue(pid, purge_all=True)
                for ss in (44,66,80,90,100):
                    request = copy.copy(self.request)
                    request['dataset_id'] = dataset_id
                    request['pid'] = pid
                    request['samplepct'] = ss
                    if ss > (reps-1.0)/reps*(100-holdout_pct):
                        request['partitions'] = [(-1,-1)]
                    else:
                        request['max_reps'] = reps
                    self.persistent.destroy(condition={'pid':ObjectId(pid)},table='leaderboard')
                    q.put([request])
                    k = 0
                    while True:
                        item = self.next_job_from_queue(pid)
                        if not item:
                            break
                        logger.debug("JOB %s %s %s %s" % (gs_options,holdout_pct,ss,k))
                        self.assertEqual(item['samplepct'], ss)
                        self.assertEqual(item['total_size'], 200.0)
                        if ss <= (reps-1.0)/reps*(100-holdout_pct):
                            self.assertEqual(item['partitions'], [[k, -1]])
                        else:
                            self.assertEqual(item['partitions'], [[-1, -1]])
                        item['max_folds'] = 1
                        item['blueprint'] = {'1': (['NUM'], ['NI'], 'T'), '2': (['1'], ['GLMB p=[1.2,1.3];%s' % gs_options], 'P')}
                        sw = SecureWorker(item, Mock())
                        with patch.object(ModelingMachine.engine.tasks.base_modeler.BaseModeler,'_grid_search') as gs_mock:
                            with patch('ModelingMachine.engine.tasks.base_modeler.inject') as imock:
                                imock.return_value = {}
                                gs_mock.return_value = None
                                lid = sw.run()
                        gs_train = gs_mock.call_args[0][2]
                        cv = gs_mock.call_args[0][5]
                        # make sure stratified GS arg works
                        if 't_s=1' in gs_options:
                            self.assertEqual(cv.__class__.__name__,'StratifiedCV')
                        else:
                            self.assertEqual(cv.__class__.__name__,'RandomCV')
                        l = 0
                        # check folds arg works
                        for train,test in cv:
                            train_size = total_size * ss/100.0
                            if 't_f' in gs_options:
                                gs_train_size = train_size * 0.75
                            else:
                                gs_train_size = train_size * (3-1) / 3
                            logger.debug('SHAPE train %s gs_train %s est %s' % (gs_train.shape[0],train.shape[0],gs_train_size))
                            self.assertLess(abs(gs_train_size-train.shape[0]),3)
                            l += 1
                        if 't_n=3' in gs_options:
                            self.assertEqual(l,3)
                        else:
                            self.assertEqual(l,1)
                        k += 1
                    if ss <= (reps-1.0)/reps*(100-holdout_pct):
                        self.assertEqual(k,reps)
                    else:
                        self.assertEqual(k,1)


    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch('ModelingMachine.engine.secure_worker.Progress')
    @patch('ModelingMachine.engine.vertex_factory.time.sleep')
    def test_GroupCV_holdout(self,*args,**kwargs):
        ''' make sure partition keys in holdout don't overlap training keys '''
        # set up a project
        dataset = self.datasets[0]
        aim_request = {'uid': str(self.uid), 'pid': self.pid,
                       'target': 'SeriousDlqin2yrs','metric':'RMSE',
                       'cv_method': 'GroupCV', 'partition_key_cols': ['age'],
                       'folds': 5, 'reps':5, 'holdout_pct':20,
                       'mode': 0}
        pid = self.create_project(dataset, aim_request=aim_request)
        uid = '532329fcbb51f7015cf64c96'
        dataset_id = self.tempstore.read(keyname='wsready', index=str(pid), result=[])[0]
        self.persistent.update(table='project',condition={'_id':self.pid},values={'version':1.1})

        # create request
        request = {
            'blueprint': {'1': (['NUM'], ['NI'], 'T'), '2': (['1'], ['GLMB'], 'P')},
            'lid': str(ObjectId()),
            'uid': uid,
            'pid': pid,
            'dataset_id': dataset_id,
            'blueprint_id': ObjectId(),
            'partitions': [[0,-1]],
            'samplepct': 64,
            'qid': 1,
            'command': 'fit',
            'total_size': 200.0,
            'icons': [0],
            'bp': 1,
            'new_lid': True,
            'max_folds': 0}

        data = RequestData(WorkerRequest(request))
        train_keys = set(data.datasets[dataset_id]['main']['age'])
        holdout_keys = set(data.datasets[dataset_id]['holdout']['age'])
        self.assertSetEqual(train_keys.intersection(holdout_keys),set())


if __name__ == '__main__':
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    unittest.main()
