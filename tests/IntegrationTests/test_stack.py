import pandas
import numpy as np
import unittest
import logging
import psutil
import os
import time
import pytest
import common
import math
from math import log
from mock import Mock, patch, DEFAULT
from bson.objectid import ObjectId
from tests.IntegrationTests.storage_test_base import StorageTestBase
from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss
from ModelingMachine.engine.tasks.GLM import GLM
from MMApp.entities.user import UserService
from MMApp.entities.project import ProjectService
from MMApp.entities.dataset import DatasetService
from ModelingMachine import worker
from ModelingMachine.engine.tasks.glm import GLMB
from ModelingMachine.engine.partition import NewPartition,Partition
from ModelingMachine.engine.mocks import DB_APIClient, DelayedStorageClient
from ModelingMachine.engine.secure_worker import SecureWorker
from common.io.dataset_reader import DatasetReader
from MMApp.entities.db_conn import DBConnections
from common.entities.dataset import DatasetServiceBase
from common.wrappers import database
import common.io as io
import common.services.eda
from config.engine import EngConfig
from config.test_config import db_config
config = db_config
import ModelingMachine
import uuid

from common.services.flippers import FLIPPERS

logger = logging.getLogger('datarobot')
logger.setLevel(logging.INFO)
stack = None
class fakeGLM(GLMB):
    def fit(self,*args,**kwargs):
        f = super(fakeGLM,self).fit(*args,**kwargs)
        global stack
        stack = f.pred_stack
        return f
    def predict(self,*args,**kwargs):
        return super(fakeGLM,self).predict(*args,**kwargs)

def load_class(c):
    if c == 'ModelingMachine.engine.tasks.glm.GLMB':
        return fakeGLM
    else:
        return common.load_class(c)

class TestStack(StorageTestBase):
    @classmethod
    def setUpClass(cls):
        super(TestStack, cls).setUpClass()
        # Call to method in StorageTestBase
        cls.test_directory, cls.datasets = cls.create_test_files()

        cls.dbs = DBConnections()
        cls.get_collection = cls.dbs.get_collection
        cls.redis_conn = cls.dbs.get_redis_connection()
        cls.tempstore = database.new("tempstore")
        cls.persistent = database.new("persistent")

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

    @classmethod
    def tearDownClass(cls):
        super(TestStack, cls).tearDownClass()
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
        super(TestStack, self).setUp()
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

    def create_project(self,dataset):
        file_id = dataset['filename']
        upload_filename = 'projects/{}/raw/{}'.format(self.pid, file_id)
        if FLIPPERS.metadata_created_at_upload:
            filepath = os.path.join(self.test_directory, file_id)
            ds, controls = io.inspect_uploaded_file(filepath, file_id)
            ds_service = DatasetService(pid=self.pid, uid=self.uid,
                                        tempstore=self.tempstore,
                                        persistent=self.persistent)
            dataset_id = ds_service.create_metadata('universe', [upload_filename], controls, ds)
            p_service = ProjectService(pid=self.pid, uid=self.uid,
                                       tempstore=self.tempstore,
                                       persistent=self.persistent)
            p_service.init_project(upload_filename, file_id, dataset_id)

        request = {'originalName': file_id, 'dataset_id': upload_filename, 'uid':str(self.uid),
                   'pid' : str(self.pid), 'metric': dataset['metric']}
        self.worker.create_project(request)

        targetname = dataset['target'][0]
        request = {'uid': str(self.uid), 'pid': str(self.pid), 'target': targetname,'metric':'RMSE',
                   'folds': 5, 'reps':5, 'holdout_pct':20, 'mode': 0}
        self.redis_conn.hset('queue_settings:'+str(self.pid), 'mode', '0')
        response = self.worker.aim(request)
        self.persistent.update(table='project',condition={'_id':self.pid},values={'version':1.1})
        return str(self.pid)


    def _impute_na(self,df):
        for col in df.columns:
            m = df[col].median()
            mi = np.isnan(df[col].values)
            if np.any(mi):
                df[col+'-mi'] = mi.astype(int)
            if np.isnan(m):
                m=1
            df[col] = df[col].fillna(m)
        return df

    def stack_result(self,pid,ss):
        ds_service = DatasetServiceBase(pid=pid, uid=self.uid,
                                    persistent=self.persistent)
        metadata = ds_service.find_by_name('universe')
        reader = DatasetReader.from_record(metadata)
        if ss<=80:
            data = reader.get_data(partition='training', part_info={'holdout_pct':20})
        else:
            train = reader.get_data(partition='training', part_info={'holdout_pct':20})
            holdout = reader.get_data(partition='holdout', part_info={'holdout_pct':20})
            data = pandas.concat([train, holdout])
        X = self._impute_na(data)
        y = X.pop('SeriousDlqin2yrs')
        yt = y.values.astype(float)
        Z = NewPartition(X.shape[0],yt=yt,samplepct=ss,reps=1,total_size=200,cv_method='RandomCV')
        Z.set(samplepct=ss,max_reps=0,partitions=[(-1,-1)])
        rows = Z.T(r=-1,k=-1)
        xt = np.ascontiguousarray(X.values).astype(float)
        m = GLM()
        m.fit(xt[rows],yt[rows],distribution='Bernoulli',trace=True)
        result = m.predict(xt)
        cv = Z.get_cv(size=X.values[rows].shape[0], yt=yt[rows], folds=3, random_state=1234)
        for train,test in cv:
            xt = np.ascontiguousarray(X.values[rows][train]).astype(float)
            xv = np.ascontiguousarray(X.values[rows][test]).astype(float)
            yt = y.values[rows][train].astype(float)
            m = GLM()
            m.fit(xt,yt,distribution='Bernoulli',trace=True)
            result[np.array(rows)[test]] = m.predict(xv)
        return result

    def score1(self,pid,pred):
        ds_service = DatasetServiceBase(pid=pid, uid=self.uid,
                                    persistent=self.persistent)
        metadata = ds_service.find_by_name('universe')
        reader = DatasetReader.from_record(metadata)
        train = reader.get_data(partition='training', part_info={'holdout_pct':20})
        holdout = reader.get_data(partition='holdout', part_info={'holdout_pct':20})
        data = pandas.concat([train, holdout])
        X = self._impute_na(data)
        y = X.pop('SeriousDlqin2yrs')
        np.random.seed(0)
        randseq = np.random.permutation(X.shape[0])
        Z = NewPartition(math.ceil(X.shape[0]*0.8),sequence=randseq,samplepct=64,total_size=X.shape[0],cv_method='RandomCV')
        rows = Z.S(r=0,k=-1)
        return round(log_loss(y.values[rows],np.column_stack((1-pred[rows],pred[rows]))),5)

    def score5(self,pid,pred):
        ds_service = DatasetServiceBase(pid=pid, uid=self.uid,
                                    persistent=self.persistent)
        metadata = ds_service.find_by_name('universe')
        reader = DatasetReader.from_record(metadata)
        train = reader.get_data(partition='training', part_info={'holdout_pct':20})
        holdout = reader.get_data(partition='holdout', part_info={'holdout_pct':20})
        data = pandas.concat([train, holdout])
        X = self._impute_na(data)
        y = X.pop('SeriousDlqin2yrs')
        np.random.seed(0)
        randseq = np.random.permutation(X.shape[0])
        Z = NewPartition(math.ceil(X.shape[0]*0.8),sequence=randseq,samplepct=64,total_size=X.shape[0],cv_method='RandomCV')
        ya=[]
        yp=[]
        for p in range(5):
            rows = Z.S(r=p,k=-1)
            ya.extend(y.values[rows])
            yp.extend(pred[rows])
        yp=np.array(yp)
        return round(log_loss(ya,np.column_stack((1-yp,yp))),5)

    def scoreh(self,pid,pred):
        ds_service = DatasetServiceBase(pid=pid, uid=self.uid,
                                    persistent=self.persistent)
        metadata = ds_service.find_by_name('universe')
        reader = DatasetReader.from_record(metadata)
        data = reader.get_data(partition='holdout', part_info={'holdout_pct':20})
        X = self._impute_na(data)
        y = X.pop('SeriousDlqin2yrs')
        pred = pred[160:]
        return round(log_loss(y.values,np.column_stack((1-pred,pred))),5)


    @patch('ModelingMachine.engine.eda_multi._is_ref_id')
    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    def test_large_sample(self,arg0,arg1,arg2,rfmock):
        ''' check stacking and scoring of large sample size models '''

        # secure_worker uses all vars but the get_dataframe call
        # in stack_result() drops low_info vars, so to make the compare equal
        # we block the ref_id check
        rfmock.return_value = False

        dataset = self.datasets[0]
        pid = self.create_project(dataset)
        dataset_id = self.tempstore.read(keyname='wsready', index=str(pid), result=[])[0]
        #for ss in (66,80,90,100):
        for ss in (66,):
            request = {
                'max_reps': 1,
                'max_folds': 0,
                'blueprint': { '1': (['NUM'],['NI'],'T'), '2': (['1'],['GLMB t_rs=1234;t_s=0;pp_sf=3'],'P') },
                'command': 'fit',
                'partitions': [(-1,-1)],
                'samplepct': ss,
                'dataset_id': dataset_id,
                'qid': 1,
                'pid': pid,
                'new_lid': True,
                'uid': '53165bb357fc7b01bafddfb2',
                'lid': '53165bb357fc7b01bafddfb2',
            }
            with patch('ModelingMachine.engine.task_map.load_class',load_class):
                sw = SecureWorker(request, Mock())
                lid = sw.run()

            # make sure base_modeler stack matches manually calculated stack
            check = self.stack_result(pid,ss)
            np.testing.assert_almost_equal(stack[(-1,-1)],check)

            # check scores
            report = self.persistent.read({'_id':ObjectId(lid)}, table='leaderboard', result={})
            self.assertEqual(len(report['test']['Gini Norm']),2)
            # check first fold score
            score = self.score1(pid,stack[(-1,-1)])
            self.assertAlmostEqual(score,report['test']['LogLoss'][0])
            # check 5 fold score
            score = self.score5(pid,stack[(-1,-1)])
            self.assertAlmostEqual(score,report['test']['LogLoss'][1])
            if ss>80:
                # check holdout score
                score = self.scoreh(pid,stack[(-1,-1)])
                self.assertAlmostEqual(score,report['holdout']['LogLoss'])


if __name__ == '__main__':
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    unittest.main()
