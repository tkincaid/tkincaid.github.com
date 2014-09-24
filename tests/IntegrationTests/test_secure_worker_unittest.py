############################################################################
#
#       unit test for Secure Worker
#
#       Author: Tom DeGodoy
#
#       Copyright DataRobot, Inc. 2013
#
###########################################################################

import pandas
import numpy as np
import unittest
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
import pytest

from config.engine import EngConfig
from config.test_config import db_config
config = db_config

from ModelingMachine.engine.pandas_data_utils import getX, getY, varTypes, varTypeString
from ModelingMachine.engine.partition import Partition
from ModelingMachine import worker
from ModelingMachine.client import MMClient, SecureBrokerClient

from MMApp.entities.project import ProjectService
from predictionapi.entities.prediction import PredictionService
from predictionapi.entities.prediction import PredictionServiceError
from predictionapi.entities.prediction import PredictionServiceUserError
from predictionapi.prediction_io import PredictionResponse
from MMApp.entities.jobqueue import QueueService
from common.entities.job import BlenderRequest
from MMApp.entities.dataset import UploadedFile, DatasetService
from MMApp.entities.user import UserService
from MMApp.entities.db_conn import DBConnections
from MMApp.utilities.file_export import export_predictions
from common.services.queue_service_base import QueueServiceBase
from common.entities.job import blender_inputs
from common.services.flippers import FLIPPERS
from common.wrappers import database
from tests.IntegrationTests.storage_test_base import StorageTestBase
from tests.ModelingMachine.old_worker_report import old_worker_report, old_worker_pred
from MMApp.api import app as api_app
from MMApp.app import app as app_app
import common.services.eda
from common.services.model_refresh import refresh_model_by_lid
import common.io as io

from ModelingMachine.engine.metrics import logloss,logloss_w,rmse,rmse_w
from ModelingMachine.engine.mocks import DB_APIClient, DelayedStorageClient
from ModelingMachine.engine.monitor import FakeMonitor
from ModelingMachine.engine.secure_worker import SecureWorker
from ModelingMachine.engine.worker_request import WorkerRequest
from common.storage import FileStorageClient
from ModelingMachine.engine.vertex_factory import VertexFactory, VertexCache
from ModelingMachine.engine.worker_request import VertexDefinition, WorkerRequest
from ModelingMachine.engine.data_processor import RequestData
from ModelingMachine.engine.blueprint_interpreter import BlueprintInterpreter
import ModelingMachine
import MMApp
from common.engine import metrics
from common.services.flippers import FLIPPERS

logger = logging.getLogger("datarobot")

python_task = {
    '1234': {
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

R_task = {
    '1234': {
        'modeltype': 'R',
        'modelfit': '''
modelfit = function(response,data) {
    model = function(data) {
        prediction = rep(0.5,dim(data)[1]);
        return(prediction);
    };
    return(model);
};
                ''',
        'modelpredict': '''
modelpredict = function(model,data) {
    predictions = model(data);
    return(predictions);
};
                '''
    }
}

def sigmoid(a):
    return 1/(1+np.exp(-a))

def logit(a):
    n=np.array(a)
    return np.log(n/(1-n))

datasets = []
datasets.append({'filename':'credit-sample-200.csv','target':[u'SeriousDlqin2yrs','Binary'], 'metric': metrics.LOGLOSS})
datasets.append({'filename':'allstate-nonzero-200.csv','target':['Claim_Amount','Regression'], 'metric': metrics.GINI_NORM})
datasets.append({'filename':'kickcars-sample-200.csv','target':['IsBadBuy','Binary'], 'metric': metrics.LOGLOSS})
datasets.append({'filename':'credit-test-200.csv','target':[u'SeriousDlqin2yrs','Binary'], 'metric': metrics.LOGLOSS})
datasets.append({'filename':'credit-test-NA-200.csv','target':[u'SeriousDlqin2yrs','Binary'], 'metric': metrics.LOGLOSS})
datasets.append({'filename':'credit-train-small.csv','target':[u'SeriousDlqin2yrs','Binary'], 'metric': metrics.LOGLOSS})
datasets.append({'filename':'fastiron-sample-400.csv','target':[u'SalePrice','Regression'], 'metric': metrics.GAMMA_DEVIANCE})
datasets.append({'filename':'fastiron-train-sample-small.csv','target':[u'SalePrice','Regression'], 'metric': metrics.GAMMA_DEVIANCE})
datasets.append({'filename':'movielense_small_weights.csv','target':[u'rating','Regression'], 'metric': metrics.RMSE})
datasets.append({'filename':'credit-sample-200-noage.csv','target':[u'SeriousDlqin2yrs','Binary'], 'metric': metrics.LOGLOSS})


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
        cls.test_directory, cls.datasets = cls.create_test_files(datasets)

        cls.dbs = DBConnections()
        cls.get_collection = cls.dbs.get_collection
        cls.redis_conn = cls.dbs.get_redis_connection()
        cls.tempstore = database.new("tempstore", host=config['tempstore']['host'],
                port=config['tempstore']['port'])
        cls.persistent = database.new("persistent", host=config['persistent']['host'],
                port=config['persistent']['port'], dbname=config['persistent']['dbname'])

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
        super(TestMMWorker, cls).tearDownClass()
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
        super(TestMMWorker, self).setUp()
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
        children = psutil.Process(os.getpid()).children(recursive=True)
        if len(children)>0:
            for child in children:
                try:
                    os.kill(child.pid,signal.SIGUSR1)
                except:
                    continue
            time.sleep(0.5) #some time for process kill signals to work

    def create_project(self,dataset, weights=None, aim_keys=None):
        file_id = dataset['filename']
        upload_filename = 'projects/{}/raw/{}'.format(self.pid, file_id)
        if FLIPPERS.metadata_created_at_upload:
            filepath = os.path.join(self.test_directory, file_id)
            ds, controls = io.inspect_uploaded_file(filepath, file_id)
            ds_service = DatasetService(pid=self.pid, uid=self.uid,
                                        tempstore=self.tempstore,
                                        persistent=self.persistent)
            dataset_id = ds_service.create_metadata(
                'universe', [upload_filename], controls, ds)
            p_service = ProjectService(pid=self.pid, uid=self.uid,
                                       tempstore=self.tempstore,
                                       persistent=self.persistent)
            p_service.init_project(upload_filename, file_id, dataset_id)

        request = {'originalName': file_id, 'dataset_id': upload_filename, 'uid':str(self.uid),
                   'pid' : str(self.pid), 'metric': dataset['metric']}
        self.worker.create_project(request)

        roles = { str(self.uid) : [  'OWNER' ] }

        self.persistent.update(table='project', condition={'_id': self.pid}, values={'roles': roles})

        targetname = dataset['target'][0]
        request = {'uid': str(self.uid), 'pid': str(self.pid), 'target': targetname,
                   'folds': 5, 'reps':5, 'holdout_pct':20, 'mode': 0,
                   'metric': dataset['metric']}
        if weights:
            request['weights'] = weights
        if aim_keys:
            request.update(aim_keys)
        self.redis_conn.hset('queue_settings:'+str(self.pid), 'mode', '0')
        print worker.EngConfig['metablueprint_classname']
        response = self.worker.aim(request)
        return str(self.pid)

    def add_dataset(self, pid, filename):
        response = DatasetService().save_metadata(pid, UploadedFile(filename, pid=pid))
        dataset_id = response['upserted']
        return str(dataset_id)

    def make_predict_request(self, item):
        """ item = a leaderboard item """
        item = copy.deepcopy(item)
        filename = 'credit-test-200.csv'
        response = DatasetService().save_metadata(item['pid'], UploadedFile(filename, pid=item['pid']))
        dataset_id = response['upserted']
        args = {'predict':1, 'scoring_dataset_id': str(dataset_id)}
        item.update(args)
        QueueService(item['pid'], Mock(), item['uid']).put([item])
        return dataset_id

    def make_predict_request_with_json(self, item):
        """ make a prediction api request including json data """
        pid = str(item['pid'])
        try:
            lid = str(item['lid'])
        except:
            lid = str(item['_id'])
        filename = 'credit-test-200.csv'
        ds = pandas.read_csv(os.path.join(self.test_directory,filename))
        data = {'mimetype': 'application/json',
                'stream': ds.to_json(orient='records')}

        out = PredictionService(pid, self.uid).job_request(pid, lid, data)
        out['command'] = 'predict'
        out['result_id'] = str(ObjectId())
        return out

    def make_bad_predict_request_with_json(self, item):
        """ make a bad prediction api request including json data """
        pid = str(item['pid'])
        try:
            lid = str(item['lid'])
        except:
            lid = str(item['_id'])
        filename = 'credit-test-200.csv'
        ds = pandas.read_csv(os.path.join(self.test_directory,filename))
        data = {'mimetype': 'application/json',
                'stream': json.dumps({'bad': 'schema'})}
        out = PredictionService(pid, self.uid).job_request(pid, lid, data)
        out['command'] = 'predict'
        out['result_id'] = 'qewr'
        return out

    @patch.object(MMApp.api, 'authenticate_ide_request')
    def make_model_request_from_ide(self,item,*args,**kwargs):
        pid = str(item['pid'])
        try:
            lid = str(item['lid'])
        except:
            lid = str(item['_id'])
        modelfit = """function(response,data){model=function(data){prediction=rep(0.5,dim(data)[1]);return(prediction);};return(model);};"""
        modelpredict = """function(model,data){predictions=model(data);return(predictions);};"""
        data = {'pid':pid, 'uid':str(self.uid), 'samplepct':64, 'partitions':[[0,-1]],
                'key': '1', 'model_type': 'user model 1', 'max_folds':0, 'classname': 'bullshit',
                'modelfit':modelfit, 'modelpredict':modelpredict, 'modelsource':'', 'modeltype':'R'}

        url = '/queue'
        with api_app.test_client() as client:
            with client.session_transaction() as session:
                session['user'] = self.username
            response = client.post(url, content_type='application/json', data = json.dumps(data))
            self.assertEqual(response.status_code, 200)

    @patch.object(MMApp.app.UserService,'get_user_info')
    def run_with_lower_samplepct(self, item, ui_mock):
        ui_mock.return_value = {"uid": item['uid'],"username":"test@datarobot.com"}
        with app_app.test_client() as client:
            response = client.post('/project/'+str(item['pid'])+'/models', content_type='application/json', data=json.dumps([item]))
        self.assertEqual(response.status_code, 200)

    def make_blend_request(self, pid, models):
        qs = QueueServiceBase(pid,Mock())
        return qs.blend({
            'blender_method': 'GLM',
            'blender_items': models,
        })

    def next_job_from_queue(self,pid,last=False):
        if last:
            queue_query = self.redis_conn.rpop('queue:'+str(pid))
        else:
            queue_query = self.redis_conn.lpop('queue:'+str(pid))
        if not queue_query:
            return {}
        item = json.loads(queue_query)
        self.redis_conn.hmset('inprogress:'+str(pid),{item['qid']: queue_query})
        #del item['model_type']
        #del item['features']
        #del item['icons']
        #del item['bp']
        item['uid'] = str(self.uid)
        if 'command' in item:
            return item
        if item.get('predict'):
            item['command'] = 'predict'
        else:
            item['command'] = 'fit'
        return item

    def purge_queue(self,pid):
        self.redis_conn.delete('queue:'+str(pid))
        self.redis_conn.delete('inprogress:'+str(pid))

    @pytest.mark.skip('old worker deprecated')
    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch('ModelingMachine.engine.secure_worker.Progress')
    def xtest_secure_worker_vs_old_worker(self,*args,**kwargs):
        dataset = self.datasets[0]
        pid = self.create_project(dataset)
        job = self.next_job_from_queue(pid)

        item = {u'blueprint': {u'1': [[u'NUM'], [u'NI'], u'T'], u'2': [[u'1'], [u'GLMB'], u'P']},
                u'lid': u'53165bb63e0fd17dff7375dc', u'samplepct': 64, u'features': [u'Missing Values Imputed'],
                u'blueprint_id': u'd4c06a5c23cf1d917019720bceba32c8', u'total_size': 160.0, u'qid': 1, u'icons': [0],
                u'pid': u'53165bb33e0fd17dff7375d9', u'max_reps': 1, u'samplepct': 64, u'bp': 1,
                u'model_type': u'Generalized Linear Model (Bernoulli Distribution)', 'command': 'fit',
                u'dataset_id': u'53165bb33e0fd17dff7375da', 'uid': '53165bb357fc7b01bafddfb2', u'max_folds': 0,
                u'reference_model': True, u'new_lid': True}
        for key in ['pid','dataset_id','uid']:
            item[key] = job[key]

        report2 = old_worker_report
        pred2 = old_worker_pred

        sw = SecureWorker(item,Mock())
        lid1 = sw.run()
        report1 = self.persistent.read({'_id':ObjectId(lid1)}, table='leaderboard', result={})
        pred1 = self.persistent.read({'lid':ObjectId(lid1)}, table='predictions', result={})
        self.assertEqual(set(pred1.keys()),set(pred2.keys()))
        for key in pred2:
            if key in ['_id','lid','pid','dataset_id']:
                continue
            if isinstance(pred1[key],list):
                for i,j in zip(pred1[key],pred2[key]):
                    self.assertAlmostEqual(i,j)
            else:
                self.assertEqual(pred1[key],pred2[key],key)

        print set(report2.keys())-set(report1.keys())
        print set(report1.keys())-set(report2.keys())

        for key in ['test','holdout','extras','total_size','task_parameters','parts_label','parts','part_size','model_type']:
            self.assertEqual(report1[key],report2[key])


    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch('ModelingMachine.engine.secure_worker.Progress')
    @patch('ModelingMachine.engine.secure_worker.Monitor', FakeMonitor)
    @patch('ModelingMachine.engine.vertex_factory.time.sleep')
    def test_fit_with_missing_features(self,*args,**kwargs):
        dataset = self.datasets[0]
        pid = self.create_project(dataset)
        item = self.next_job_from_queue(pid)
        # credit_train_200 only has numeric features
        # the app should handle the empty CAT vertex
        item['blueprint'] = {
            '1': [['NUM'], ['NI'],'T'],
            '2': [['CAT'], ['DM2'],'T'],
            '3': [['1','2'], ['GLMB'],'P'],
        }
        sw = SecureWorker(item,Mock())
        lid1 = sw.run()
        report1 = self.persistent.read({'_id':ObjectId(lid1)}, table='leaderboard', result={})
        self.assertGreater(len(report1['test']['Gini']),0)

    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch('ModelingMachine.engine.secure_worker.Progress')
    @patch('ModelingMachine.engine.secure_worker.Monitor', FakeMonitor)
    @patch('ModelingMachine.engine.vertex_factory.time.sleep')
    def test_vertex_cache(self,*args,**kwargs):
        dataset = self.datasets[0]
        pid = self.create_project(dataset)
        item = self.next_job_from_queue(pid)

        sw = SecureWorker(item,Mock())
        lid1 = sw.run()
        report1 = self.persistent.read({'_id':ObjectId(lid1)}, table='leaderboard', result={})
        pred1 = self.persistent.read({'lid':ObjectId(lid1)}, table='predictions', result={})
        task_info1 = report1['task_info']['reps=1']
        for v in task_info1:
            for t in v:
                self.assertEqual(t['cached'],False)

        sw = SecureWorker(item,Mock())
        lid2 = sw.run()
        report2 = self.persistent.read({'_id':ObjectId(lid2)}, table='leaderboard', result={})
        task_info1 = report2['task_info']['reps=1']
        for v in task_info1:
            for t in v:
                self.assertEqual(t['cached'],True,"Error: task %s not cached" % (t['task_name']))


    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch('ModelingMachine.engine.secure_worker.Progress')
    @patch('ModelingMachine.engine.secure_worker.Monitor', FakeMonitor)
    @patch('ModelingMachine.engine.vertex_factory.time.sleep')
    def test_task_cache(self,*args,**kwargs):

        dataset = self.datasets[5]
        pid = self.create_project(dataset)
        item = self.next_job_from_queue(pid)

        sw = SecureWorker(item,Mock())
        lid1 = sw.run()
        report1 = self.persistent.read({'_id':ObjectId(lid1)}, table='leaderboard', result={})

        #empty queue
        self.purge_queue(pid)

        #check predictions on new data
        scoring_dataset_id = self.make_predict_request(report1)

        item = self.next_job_from_queue(pid)

        sw = SecureWorker(item,Mock())
        sw.run()

        predictions1 = self.persistent.read({'dataset_id':item['scoring_dataset_id'], 'lid':ObjectId(item['lid'])},
                    table='predictions', result={})

        self.assertEqual(len(predictions1['predicted-0']), 200)

        self.persistent.destroy({'dataset_id':item['scoring_dataset_id'], 'lid':ObjectId(item['lid'])},
                    table='predictions')

        #test with cold cache
        vertex_cache = VertexCache(item)
        item['vertex_cache'] = vertex_cache

        sw = SecureWorker(item,Mock())
        sw.run()

        predictions2 = self.persistent.read({'dataset_id':item['scoring_dataset_id'], 'lid':ObjectId(item['lid'])},
                    table='predictions', result={})

        self.assertEqual(len(predictions2['predicted-0']), 200)

        self.persistent.destroy({'dataset_id':item['scoring_dataset_id'], 'lid':ObjectId(item['lid'])},
                    table='predictions')

        #test with warm cache
        vertex_cache = sw.vertices
        item['vertex_cache'] = vertex_cache

        sw = SecureWorker(item,Mock())
        sw.run()

        predictions3 = self.persistent.read({'dataset_id':item['scoring_dataset_id'], 'lid':ObjectId(item['lid'])},
                    table='predictions', result={})

        self.assertEqual(len(predictions3['predicted-0']), 200)

        self.assertEqual(predictions1['predicted-0'], predictions2['predicted-0'])
        self.assertEqual(predictions1['predicted-0'], predictions3['predicted-0'])




    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch('ModelingMachine.engine.secure_worker.Progress')
    @patch('ModelingMachine.engine.secure_worker.Monitor', FakeMonitor)
    @patch('ModelingMachine.engine.vertex_factory.time.sleep')
    def test_fit_predict_drmodel(self,*args,**kwargs):

        dataset = self.datasets[0]
        pid = self.create_project(dataset)
        item = self.next_job_from_queue(pid)

        sw = SecureWorker(item,Mock())
        lid1 = sw.run()
        report1 = self.persistent.read({'_id':ObjectId(lid1)}, table='leaderboard', result={})
        pred1 = self.persistent.read({'lid':ObjectId(lid1)}, table='predictions', result={})

        inprogress = self.tempstore.read(keyname='inprogress', index=str(pid), result={})
        self.assertEqual(inprogress,{})

        item['new_lid'] = False

        #check ability to update the leaderboard for additional partitions
        for r in (1,2,4):
            item['lid']=str(lid1)
            item['partitions']=[[r,-1]]
            sw = SecureWorker(item,Mock())
            lid3 = sw.run()
        #use delay in file storage in order to simulate a race condition in final task coordination
        with patch('ModelingMachine.engine.vertex_factory.FileStorageClient', DelayedStorageClient) as mock:
            r = 3
            item['lid']=str(lid1)
            item['partitions']=[[r,-1]]
            sw = SecureWorker(item,Mock())
            lid3 = sw.run()

        report3 = self.persistent.read({'_id':ObjectId(lid3)}, table='leaderboard', result={})
        self.assertEqual(len(report3['test']['Gini']), 2) #verify the metric for the 5cv has been added
        self.assertIn('reps=4',report3['task_info']) #check that final partition info is saved

        # check prediction download
        pservice = ProjectService(uid=item['uid'])
        query = pservice.get_predictions(lid3, item['dataset_id'])
        export_file = os.path.join(self.test_directory,'pred.test')
        export_predictions(query, export_file, 64)
        pf = pandas.read_csv(export_file)
        self.assertEqual(pf.shape[0],200)
        #self.assertEqual(list(pf.columns),['Partitions','RowId','CV1','CV2','CV3','CV4','CV5','Stacked out of sample','Avg of CV','Full Model 80%'])
        self.assertEqual(list(pf.columns),['RowId','Cross-Validation Prediction'])
        self.assertTrue(np.all(pf['RowId']==sorted(pf['RowId'])))

        updated_keys = ['extras','partition_stats','resource_summary','finish_time','task_info','time','ec2','job_info']
        self.assertEqual(lid1,lid3)
        for key in report3:
            if key in ['roc','part_size','parts','test','lift','time_real','holdout']:
                continue
            if key not in updated_keys:
                # dict fields are updated with new partition info - make sure the old one is unchanged
                if isinstance(report1[key], dict):
                    self.assertDictContainsSubset(report1[key],report3[key], key)
                else:
                    self.assertEqual(report1[key],report3[key], key)
            elif key !='time':
                self.assertEqual(len(report3[key]), 5, key)

        for key in report1['test']:
            #verify metrics for 1st partitions haven't changed
            self.assertEqual(report1['test'][key][0], report3['test'][key][0])

        #empty queue
        self.purge_queue(pid)

        #blend some models
        self.make_blend_request(pid,[lid1,lid1])
        item = self.next_job_from_queue(pid)
        sw = SecureWorker(item,Mock())
        lid1b = sw.run()
        report1b = self.persistent.read({'_id':ObjectId(lid1b)}, table='leaderboard', result={})
        pred1b = self.persistent.read({'lid':ObjectId(lid1b)}, table='predictions', result={})
        # total_size does not include holout
        self.assertEqual(report1b['total_size']+40,len(pred1b['predicted-0']))
        blender_bp = report1b['blueprint_id']

        #check predictions on new data
        scoring_dataset_id = self.make_predict_request(report3)

        item = self.next_job_from_queue(pid)

        sw = SecureWorker(item,Mock())
        sw.run()

        predictions = self.persistent.read({'dataset_id':item['scoring_dataset_id'], 'lid':ObjectId(item['lid'])},
                    table='predictions', result={})

        self.assertEqual(len(predictions['predicted-0']), 200)
        self.assertEqual(len(predictions['row_index']), 200)
        self.assertEqual(predictions['newdata'], 'YES')

        metadata = self.persistent.read({'_id':ObjectId(item['scoring_dataset_id'])}, table='metadata', result={})
        self.assertEqual(metadata['computed'], [ObjectId(item['lid'])])

        #check predictions via api with json data
        item = self.make_predict_request_with_json(report3)
        sw = SecureWorker(item,Mock())
        out = sw.run()
        prediction = PredictionService(pid, self.uid, tempstore=self.tempstore)
        out = prediction._block_for_predictions(model_id=item['lid'], result_id=item['result_id'])
        self.assertIsInstance(out, PredictionResponse)

        #check predictions via api with json data and NULLs
        item = self.make_predict_request_with_json(report3)
        item['scoring_data']['stream'] = item['scoring_data']['stream'].replace(
            '"MonthlyIncome": [9141.0],','"MonthlyIncome": ["NULL"],')
        sw = SecureWorker(item,Mock())
        out = sw.run()
        prediction = PredictionService(pid, self.uid, tempstore=self.tempstore)
        out = prediction._block_for_predictions(model_id=item['lid'], result_id=item['result_id'])
        logger.info(out)
        self.assertIsInstance(out, PredictionResponse)

        #check blended model predictions on new data
        args = {'predict':1, 'scoring_dataset_id': str(scoring_dataset_id),'partitions':[(-1,-1)]}
        report1b.update(args)
        QueueService(item['pid'], Mock(), item['uid']).put([report1b])

        item = self.next_job_from_queue(pid)

        sw = SecureWorker(item,Mock())
        sw.run()

        pred2b = self.persistent.read({'dataset_id':item['scoring_dataset_id'], 'lid':ObjectId(item['lid'])},
                    table='predictions', result={})
        report2b = self.persistent.read({'_id':ObjectId(item['lid'])}, table='leaderboard', result={})
        self.assertIn('predicted-0',pred2b)
        self.assertEqual(200,len(pred2b['predicted-0']))
        self.assertTrue(all(np.isfinite(pred2b['predicted-0'])))

        #
        # run blender with different sample sizes
        #

        # _create_partition should be called with these args:
        expected_parts = {
            44: {
                'size':160,
                'rpart': {u'total_size': 200, u'folds': 5, u'seed': 0, u'reps': 5, u'holdout_pct': 20},
                'max_folds': 1,
                'partitions': [(0,-1)]
            },
            66: {
                'size':160,
                'rpart': {u'total_size': 200, u'folds': 5, u'seed': 0, u'reps': 1, u'holdout_pct': 20},
                'max_folds': 1,
                'partitions': [(-1,-1)]
            },
            88: {
                'size':200,
                'rpart': {u'total_size': 200, u'folds': 5, u'seed': 0, u'reps': 1, u'holdout_pct': 20},
                'max_folds': 1,
                'partitions': [(-1,-1)]
            },
        }
        for ss in (44,66,88):
            newsize_request = {
                'pid': pid,
                'samplepct': ss,
                'blueprint_id': blender_bp,
                'bp': '12+16',
                'max_reps': 1,
                'dataset_id': item['dataset_id'],
            }
            QueueService(item['pid'], Mock(), item['uid']).put([newsize_request])
            item = self.next_job_from_queue(pid)
            item['max_folds']=1
            item['max_reps']=1
            item['command']='fit'
            sw = SecureWorker(item,Mock())
            with patch.object(ModelingMachine.engine.blueprint_interpreter.BuildData,'_create_partition') as cpmock:
                partition = Partition(size=expected_parts[ss]['size'], **expected_parts[ss]['rpart'])
                partition.set(
                    samplepct=ss,
                    max_folds=expected_parts[ss]['max_folds'],
                    partitions=expected_parts[ss]['partitions'])
                cpmock.return_value=partition
                lid1b = sw.run()
                # make sure _create_partition is called with correct args
                self.assertEqual(cpmock.call_args[0][1], expected_parts[ss]['size'])
                self.assertEqual(cpmock.call_args[0][0]['samplepct'], ss)
                self.assertEqual(cpmock.call_args[0][0]['max_folds'], expected_parts[ss]['max_folds'])
                self.assertEqual(cpmock.call_args[0][0]['partitions'], expected_parts[ss]['partitions'])
            report2c = self.persistent.read({'_id':ObjectId(lid1b)}, table='leaderboard', result={})
            pred2c = self.persistent.read({'lid':ObjectId(lid1b)}, table='predictions', result={})
            self.assertEqual(report2c['samplepct'],ss)
            if ss>64:
                self.assertEqual(200,len(pred2c['predicted--1']))
                self.assertTrue(all(np.isfinite(pred2c['predicted--1'])))
            else:
                self.assertEqual(200,len(pred2c['predicted-0']))
                self.assertTrue(all(np.isfinite(pred2c['predicted-0'])))

    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch('ModelingMachine.engine.secure_worker.Progress')
    @patch('ModelingMachine.engine.secure_worker.Monitor', FakeMonitor)
    @patch('ModelingMachine.engine.vertex_factory.time.sleep')
    def test_model_blending(self, *args, **kwargs):
        dataset = self.datasets[0]
        pid = self.create_project(dataset)
        uid = ObjectId()

        # run model with small sample size
        item = self.next_job_from_queue(pid)
        item['samplepct'] = 54
        ls_item=copy.deepcopy(item)
        sw = SecureWorker(item,Mock())
        lid1 = sw.run()
        self.assertIsNotNone(lid1)
        report1 = self.persistent.read({'_id':ObjectId(lid1)}, table='leaderboard', result={})
        pred1 = self.persistent.read({'lid':ObjectId(lid1)}, table='predictions', result={})

        #empty queue
        self.purge_queue(pid)

        # run model with large sample size
        ls_item['samplepct']=89
        ls_item['partitions']=[[-1,-1]]
        ls_item['lid']='53165bb357fc7b01bafddfa4'
        ls_item['new_lid'] = True
        sw = SecureWorker(ls_item, Mock())
        lid1ls = sw.run()
        self.assertIsNotNone(lid1ls)
        pred1ls = self.persistent.read({'lid':ObjectId(lid1ls)}, table='predictions', result={})
        report1ls = self.persistent.read({'_id':ObjectId(lid1ls)}, table='leaderboard', result={})
        self.assertEqual(report1ls['samplepct'], 89)

        # blend models with small and large sample sizes
        blender_request = BlenderRequest.create_blender_request(
            'AVG',
            [lid1,lid1ls],
            'Binary',
            'LogLoss')
        logger.info('blender_request: %r', blender_request)
        QueueService(pid, Mock(), uid).blend(blender_request)
        item = self.next_job_from_queue(pid)
        sw = SecureWorker(item,Mock())
        lid_bl = sw.run()
        pred1bl = self.persistent.read({'lid':ObjectId(lid_bl)}, table='predictions', result={})
        report1bl = self.persistent.read({'_id':ObjectId(lid_bl)}, table='leaderboard', result={})
        self.assertEqual(report1bl['samplepct'], 89)

        # blend models with small and small sample sizes
        # Add the other partitions to the q so that it calculates correctly.
        ls5cv_item = copy.deepcopy(ls_item)
        ls5cv_item['lid'] = str(lid1)
        ls5cv_item['new_lid'] = False
        ls5cv_item['partitions'] =[[i, -1] for i in range(1,5)]
        ls5cv_item['samplepct'] = 54
        QueueService(pid, Mock(), uid).put([ls5cv_item])

        # re-run small size model with one partition so we can check
        # that it matches how blueprint_interpreter behaves for mixed size blenders
        ls_item['samplepct'] = 54  # FIXME was 54
        ls_item['partitions'] = [[-1, -1]]
        ls_item['lid']='53165bb357fc7b01bafddfa5'
        ls_item['new_lid'] = True
        sw = SecureWorker(ls_item,Mock())
        lid1op = sw.run()
        pred1op = self.persistent.read({'lid': ObjectId(lid1op)}, table='predictions', result={})
        report1op = self.persistent.read({'_id': ObjectId(lid1op)}, table='leaderboard', result={})
        self.assertEqual(report1op['samplepct'], ls_item['samplepct'])

        # make sure blender output matches inputs - check if manual AVG blend matches the blender
        def _compare_blender(pred_blend, pred_a, pred_b, blend_field='predicted--1', ab_field='Full Model 80%'):
            desired = sigmoid(np.mean((logit(pred_a[ab_field]), logit(pred_b[ab_field])), axis=0))
            actual = np.array(pred_blend[blend_field])
            mismatch_mask = desired != actual
            self.assertEqual(actual.shape, desired.shape)
            self.assertEqual(pred_a['row_index'], pred_b['row_index'])
            self.assertEqual(pred_blend['row_index'], pred_b['row_index'])
            self.assertEqual(pred_a['actual'], pred_b['actual'])
            self.assertEqual(pred_blend['actual'], pred_b['actual'])
            np.testing.assert_array_almost_equal(actual, desired)

        _compare_blender(pred1bl, pred1op, pred1ls)

        for i in range(4):
            item = self.next_job_from_queue(pid)
            sw = SecureWorker(item,Mock())
            lid_bl = sw.run()
        blender_request = BlenderRequest.create_blender_request(
            'AVG',
            [lid1,lid1],
            'Binary',
            'LogLoss')
        print "Blender request: %r"% blender_request
        QueueService(pid, Mock(), uid).blend(blender_request)

        item = self.next_job_from_queue(pid)
        sw = SecureWorker(item,Mock())
        lid_bl = sw.run()
        pred1bl = self.persistent.read({'lid':ObjectId(lid_bl)}, table='predictions', result={})
        report1bl = self.persistent.read({'_id':ObjectId(lid_bl)}, table='leaderboard', result={})
        self.assertEqual(report1bl['samplepct'],54)
        self.purge_queue(pid)

        _compare_blender(pred1bl, pred1, pred1, blend_field='predicted-0')

        # blend models with large and large sample sizes
        blender_request = BlenderRequest.create_blender_request(
            'AVG',
            [lid1ls,lid1ls],
            'Binary',
            'LogLoss')
        QueueService(pid, Mock(), uid).blend(blender_request)
        item = self.next_job_from_queue(pid)
        sw = SecureWorker(item,Mock())
        lid_bl = sw.run()
        pred1bl = self.persistent.read({'lid':ObjectId(lid_bl)}, table='predictions', result={})
        report1bl = self.persistent.read({'_id':ObjectId(lid_bl)}, table='leaderboard', result={})
        self.assertEqual(report1bl['samplepct'],89)

        _compare_blender(pred1bl, pred1ls, pred1ls, blend_field='predicted--1')

    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch('ModelingMachine.engine.secure_worker.Progress')
    @patch('ModelingMachine.engine.secure_worker.Monitor', FakeMonitor)
    def test_user_model_from_ide(self,*args,**kwargs):

        dataset = self.datasets[0]
        pid = self.create_project(dataset)
        item = self.next_job_from_queue(pid)
        lid0 = SecureWorker(item, Mock()).run()
        self.purge_queue(pid)

        self.make_model_request_from_ide(item)

        item = self.next_job_from_queue(pid)
        item['partition'] = {'holdout_pct':20}

        self.assertTrue('key' not in item)

        sw = SecureWorker(item,Mock())
        lid1 = sw.run()
        report1 = self.persistent.read({'_id':ObjectId(lid1)}, table='leaderboard', result={})
        pred1 = self.persistent.read({'lid':ObjectId(lid1)}, table='predictions', result={})

        self.assertTrue(report1)
        self.assertTrue(pred1)

        q1 = QueueService(item['pid'], Mock(), item['uid']).get()

        #check that metablueprint works with a user model in leaderboard:
        self.worker.next_steps({'pid':str(pid),'uid':str(self.uid)})

        q2 = QueueService(item['pid'], Mock(), item['uid']).get()

        self.assertGreater(len(q2),len(q1)) #MB must have added something to the queue

        self.purge_queue(item['pid'])
        item['samplepct'] = 50
        self.run_with_lower_samplepct(item)
        item = self.next_job_from_queue(pid)
        item['max_folds'] = 1
        sw = SecureWorker(item,Mock())
        lid1 = sw.run()
        report1 = self.persistent.read({'_id':ObjectId(lid1)}, table='leaderboard', result={})
        self.assertEqual(report1['samplepct'],50)
        self.assertTrue(isinstance(report1['test']['Gini'][0],float))


    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch('ModelingMachine.engine.secure_worker.Monitor', FakeMonitor)
    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch.object(ModelingMachine.engine.data_processor.DataProcessor,'user_tasks',python_task)
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    def test_python_user_model(self,mock1,mock2,mock3):
        dataset = self.datasets[0]
        pid = self.create_project(dataset)
        dataset_id = self.tempstore.read(keyname='wsready', index=str(pid), result=[])[0]
        request = {
            'blueprint': { '1': (['ALL'],['USERTASK id=1234'],'P') },
            'command': 'fit',
            'partitions': [(0,-1)],
            'samplepct': 50,
            'dataset_id': dataset_id,
            'qid': 1,
            'pid': pid,
            'uid': '53165bb357fc7b01bafddfb2',
            'new_lid': 'new',
            'lid': '53165bb357fc7b01bafddfb2',
            'max_folds': 1,
        }
        sw = SecureWorker(request, Mock())
        lid = sw.run()
        report = self.persistent.read({'_id':ObjectId(lid)}, table='leaderboard', result={})
        self.assertIsInstance(report['test']['Gini Norm'][0],float)
        vf = VertexFactory(python_task)
        vertex = vf.get(VertexDefinition(WorkerRequest(request),1))
        self.assertTrue(os.path.isfile(vertex['stored_files'][(0,-1)]))

    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch('ModelingMachine.engine.secure_worker.Monitor', FakeMonitor)
    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    @patch.object(ModelingMachine.engine.data_processor.DataProcessor,'user_tasks',R_task)
    def test_R_user_model(self,mock1,mock1a,mock2):
        dataset = self.datasets[0]
        pid = self.create_project(dataset)
        dataset_id = self.tempstore.read(keyname='wsready', index=str(pid), result=[])[0]
        request = {
            'blueprint': { '1': (['ALL'],['USERTASK id=1234'],'P') },
            'command': 'fit',
            'partitions': [(0,-1)],
            'samplepct': 50,
            'dataset_id': dataset_id,
            'qid': 1,
            'pid': pid,
            'uid': '53165bb357fc7b01bafddfb2',
            'new_lid': 'new',
            'lid': '53165bb357fc7b01bafddfb2',
            'max_folds': 1,
        }
        sw = SecureWorker(request, Mock())
        lid = sw.run()
        report = self.persistent.read({'_id':ObjectId(lid)}, table='leaderboard', result={})
        self.assertIsInstance(report['test']['Gini Norm'][0],float)
        vf = VertexFactory(R_task)
        vertex = vf.get(VertexDefinition(WorkerRequest(request),1))
        self.assertTrue(os.path.isfile(vertex['stored_files'][(0,-1)]))

    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch('ModelingMachine.engine.secure_worker.Monitor', FakeMonitor)
    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    def test_no_task_output(self,mock1,mock2,mock3):
        dataset = self.datasets[1]
        pid = self.create_project(dataset)
        dataset_id = self.tempstore.read(keyname='wsready', index=str(pid), result=[])[0]
        request = {
            'blueprint': { '1': (['CAT'],['DM2 cm=0'],'T'), '2': (['1'],['GLMR'],'P') },
            'command': 'fit',
            'partitions': [(0,-1)],
            'samplepct': 50,
            'dataset_id': dataset_id,
            'qid': 1,
            'pid': pid,
            'uid': '53165bb357fc7b01bafddfb2',
            'new_lid': 'new',
            'lid': '53165bb357fc7b01bafddfb2',
            'max_folds': 1,
        }
        sw = SecureWorker(request, Mock())
        lid = sw.run()
        report = self.persistent.read({'_id':ObjectId(lid)}, table='leaderboard', result={})
        self.assertIsInstance(report['test']['Gini Norm'][0],float)
        vf = VertexFactory(request['blueprint'])
        vertex = vf.get(VertexDefinition(WorkerRequest(request),1))
        self.assertTrue(os.path.isfile(vertex['stored_files'][(0,-1)]))

    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch('ModelingMachine.engine.secure_worker.Progress')
    @patch('ModelingMachine.engine.secure_worker.Monitor', FakeMonitor)
    @patch('ModelingMachine.engine.vertex_factory.time.sleep')
    def test_bad_prediction_api_request(self,*args,**kwargs):
        dataset = self.datasets[0]
        pid = self.create_project(dataset)
        item = self.next_job_from_queue(pid)

        sw = SecureWorker(item,Mock())
        lid = sw.run()
        inprogress = self.tempstore.read(keyname='inprogress', index=str(pid), result={})
        self.assertEqual(inprogress,{})
        report = self.persistent.read({'_id':ObjectId(lid)}, table='leaderboard', result={})
        for attr in ['test', 'extras', 'lift', 'features', 'task_info', 'grid_scores',
                     'resource_summary', 'ec2', 'roc', 'time', 'holdout']:
            if attr in report:
                del report[attr]
        logger.info('----------------------')
        logger.info(report)
        item = self.make_bad_predict_request_with_json(report)
        sw = SecureWorker(item, Mock())
        out = sw.run()

        prediction = PredictionService(pid, self.uid, tempstore=self.tempstore)
        with self.assertRaises(PredictionServiceUserError) as e:
            prediction._block_for_predictions(model_id=item['lid'], result_id=item['result_id'])

        # Check if exception msg is correct
        self.assertRegexpMatches(str(e.exception), 'JSON uploads .* array of objects')

    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch('ModelingMachine.engine.secure_worker.Progress')
    @patch('ModelingMachine.engine.secure_worker.Monitor', FakeMonitor)
    @patch('ModelingMachine.engine.vertex_factory.time.sleep')
    def test_old_project_large_sample(self,*args,**kwargs):
        # set up a project
        dataset = self.datasets[0]
        pid = self.create_project(dataset)
        uid = '532329fcbb51f7015cf64c96'
        dataset_id = self.tempstore.read(keyname='wsready', index=str(pid), result=[])[0]
        self.purge_queue(pid)
        # simulate an old project
        part_info = self.persistent.read(table='project',condition={'_id':ObjectId(pid)},result={})
        part_info['partition'].pop('cv_method')
        part_info['version'] = 1
        part_info.pop('_id')
        self.persistent.update(table='project',condiction={'_id':ObjectId(pid)},values=part_info)
        # fit blueprint at 80% sample
        request = {
            'blueprint': {'1': (['NUM'], ['NI'], 'T'), '2': (['1'], ['GLMB'], 'P')},
            'lid': str(ObjectId()),
            'uid': uid,
            'pid': pid,
            'dataset_id': dataset_id,
            'blueprint_id': ObjectId(),
            'partitions': [[-1,-1]],
            'samplepct': 80,
            'qid': 1,
            'command': 'fit',
            'total_size': 200.0,
            'icons': [0],
            'bp': 1,
            'new_lid': True,
            'max_folds': 0}
        # Run job
        sw = SecureWorker(request,Mock())
        lid1 = sw.run()

        # check results
        report1 = self.persistent.read({'_id':ObjectId(lid1)}, table='leaderboard', result={})
        self.assertIn('test',report1)
        self.assertIsInstance(report1['test']['Gini'][0],float)

    @patch.dict(EngConfig, {'MODEL_REFRESH_API': True})
    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.Monitor', FakeMonitor)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch('ModelingMachine.engine.secure_worker.Progress')
    @patch('ModelingMachine.engine.secure_worker.Monitor', FakeMonitor)
    @patch('ModelingMachine.engine.vertex_factory.time.sleep')
    def test_refresh_drmodel(self,*args,**kwargs):

        dataset = self.datasets[6]
        pid = self.create_project(dataset)

        for i in range(2):
            item = self.next_job_from_queue(pid)
        print
        print item['blueprint']
        print

        sw = SecureWorker(item,Mock())
        lid1 = sw.run()
        report1 = self.persistent.read({'_id':ObjectId(lid1)}, table='leaderboard', result={})
        pred1 = self.persistent.read({'lid':ObjectId(lid1)}, table='predictions', result={})

        #refresh on new data
        self.purge_queue(pid)

        dataset_id = self.add_dataset(pid, self.datasets[7]['filename'])

        refresh_model_by_lid(pid, lid1, file_ids=[dataset_id])

        item = self.next_job_from_queue(pid)

        self.assertEqual(item['command'],'refresh')

        print
        print item
        print

        sw = SecureWorker(item,Mock())
        lid2 = sw.run()
        report2 = self.persistent.read({'_id':ObjectId(lid1)}, table='leaderboard', result={})
        pred2 = self.persistent.read({'lid':ObjectId(lid1)}, table='predictions', result={})

        self.assertEqual(lid1, lid2)
        self.assertEqual(pred1, pred2)
        for key in report1:
            self.assertEqual(report1[key], report2[key])

    @patch.dict(EngConfig, {'MODEL_REFRESH_API': True})
    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch('ModelingMachine.engine.secure_worker.Progress')
    @patch('ModelingMachine.engine.secure_worker.Monitor', FakeMonitor)
    @patch('ModelingMachine.engine.vertex_factory.time.sleep')
    def test_refresh_build(self,*args,**kwargs):

        dataset = self.datasets[6]
        pid = self.create_project(dataset)

        item = self.next_job_from_queue(pid)
        print
        print item['blueprint']
        print
        item['blueprint'] = {'1':[['NUM'],['NI'],'T'], '2':[['1'],['GBR md=[5,10,15]'],'P']}

        sw = SecureWorker(item,Mock())
        lid1 = sw.run()
        report1 = self.persistent.read({'_id':ObjectId(lid1)}, table='leaderboard', result={})
        pred1 = self.persistent.read({'lid':ObjectId(lid1)}, table='predictions', result={})

        #refresh on new data
        self.purge_queue(pid)

        dataset_id = self.add_dataset(pid, self.datasets[7]['filename'])

        refresh_model_by_lid(pid, lid1, file_ids=[dataset_id])

        item = self.next_job_from_queue(pid)

        self.assertEqual(item['command'],'refresh')

        print
        print item
        print

        request = WorkerRequest(item)

        data = RequestData(request)

        self.assertIn('main', data.target_vector)
        self.assertIsInstance(data.target_vector['main'], pandas.Series)
        self.assertEqual(len(data.target_vector['main']), 2000)

        self.assertIn(item['dataset_id'], data.datasets.keys())
        for value in data.datasets.values():
            self.assertIn('main', value)
            self.assertIsInstance(value['main'], pandas.DataFrame)
            self.assertEqual(value['main'].shape, (2000, 49))
            self.assertNotIn(data.target_name, value['main'].columns)
            self.assertEqual(len(value['vartypes']), 49)

        request.validate(data) #validate the request using the data loaded for the request

        vertices = VertexCache(request)

        model = BlueprintInterpreter(vertices, data, request)

        print '############ REFRESH ###############'
        model.refresh()

        print vertices

        for key,value in vertices.cache.items():
            print key
            print value
            task = value.steps[0][0]
            if hasattr(task, 'best_parameters'):
                print task.best_parameters
                out = task.best_parameters.values()

        print report1['best_parameters']
        expected = report1['best_parameters'].values()

        self.assertEqual(out, expected)

    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch('ModelingMachine.engine.secure_worker.Progress')
    @patch('ModelingMachine.engine.secure_worker.Monitor', FakeMonitor)
    @patch('ModelingMachine.engine.vertex_factory.time.sleep')
    def test_weights(self,*args,**kwargs):
        dataset = self.datasets[0]
        pid = self.create_project(dataset, {'weight':'NumberOfOpenCreditLinesAndLoans'})
        #pid = self.create_project(dataset, {'weight':'NumberOfOpenCreditLinesAndLoans'})

        item = self.next_job_from_queue(pid)
        item['command'] = 'fit'

        request = WorkerRequest(item)

        data = RequestData(request)

        #check weights in request_data
        self.assertTrue('weight' in data.weights)
        self.assertIsInstance(data.weights['weight']['main'], np.ndarray)
        self.assertIsInstance(data.weights['weight']['holdout'], np.ndarray)
        self.assertEqual(len(data.weights['weight']['main']), 160)
        self.assertEqual(len(data.weights['weight']['holdout']), 40)


    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.vertex_factory.time.sleep')
    @patch('ModelingMachine.engine.secure_worker.Progress')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.Monitor', FakeMonitor)
    def test_update_model(self,*args,**kwargs):
        # set up a project
        dataset = self.datasets[0]
        pid = self.create_project(dataset)
        uid = '532329fcbb51f7015cf64c96'
        dataset_id = self.tempstore.read(keyname='wsready', index=str(pid), result=[])[0]

        # test updating a model with and without a starting grid
        for gs_opt in ('',' p=[1.2,1.4,1.6]'):
            # create fit request
            request = {
                'blueprint': {'1': [['NUM'], ['NI'], 'T'], '2': [['1'], ['GLMT'+gs_opt], 'P']},
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
                'wid': 0,
                'new_lid': True,
                'max_folds': 0}
            # Run job
            logger.debug("Testing fit %s" % ('GLMT'+gs_opt,))
            sw = SecureWorker(request,Mock())
            lid1 = sw.run()

            # create update request
            request['new_lid'] = False
            request['new_grid'] = { '2': { 'method': 'Tom', 'grid' : { 'p' : '1.1,1.3' } } }
            # Run job
            logger.debug("Testing update %s" % ('GLMT'+gs_opt,))
            sw = SecureWorker(request,Mock())
            lid1 = sw.run()

            # check results
            report1 = self.persistent.read({'_id':ObjectId(lid1)}, table='leaderboard', result={})
            self.assertIn('grid_scores',report1)
            self.assertIn('(0, -1)',report1['grid_scores'])
            grid_scores = report1['grid_scores']['(0, -1)']
            logger.debug("GRID SCORES %s" % grid_scores)
            # make sure parameter is present
            self.assertEqual(grid_scores[0][0].keys(),['p'])
            # make sure at least one of the update values is present
            p_values=set([i[0]['p'] for i in grid_scores])
            self.assertGreater(len(set((1.1,1.3)) & p_values),0)
            if gs_opt:
                # make sure at least one of the original values is present
                self.assertGreater(len(set((1.2,1.4,1.6)) & p_values),0)
            # check we have scores
            scores = [i[1] for i in grid_scores]
            self.assertTrue(all([isinstance(i,float) for i in scores]))
            self.assertTrue(np.all(np.isfinite(scores)))
            #TODO: test update of an update

    @patch.object(common.services.project.ProjectServiceBase, 'assert_has_permission')
    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch('ModelingMachine.engine.secure_worker.Progress')
    @patch('ModelingMachine.engine.vertex_factory.time.sleep')
    def test_newdata_weights(self,*args):
        # set up a project with training data
        uid = '532329fcbb51f7015cf64c96'
        dataset = self.datasets[0]
        pid = self.create_project(dataset, weights={'weight':'age'})
        dataset_id = self.tempstore.read(keyname='wsready', index=str(pid), result=[])[0]
        pservice = ProjectService(pid=pid,uid=uid)

        # upload new data
        filename = dataset['filename']
        response = DatasetService().save_metadata(pid, UploadedFile(filename, pid=pid))
        new_dataset_id = response['upserted']

        # upload new data without weights
        filename = 'credit-sample-200-noage.csv'
        response = DatasetService().save_metadata(pid, UploadedFile(filename, pid=pid))
        new_dataset_id_no_w = response['upserted']

        request = {
            'blueprint': {'1': (['NUM'], ['NI'], 'T'), '2': (['1'], ['GLMR'], 'P')},
            'lid': str(ObjectId()),
            'uid': uid,
            'partitions': [[0,-1]],
            'samplepct': 64,
            'command': 'fit',
            'pid': pid,
            'dataset_id': dataset_id,
            'blueprint_id': ObjectId(),
            'qid': 1,
            'total_size': 100.0,
            'icons': [0],
            'bp': 1,
            'new_lid': True,
            'max_folds': 0}

        # test training data
        sw = SecureWorker(request,Mock())
        lid1 = sw.run()

        # get predticions from DB
        query = pservice.get_predictions(lid1, request['dataset_id'])

        # predict on new data with weight col
        request['command'] = 'predict_dataset_id'
        request['predict'] = 1
        request['scoring_dataset_id'] = str(new_dataset_id)
        sw = SecureWorker(request,Mock())
        lid1 = sw.run()['lid']

        # get predticions from DB
        query = pservice.get_predictions(lid1, request['scoring_dataset_id'])

        # predict on new data with no weight col
        request['scoring_dataset_id'] = str(new_dataset_id_no_w)
        sw = SecureWorker(request,Mock())
        lid1 = sw.run()['lid']

        # get predticions from DB
        query = pservice.get_predictions(lid1, request['scoring_dataset_id'])

    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch('ModelingMachine.engine.secure_worker.Progress')
    @patch('ModelingMachine.engine.vertex_factory.time.sleep')
    def test_weighted_metrics(self,*args,**kwargs):
        ''' make sure weighted metrics are calculated by secure worker and
            that they are correct '''
        # set up a project
        dataset = self.datasets[0]
        pid = self.create_project(dataset, weights={'weight':'age'})
        uid = '532329fcbb51f7015cf64c96'
        dataset_id = self.tempstore.read(keyname='wsready', index=str(pid), result=[])[0]

        # create request
        request = {
            'blueprint': {'1': (['NUM'], ['NI'], 'T'), '2': (['1'], ['GLMB'], 'P')},
            'lid': str(ObjectId()),
            'uid': uid,
            'pid': pid,
            'dataset_id': dataset_id,
            'blueprint_id': ObjectId(),
            'partitions': [[0,-1]],
            'weight': 'age',
            'samplepct': 64,
            'qid': 1,
            'command': 'fit',
            'total_size': 200.0,
            'icons': [0],
            'bp': 1,
            'new_lid': True,
            'max_folds': 0}

        # test 64%, 81% & 100% sample sizes
        for test_case in ({'samplepct':64,'partitions':[[0,-1]]},
                          {'samplepct':81,'partitions':[[-1,-1]]},
                          {'samplepct':100,'partitions':[[-1,-1]]}):
            request.update(test_case)
            data = RequestData(WorkerRequest(request))

            # Run 1st Fold
            sw = SecureWorker(request,Mock())
            lid1 = sw.run()

            # get leaderboard entry and predictions
            report1 = self.persistent.read({'_id':ObjectId(lid1)}, table='leaderboard', result={})
            pred1 = self.persistent.read({'lid':ObjectId(lid1)}, table='predictions', result={})

            # manually calculate expected scores for CV1
            if test_case['samplepct'] <= 64:
                rows_cv1 = np.array(pred1['partition']).astype(str)=='0.0'
                pred_cv1 = np.array(pred1['predicted-0'])[rows_cv1]
            else:
                pred_cv1 = np.array(pred1['Full Model 80%'])[rows_cv1]
            actual_cv1 = pandas.concat((data.target_vector['main'],data.target_vector['holdout']))
            actual_cv1 = actual_cv1.values[rows_cv1]
            weights_cv1 = pandas.concat((data.datasets[dataset_id]['main']['age'],data.datasets[dataset_id]['holdout']['age']))
            weights_cv1 = weights_cv1.values[rows_cv1]
            expected_cv1 = round(logloss(actual_cv1,pred_cv1),5)
            expected_cv1w = round(logloss_w(actual_cv1,pred_cv1,weights_cv1),5)

            # check manually calculated scores against the leaderboard
            self.assertEqual(expected_cv1, report1['test']['LogLoss'][0])
            self.assertEqual(expected_cv1w, report1['test']['Weighted LogLoss'][0])
            self.assertNotEqual(expected_cv1, expected_cv1w)

            # manually calculate expected scores for holdout
            if test_case['samplepct'] <= 64:
                rows_holdout = np.array(pred1['partition'])=='Holdout'
                pred_holdout = np.array(pred1['predicted-0'])[rows_holdout]
            else:
                pred_holdout = np.array(pred1['Full Model 80%'])[rows_holdout]
            actual_holdout = pandas.concat((data.target_vector['main'],data.target_vector['holdout']))
            actual_holdout = actual_holdout.values[rows_holdout]
            weights_holdout = pandas.concat((data.datasets[dataset_id]['main']['age'],data.datasets[dataset_id]['holdout']['age']))
            weights_holdout = weights_holdout.values[rows_holdout]
            expected_holdout = round(logloss(actual_holdout,pred_holdout,weights_holdout),5)
            expected_holdoutw = round(logloss_w(actual_holdout,pred_holdout,weights_holdout),5)

            # check manually calculated scores against the leaderboard
            self.assertEqual(expected_holdout, report1['holdout']['LogLoss'])
            self.assertEqual(expected_holdoutw, report1['holdout']['Weighted LogLoss'])
            self.assertNotEqual(expected_holdout, expected_holdoutw)


    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch('ModelingMachine.engine.secure_worker.Progress')
    @patch('ModelingMachine.engine.vertex_factory.time.sleep')
    def test_recommender_metrics(self,*args,**kwargs):
        ''' make sure weighted metrics for recommender projects are calculated
            by secure worker and that they are correct '''
        # set up a project
        dataset = self.datasets[8]
        pid = self.create_project(dataset,
                                  aim_keys={
                                    'cv_method': 'GroupCV',
                                    'recommender_item_id': 'item_id',
                                    'recommender_user_id': 'user_id',
                                    'is_recommender': True,
                                    'partition_key_cols': ['user_id']})
        uid = '532329fcbb51f7015cf64c96'
        dataset_id = self.tempstore.read(keyname='wsready', index=str(pid), result=[])[0]
        self.persistent.update(table='project',condition={'_id':self.pid},values={'version':1.1})

        # create request
        request = {
            'blueprint': {'1': (['NUM'], ['NI'], 'T'), '2': (['1'], ['GLMG'], 'P')},
            'lid': str(ObjectId()),
            'uid': uid,
            'pid': pid,
            'dataset_id': dataset_id,
            'blueprint_id': ObjectId(),
            'partitions': [[0,-1]],
            'recommender': 1,
            'samplepct': 64,
            'qid': 1,
            'command': 'fit',
            'total_size': 200.0,
            'icons': [0],
            'bp': 1,
            'new_lid': True,
            'max_folds': 0}

        # test 64%, 81% & 100% sample sizes
        for test_case in ({'samplepct':64,'partitions':[[0,-1]]},
                          {'samplepct':81,'partitions':[[-1,-1]]},
                          {'samplepct':100,'partitions':[[-1,-1]]}):
            request.update(test_case)
            data = RequestData(WorkerRequest(request))

            # Run 1st Fold
            sw = SecureWorker(request,Mock())
            lid1 = sw.run()

            # get leaderboard entry and predictions
            report1 = self.persistent.read({'_id':ObjectId(lid1)}, table='leaderboard', result={})
            pred1 = self.persistent.read({'lid':ObjectId(lid1)}, table='predictions', result={})

            # manually calculate expected scores for CV1
            if test_case['samplepct'] <= 64:
                rows_cv1 = np.logical_or(np.array(pred1['partition']).astype(str)=='0.0',np.array(pred1['partition']).astype(str)=='0')
                pred_cv1 = np.array(pred1['predicted-0'])[rows_cv1]
            else:
                pred_cv1 = np.array(pred1['Full Model 80%'])[rows_cv1]
            actual_cv1 = pandas.concat((data.target_vector['main'],data.target_vector['holdout']))
            actual_cv1 = actual_cv1.values[rows_cv1]
            weights_cv1 = None
            expected_cv1 = round(rmse(actual_cv1,pred_cv1),5)
            expected_cv1w = round(rmse_w(actual_cv1,pred_cv1,weights_cv1),5)

            # check manually calculated scores against the leaderboard
            self.assertEqual(expected_cv1, report1['test']['RMSE'][0])
            self.assertEqual(expected_cv1w, report1['test']['Weighted RMSE'][0])
            self.assertEqual(expected_cv1, expected_cv1w)

            # manually calculate expected scores for holdout
            if test_case['samplepct'] <= 64:
                rows_holdout = np.array(pred1['partition'])=='Holdout'
                pred_holdout = np.array(pred1['predicted-0'])[rows_holdout]
            else:
                pred_holdout = np.array(pred1['Full Model 80%'])[rows_holdout]
            actual_holdout = pandas.concat((data.target_vector['main'],data.target_vector['holdout']))
            actual_holdout = actual_holdout.values[rows_holdout]
            weights_holdout = None
            expected_holdout = round(rmse(actual_holdout,pred_holdout,weights_holdout),5)
            expected_holdoutw = round(rmse_w(actual_holdout,pred_holdout,weights_holdout),5)

            # check manually calculated scores against the leaderboard
            self.assertEqual(expected_holdout, report1['holdout']['RMSE'])
            self.assertEqual(expected_holdoutw, report1['holdout']['Weighted RMSE'])
            self.assertEqual(expected_holdout, expected_holdoutw)


    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch('ModelingMachine.engine.secure_worker.Progress')
    @patch('ModelingMachine.engine.vertex_factory.time.sleep')
    def test_weighted_recommender_metrics(self,*args,**kwargs):
        ''' make sure weighted metrics for recommender projects are calculated
            by secure worker and that they are correct '''
        # set up a project
        dataset = self.datasets[8]
        pid = self.create_project(dataset, weights={'weight':'weight'},
                                  aim_keys={
                                    'cv_method': 'GroupCV',
                                    'recommender_item_id': 'item_id',
                                    'recommender_user_id': 'user_id',
                                    'is_recommender': True,
                                    'partition_key_cols': ['user_id']})
        uid = '532329fcbb51f7015cf64c96'
        dataset_id = self.tempstore.read(keyname='wsready', index=str(pid), result=[])[0]
        self.persistent.update(table='project',condition={'_id':self.pid},values={'version':1.1})

        # create request
        request = {
            'blueprint': {'1': (['NUM'], ['NI'], 'T'), '2': (['1'], ['GLMG'], 'P')},
            'lid': str(ObjectId()),
            'uid': uid,
            'pid': pid,
            'dataset_id': dataset_id,
            'blueprint_id': ObjectId(),
            'partitions': [[0,-1]],
            'weight': 'weight',
            'recommender': 1,
            'samplepct': 64,
            'qid': 1,
            'command': 'fit',
            'total_size': 200.0,
            'icons': [0],
            'bp': 1,
            'new_lid': True,
            'max_folds': 0}

        # test 64%, 81% & 100% sample sizes
        for test_case in ({'samplepct':64,'partitions':[[0,-1]]},
                          {'samplepct':81,'partitions':[[-1,-1]]},
                          {'samplepct':100,'partitions':[[-1,-1]]}):
            request.update(test_case)
            data = RequestData(WorkerRequest(request))

            # Run 1st Fold
            sw = SecureWorker(request,Mock())
            lid1 = sw.run()

            # get leaderboard entry and predictions
            report1 = self.persistent.read({'_id':ObjectId(lid1)}, table='leaderboard', result={})
            pred1 = self.persistent.read({'lid':ObjectId(lid1)}, table='predictions', result={})

            # manually calculate expected scores for CV1
            if test_case['samplepct'] <= 64:
                rows_cv1 = np.logical_or(np.array(pred1['partition']).astype(str)=='0.0',np.array(pred1['partition']).astype(str)=='0')
                pred_cv1 = np.array(pred1['predicted-0'])[rows_cv1]
            else:
                pred_cv1 = np.array(pred1['Full Model 80%'])[rows_cv1]
            actual_cv1 = pandas.concat((data.target_vector['main'],data.target_vector['holdout']))
            actual_cv1 = actual_cv1.values[rows_cv1]
            weights_cv1 = pandas.concat((data.datasets[dataset_id]['main']['weight'],data.datasets[dataset_id]['holdout']['weight']))
            weights_cv1 = weights_cv1.values[rows_cv1]
            expected_cv1 = round(rmse(actual_cv1,pred_cv1),5)
            expected_cv1w = round(rmse_w(actual_cv1,pred_cv1,weights_cv1),5)

            # check manually calculated scores against the leaderboard
            self.assertEqual(expected_cv1, report1['test']['RMSE'][0])
            self.assertEqual(expected_cv1w, report1['test']['Weighted RMSE'][0])
            self.assertNotEqual(expected_cv1, expected_cv1w)

            # manually calculate expected scores for holdout
            if test_case['samplepct'] <= 64:
                rows_holdout = np.array(pred1['partition'])=='Holdout'
                pred_holdout = np.array(pred1['predicted-0'])[rows_holdout]
            else:
                pred_holdout = np.array(pred1['Full Model 80%'])[rows_holdout]
            actual_holdout = pandas.concat((data.target_vector['main'],data.target_vector['holdout']))
            actual_holdout = actual_holdout.values[rows_holdout]
            weights_holdout = pandas.concat((data.datasets[dataset_id]['main']['weight'],data.datasets[dataset_id]['holdout']['weight']))
            weights_holdout = weights_holdout.values[rows_holdout]
            expected_holdout = round(rmse(actual_holdout,pred_holdout,weights_holdout),5)
            expected_holdoutw = round(rmse_w(actual_holdout,pred_holdout,weights_holdout),5)

            # check manually calculated scores against the leaderboard
            self.assertEqual(expected_holdout, report1['holdout']['RMSE'])
            self.assertEqual(expected_holdoutw, report1['holdout']['Weighted RMSE'])
            self.assertNotEqual(expected_holdout, expected_holdoutw)


    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch('ModelingMachine.engine.secure_worker.Progress')
    @patch('ModelingMachine.engine.vertex_factory.time.sleep')
    def test_stack_transform(self,*args,**kwargs):
        ''' make sure stacked predictions are transformed correctly
            for blenders and prediction download
        '''
        # set up a project
        dataset = self.datasets[1]
        pid = self.create_project(dataset)
        uid = '532329fcbb51f7015cf64c96'
        dataset_id = self.tempstore.read(keyname='wsready', index=str(pid), result=[])[0]
        self.persistent.update(table='project',condition={'_id':self.pid},values={'version':1.1})

        # create request with transform and no calibrate
        request = {
            'blueprint': {'1': (['NUM'], ['NI'], 'T'), '2': (['1'], ['GLMG logy'], 'P')},
            'lid': str(ObjectId()),
            'uid': uid,
            'pid': pid,
            'dataset_id': dataset_id,
            'blueprint_id': ObjectId(),
            'partitions': [[0,-1]],
            'weight': 'weight',
            'recommender': 1,
            'samplepct': 64,
            'qid': 1,
            'command': 'fit',
            'total_size': 200.0,
            'icons': [0],
            'bp': 1,
            'new_lid': True,
            'max_folds': 0}

        # Run 1st model
        sw = SecureWorker(request,Mock())
        lid1 = sw.run()

        # create request with transform and calibrate
        request['blueprint'] = {
            '1': (['NUM'], ['NI'], 'T'),
            '2': (['1'], ['GLMG logy'], 'S'),
            '3': (['2'], ['CALIB'], 'P')}

        # Run 2nd model
        sw = SecureWorker(request,Mock())
        lid2 = sw.run()

        blender_request = BlenderRequest.create_blender_request(
            'AVG',
            [lid1,lid2],
            'Regression',
            'RMSE')
        QueueService(pid, Mock(), uid).blend(blender_request)
        request = self.next_job_from_queue(pid)

        # Run blender
        sw = SecureWorker(request,Mock())
        lid2 = sw.run()

        # check predictions
        pred1 = self.persistent.read({'lid':ObjectId(lid1)}, table='predictions', result={})
        self.assertTrue(np.all(np.isfinite(pred1['predicted-0'])))

    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch('ModelingMachine.engine.secure_worker.Progress')
    @patch('ModelingMachine.engine.vertex_factory.time.sleep')
    def test_weight_balance_side_effects(self, *args, **kwargs):
        ''' make sure weight balancing does not affect weighted metrics
            or predictions
        '''
        # set up a project
        dataset = self.datasets[0]
        weights = {'weight': 'age'}
        pid = self.create_project(dataset, weights=weights)
        uid = '532329fcbb51f7015cf64c96'
        dataset_id = self.tempstore.read(keyname='wsready', index=str(pid), result=[])[0]
        self.persistent.update(table='project',
                               condition={'_id': self.pid}, values={'version': 1.1})

        # create request
        request = {
            'lid': str(ObjectId()),
            'uid': uid,
            'pid': pid,
            'dataset_id': dataset_id,
            'blueprint_id': ObjectId(),
            'partitions': [[0, -1]],
            'weight': 'weight',
            'recommender': 1,
            'samplepct': 64,
            'qid': 1,
            'command': 'fit',
            'total_size': 200.0,
            'icons': [0],
            'bp': 1,
            'new_lid': True,
            'max_folds': 0}

        # run model with and without weight balance
        for task_opt in (0, 1):
            # create blueprint with model that doesn't use weights
            request['blueprint'] = {'1': (['NUM'], ['NI'], 'T'),
                                    '2': (['1'], ['RC wa_bw=%d' % task_opt], 'P')}

            # Run model
            sw = SecureWorker(request, Mock())
            lid1 = sw.run()

            # check score and predictions are the same with or without
            # the weight balance option
            pred1 = self.persistent.read({'lid': ObjectId(lid1)}, table='predictions', result={})
            report1 = self.persistent.read({'_id':ObjectId(lid1)}, table='leaderboard', result={})
            self.assertEqual(report1['test']['LogLoss'][0], 0.66169)
            self.assertEqual(report1['test']['Weighted LogLoss'][0], 0.65152)
            self.assertEqual(np.sum(pred1['predicted-0']), 76.5625)


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logfile = logging.FileHandler(__file__+'.log',mode='w')
    logger.addHandler(console)
    logger.addHandler(logfile)

    unittest.main()

