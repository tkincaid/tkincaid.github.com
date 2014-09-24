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
import shutil
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
from ModelingMachine.engine.workspace import Workspace
from ModelingMachine.client import MMClient, SecureBrokerClient
from ModelingMachine.engine.metrics import logloss1D,LOGLOSS,MAD
from common.storage import FileObject

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
import common.io as io

from ModelingMachine.engine.mocks import DB_APIClient, DelayedStorageClient
from ModelingMachine.engine.secure_worker import SecureWorker
from common.storage import FileStorageClient
from ModelingMachine.engine.vertex_factory import VertexFactory, VertexCache
from ModelingMachine.engine.worker_request import VertexDefinition, WorkerRequest
import ModelingMachine
import MMApp

from common.services.flippers import FLIPPERS

logger = logging.getLogger("datarobot")

def sigmoid(a):
    return 1/(1+np.exp(-a))

def logit(a):
    n=np.array(a)
    return np.log(n/(1-n))

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
        children = psutil.Process(os.getpid()).get_children(recursive=True)
        if len(children)>0:
            for child in children:
                try:
                    os.kill(child.pid,signal.SIGUSR1)
                except:
                    continue
            time.sleep(0.5) #some time for process kill signals to work

    def copy_test_files(self):
        datasets = self.get_default()

        self.clear_test_dir()

        os.mkdir(self.test_directory)

        for each in datasets:
            testdatafile = each['filename'] if isinstance(each, dict) else each
            fin = os.path.join(self.testdatadir, testdatafile)
            fout = os.path.join(self.test_directory, testdatafile)
            shutil.copy(fin,fout)
            testdatafile = 'projects/'+str(self.pid)+'/raw/' + testdatafile
            FileObject(testdatafile).put(fin)

        return self.test_directory, datasets

    def create_project(self,dataset,cv_options=None):
        if cv_options is None:
            cv_options = {
                'reps': 5,
                'holdout_pct': 20,
            }
        file_id = dataset['filename']
        self.copy_test_files()

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

        request = {'originalName': file_id, 'dataset_id': upload_filename,
                   'uid':str(self.uid), 'pid' : str(self.pid), 'metric': dataset['metric']}
        self.worker.create_project(request)

        roles = { str(self.uid) : [  'OWNER' ] }

        self.persistent.update(table='project', condition={'_id': self.pid}, values={'roles': roles})

        targetname = dataset['target'][0]
        request = {'uid': str(self.uid), 'pid': str(self.pid), 'target': targetname,
                   'folds': 5, 'mode': 0, 'metric': dataset['metric']}
        request.update(cv_options)
        self.redis_conn.hset('queue_settings:'+str(self.pid), 'mode', '0')
        response = self.worker.aim(request)
        self.persistent.update(table='project',condition={'_id':self.pid},values={'version':1.1})
        return str(self.pid)


    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch('ModelingMachine.engine.secure_worker.Progress')
    @patch('ModelingMachine.engine.vertex_factory.time.sleep')
    @pytest.mark.skip
    def test_download_predictions(self,*args,**kwargs):
        # set up a project
        # training data
        dataset = {'filename':'credit-train-small.csv','target':[u'SeriousDlqin2yrs','Binary'], 'metric': LOGLOSS}
        ds = pandas.read_csv(os.path.join(self.test_directory,dataset['filename']))
        target = dataset['target'][0]
        pid = self.create_project(dataset)
        uid = '532329fcbb51f7015cf64c96'
        dataset_id = self.tempstore.read(keyname='wsready', index=str(pid), result=[])[0]
        pservice = ProjectService(uid=uid)
        # new data
        filename = dataset['filename']
        response = DatasetService().save_metadata(pid, UploadedFile(filename, pid=pid))
        new_dataset_id = response['upserted']
        logger.debug("DATASET %s %s" % (dataset_id,new_dataset_id))

        # test with and without grid search
        for task_opt in ('',' p=[1.2,1.3];t_f=0.25;t_n=1'):
            # check different sample %
            for ss in (64,78,95):
                logger.debug("===== Testing %d%% Sample with %s option =====" % (ss,task_opt))
                # create request
                request = {
                    'blueprint': {'1': (['NUM'], ['NI'], 'T'), '2': (['1'], ['GLMB%s' % task_opt], 'P')},
                    'lid': str(ObjectId()),
                    'uid': uid,
                    'pid': pid,
                    'dataset_id': dataset_id,
                    'blueprint_id': ObjectId(),
                    'partitions': [[0,-1]],
                    'samplepct': ss,
                    'qid': 1,
                    'command': 'fit',
                    'total_size': 15000.0,
                    'icons': [0],
                    'bp': 1,
                    'new_lid': True,
                    'max_folds': 0}
                if ss > 64:
                    request['partitions'] = [[-1,-1]]

                # Run 1st Fold
                sw = SecureWorker(request,Mock())
                lid1 = sw.run()

                # get leaderboard entry
                report1 = self.persistent.read({'_id':ObjectId(lid1)}, table='leaderboard', result={})

                # get predticions from DB
                query = pservice.get_predictions(lid1, request['dataset_id'])
                # save predictions to file
                export_file = os.path.join(self.test_directory,'pred.test')
                export_predictions(query, export_file, ss)
                # load predictions from file
                pf = pandas.read_csv(export_file)

                # check file structure
                self.assertEqual(pf.shape[0],15000)
                if ss > 64:
                    self.assertEqual(list(pf.columns),['Partitions','RowId','Full Model 80%'])
                else:
                    self.assertEqual(list(pf.columns),['Partitions','RowId','CV1','Stacked out of sample','Avg of CV','Full Model 80%'])
                # check rows are sorted
                self.assertTrue(np.all(pf['RowId']==sorted(pf['RowId'])))
                # check partitions are random
                if ss<= 80:
                    self.assertLess(abs((pf['Partitions'][:7500]=='Holdout').sum()-1500),100)

                # check leaderboard scores
                if ss <= 64:
                    # make sure 1-fold CV score matches
                    non_holdout_pred = pf['CV1'][pf['Partitions'] == 'Fold-1']
                    y = ds[target][pf['Partitions'] == 'Fold-1']
                    self.assertAlmostEqual(logloss1D(y,non_holdout_pred),report1['test']['LogLoss'][0],5)
                    # make sure holdout score matches
                    holdout_pred = pf['CV1'][pf['Partitions'] == 'Holdout']
                    y = ds[target][pf['Partitions'] == 'Holdout']
                    self.assertAlmostEqual(logloss1D(y,holdout_pred),report1['holdout']['LogLoss'],5)
                # make sure stacked prediction is close to 1-fold score
                stacked_non_holdout_pred = pf['Full Model 80%'][pf['Partitions'] == 'Fold-1']
                y = ds[target][pf['Partitions'] == 'Fold-1']
                if ss <= 64:
                    # 64% sample should be about the same
                    self.assertAlmostEqual(logloss1D(y,stacked_non_holdout_pred),report1['test']['LogLoss'][0],3)
                    # larger samples should do a little better, but not too much better
                    self.assertLess(logloss1D(y,stacked_non_holdout_pred),report1['test']['LogLoss'][0])
                    self.assertLess(report1['test']['LogLoss'][0]-logloss1D(y,stacked_non_holdout_pred),0.02)

                logger.debug("===== Testing %d%% Sample with %s option 5 Fold =====" % (ss,task_opt))
                # run remaining folds for 64% sample
                if ss <= 64:
                    request['new_lid'] = False
                    for r in (1,2,3,4):
                        request['lid']=str(lid1)
                        request['partitions']=[[r,-1]]
                        sw = SecureWorker(request,Mock())
                        lid3 = sw.run()

                    # get leaderboard entry
                    report3 = self.persistent.read({'_id':ObjectId(lid3)}, table='leaderboard', result={})

                    # get predticions from DB
                    query = pservice.get_predictions(lid3, request['dataset_id'])
                    # save predictions to file
                    export_file = os.path.join(self.test_directory,'pred.test')
                    export_predictions(query, export_file, ss)
                    # load predictions from file
                    pf = pandas.read_csv(export_file)

                    # check file structure
                    self.assertEqual(pf.shape[0],15000)
                    self.assertEqual(list(pf.columns),['Partitions','RowId','CV1','CV2','CV3','CV4','CV5','Stacked out of sample','Avg of CV','Full Model 80%'])
                    self.assertTrue(np.all(pf['RowId']==sorted(pf['RowId'])))
                    # check partitions are random
                    if ss<= 80:
                        self.assertLess(abs((pf['Partitions'][:7500]=='Holdout').sum()-1500),100)

                    # get exported predictions
                    non_holdout_pred = pf['Stacked out of sample'][pf['Partitions'] != 'Holdout']
                    holdout_pred = pf['CV1'][pf['Partitions'] == 'Holdout']
                    stacked_holdout_pred = pf['Full Model 80%'][pf['Partitions'] == 'Holdout']
                    # make sure 5-fold CV score matches
                    y = ds[target][pf['Partitions'] != 'Holdout']
                    self.assertAlmostEqual(logloss1D(y,non_holdout_pred),report3['test']['LogLoss'][1],5)
                    # make sure holdout score matches
                    y = ds[target][pf['Partitions'] == 'Holdout']
                    self.assertAlmostEqual(logloss1D(y,holdout_pred),report3['holdout']['LogLoss'],5)
                    # make sure stacked prediction is close to holdout score
                    self.assertAlmostEqual(logloss1D(y,stacked_holdout_pred),report1['holdout']['LogLoss'],3)

                # Simulate Compute Prediction (train+holdout)
                request['command'] = 'predict_dataset_id'
                request['predict'] = 1
                request['scoring_dataset_id'] = request['dataset_id']

                logger.debug("===== Testing %d%% Sample with %s option holdout =====" % (ss,task_opt))
                # run prediction request
                if ss > 64:
                    parts = (-1,)
                else:
                    parts = (0,1,2,3,4)
                for r in parts:
                    request['lid']=str(lid1)
                    request['partitions']=[[r,-1]]
                    sw = SecureWorker(request,Mock())
                    sw.run()

                # get leaderboard entry
                report3 = self.persistent.read({'_id':ObjectId(request['lid'])}, table='leaderboard', result={})

                # get predticions from DB
                query = pservice.get_predictions(request['lid'], request['dataset_id'])
                # save predictions to file
                export_file = os.path.join(self.test_directory,'pred.test')
                export_predictions(query, export_file, ss)
                # load predictions from file
                pf = pandas.read_csv(export_file)

                # check file structure
                self.assertEqual(pf.shape[0],15000)
                if ss > 64:
                    self.assertEqual(list(pf.columns),['Partitions','RowId','Full Model 80%'])
                else:
                    self.assertEqual(list(pf.columns),['Partitions','RowId','CV1','CV2','CV3','CV4','CV5','Stacked out of sample','Avg of CV','Full Model 80%'])
                self.assertTrue(np.all(pf['RowId']==sorted(pf['RowId'])))
                # check partitions are random
                if ss<= 80:
                    self.assertLess(abs((pf['Partitions'][:7500]=='Holdout').sum()-1500),100)

                # check exported prediction scores
                if ss <= 64:
                    non_holdout_pred = pf['Stacked out of sample'][pf['Partitions'] != 'Holdout']
                    holdout_pred = pf['CV1'][pf['Partitions'] == 'Holdout']
                    # make sure 5-fold CV score matches
                    y = ds[target][pf['Partitions'] != 'Holdout']
                    self.assertAlmostEqual(logloss1D(y,non_holdout_pred),report3['test']['LogLoss'][1],5)
                    # make sure holdout score matches
                    y = ds[target][pf['Partitions'] == 'Holdout']
                    self.assertAlmostEqual(logloss1D(y,holdout_pred),report3['holdout']['LogLoss'],5)
                    # save holdout row mask since > 80% sample will need it
                    holdout_mask = pf['Partitions'] == 'Holdout'
                    # save partitions so we can check newdata
                    partitions = pf['Partitions']
                # make sure stacked prediction is close to holdout score
                stacked_holdout_pred = pf['Full Model 80%'][holdout_mask]
                y = ds[target][holdout_mask]
                self.assertAlmostEqual(logloss1D(y,stacked_holdout_pred),report1['holdout']['LogLoss'],3)

                # predict on new data
                request['command'] = 'predict_dataset_id'
                request['predict'] = 1
                request['scoring_dataset_id'] = str(new_dataset_id)

                logger.debug("===== Testing %d%% Sample with %s option new data =====" % (ss,task_opt))
                # run prediction request
                if ss > 64:
                    parts = (-1,)
                else:
                    parts = (0,1,2,3,4)
                for r in parts:
                    request['lid']=str(lid1)
                    request['partitions']=[[r,-1]]
                    sw = SecureWorker(request,Mock())
                    sw.run()

                # get leaderboard entry
                report3 = self.persistent.read({'_id':ObjectId(request['lid'])}, table='leaderboard', result={})

                # get predticions from DB
                query = pservice.get_predictions(request['lid'], request['scoring_dataset_id'])
                # save predictions to file
                export_file = os.path.join(self.test_directory,'pred.test')
                export_predictions(query, export_file, ss)
                # load predictions from file
                pf = pandas.read_csv(export_file)

                # check file structure
                self.assertEqual(pf.shape[0],15000)
                if ss > 64:
                    self.assertEqual(list(pf.columns),['Partitions','RowId','Full Model %s%%' % ss])
                else:
                    self.assertEqual(list(pf.columns),['Partitions','RowId','CV1','CV2','CV3','CV4','CV5','Stacked out of sample','Avg of CV'])
                self.assertTrue(np.all(pf['RowId']==sorted(pf['RowId'])))

                # check exported prediction scores
                if ss <= 64:
                    pred = pf['CV1']
                    # make sure stacked newdata scores match stacked training data scores
                    # since we uploaded the same dataset
                    non_holdout =np.logical_not(holdout_mask)
                    y = ds[target][non_holdout]
                    pred = np.empty(len(partitions))
                    for i in range(5):
                        mask = partitions.astype('object')==('Fold-%d' % (i+1))
                        pred[mask] = np.array(pf['CV%s' % (i+1)])[mask]
                    self.assertAlmostEqual(logloss1D(y,pred[non_holdout]),report3['test']['LogLoss'][1],5)
                else:
                    # since we uploaded the same file twice, the new data
                    # prediction score from non-partitioned training should be
                    # better than prdictions from cross validation
                    non_holdout =np.logical_not(holdout_mask)
                    pred = pf['Full Model %d%%' % ss][non_holdout]
                    y = ds[target][non_holdout]
                    self.assertGreater(report3['test']['LogLoss'][1]-logloss1D(y,pred),0.0001)

    def test_export_predictions_with_na(self):
        ''' make sure that the export can handle missing rows '''
        export_file = os.path.join(self.test_directory,'pred.test')
        query = {
            'row_index': [1,3,5],
            'Full Model 80%': [1,2,3,4,5,6]
        }
        export_predictions(query, export_file, 64)
        pf = pandas.read_csv(export_file)
        self.assertEqual(pf['Cross-Validation Prediction'].tolist(),[2,4,6])

    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch('ModelingMachine.engine.secure_worker.Progress')
    @patch('ModelingMachine.engine.vertex_factory.time.sleep')
    def test_download_with_transform(self,*args,**kwargs):
        # set up a project
        dataset = self.datasets[0]
        pid = self.create_project(dataset)
        uid = '532329fcbb51f7015cf64c96'
        dataset_id = self.tempstore.read(keyname='wsready', index=str(pid), result=[])[0]

        # create request with y transform
        request = {
            'blueprint': {'1': (['NUM'], ['NI'], 'T'), '2': (['1'], ['GLMB logy'], 'P')},
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

        # Run 1st Fold
        sw = SecureWorker(request,Mock())
        lid1 = sw.run()

        # get predticions from DB
        pservice = ProjectService(uid=uid)
        query = pservice.get_predictions(lid1, request['dataset_id'])

        # check stacked predictions are exp(y)-1
        partial_expected = np.array([1.3621779542328145, 0.22348889871324396, 0.5089503391552659,
            0.5034874108888467, 0.5869282642670017, 0.27691545890145486, 0.16877211745211884,
            0.24929503091778216, 0.015283116921862128, 0.08352678504135169, 0.2517937442846747,
            0.055906002939369426, 0.14415969609876922, 0.441829170191977, 0.13901871279946265,
            0.49668191753158375, 0.052837411863837946, 0.5538418834270704, 0.19939166100933514])
        np.testing.assert_almost_equal(np.array(query['Full Model 80%'][:19]),partial_expected)

    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch('ModelingMachine.engine.secure_worker.Progress')
    @patch('ModelingMachine.engine.vertex_factory.time.sleep')
    def test_classification_predictions(self,*args):
        for cv_options in (
                { 'cv_method': 'StratifiedCV', 'reps': 5, 'holdout_pct': 0, },
                { 'cv_method': 'StratifiedCV', 'reps': 5, 'holdout_pct': 20, },
                { 'cv_method': 'StratifiedCV', 'reps': 1, 'validation_pct': 0.15, 'holdout_pct': 0, },
                { 'cv_method': 'StratifiedCV', 'reps': 1, 'validation_pct': 0.15, 'holdout_pct': 20, },
                ):
            # set up a project with training data
            uid = '532329fcbb51f7015cf64c96'
            self.pid = ObjectId()
            dataset = self.datasets[8]
            pid = self.create_project(dataset,cv_options=cv_options)
            dataset_id = self.tempstore.read(keyname='wsready', index=str(pid), result=[])[0]
            pservice = ProjectService(pid=self.pid,uid=uid)
            ds = pandas.read_csv(os.path.join(self.test_directory,self.datasets[7]['filename']))

            # upload new data
            filename = dataset['filename']
            response = DatasetService().save_metadata(pid, UploadedFile(filename, pid=pid))
            new_dataset_id = response['upserted']

            holdout_pct = pservice.partition['holdout_pct']
            if 'validation_pct' in pservice.partition:
                max_sample = 100 * (1-pservice.partition['validation_pct']) - holdout_pct
            else:
                reps = pservice.partition['reps']
                max_sample = (100 - holdout_pct) * (reps - 1) / reps
            request = {
                'blueprint': {'1': (['NUM'], ['NI'], 'T'), '2': (['1'], ['GLMR'], 'P')},
                'lid': str(ObjectId()),
                'uid': uid,
                'pid': pid,
                'dataset_id': dataset_id,
                'blueprint_id': ObjectId(),
                'qid': 1,
                'total_size': 100.0,
                'icons': [0],
                'bp': 1,
                'new_lid': True,
                'max_folds': 0}

            # test with and without grid search
            for gs_options in ('',' p=[1.1,1.2]'):
                request['blueprint'] = {'1': (['NUM'], ['NI'], 'T'), '2': (['1'], ['GLMB%s' % gs_options], 'P')}
                # test small/medium/large sample sizes
                for ss in (max_sample-1,max_sample+1,99):
                    # predict on training data
                    logger.debug("Testing predicion sample %s gs %s CV %s" % (ss,gs_options,cv_options))
                    request['samplepct'] = ss
                    request['command'] = 'fit'
                    request.pop('scoring_dataset_id',1)
                    if ss>max_sample:
                        request['partitions'] = [[-1,-1]]
                    else:
                        request['partitions'] = [[0,-1]]
                    # Run 1st Fold
                    sw = SecureWorker(request,Mock())
                    lid1 = sw.run()

                    # get predticions from DB
                    query = pservice.get_predictions(lid1, request['dataset_id'])
                    # save predictions to file
                    export_file = os.path.join(self.test_directory,'pred.test')
                    export_predictions(query, export_file, max_sample)
                    # load predictions from file
                    pf = pandas.read_csv(export_file)
                    # predictions should be sequential and 2 * the predictor
                    logger.debug("PRED %s" % pf['Cross-Validation Prediction'].tolist())
                    self.assertTrue(all([pf['Cross-Validation Prediction'][i]<=pf['Cross-Validation Prediction'][i+1] for i in range(pf.shape[0]-1)]))
                    np.allclose(pf['Cross-Validation Prediction'],ds['c'],atol=0.01)

                    # predict on new data
                    logger.debug("Testing new predicion sample %s gs %s CV %s" % (ss,gs_options,cv_options))
                    request['command'] = 'predict_dataset_id'
                    request['predict'] = 1
                    request['scoring_dataset_id'] = str(new_dataset_id)

                    # Run 1st Fold
                    sw = SecureWorker(request,Mock())
                    lid1 = sw.run()['lid']

                    # get predticions from DB
                    query = pservice.get_predictions(lid1, request['scoring_dataset_id'])
                    # save predictions to file
                    export_file = os.path.join(self.test_directory,'pred.test')
                    export_predictions(query, export_file, max_sample)
                    # load predictions from file
                    pf = pandas.read_csv(export_file)
                    # predictions should be sequential and 2 * the predictor
                    self.assertTrue(all([pf['Prediction'][i]<=pf['Prediction'][i+1] for i in range(pf.shape[0]-1)]))
                    np.allclose(pf['Prediction'],ds['c'],atol=0.01)

    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch('ModelingMachine.engine.secure_worker.Progress')
    @patch('ModelingMachine.engine.vertex_factory.time.sleep')
    def test_regression_predictions(self,*args):
        for cv_options in (
                { 'cv_method': 'RandomCV', 'reps': 5, 'holdout_pct': 0, },
                { 'cv_method': 'RandomCV', 'reps': 5, 'holdout_pct': 20, },
                { 'cv_method': 'RandomCV', 'reps': 1, 'validation_pct': 0.15, 'holdout_pct': 0, },
                { 'cv_method': 'RandomCV', 'reps': 1, 'validation_pct': 0.15, 'holdout_pct': 20, },
                { 'cv_method': 'UserCV', 'partition_col':'i', 'cv_holdout_level': 3, },
                { 'cv_method': 'UserCV', 'partition_col':'j', 'training_level': 0, 'validation_level':1, 'holdout_level': 2, },
                { 'cv_method': 'UserCV', 'partition_col':'k', 'training_level': 0, 'validation_level':1, },
                { 'cv_method': 'GroupCV', 'partition_key_cols':['i'], 'reps': 4, 'holdout_pct': 0 },
                { 'cv_method': 'GroupCV', 'partition_key_cols':['i'], 'reps': 4, 'holdout_pct': 10 },
                { 'cv_method': 'GroupCV', 'partition_key_cols':['x'], 'reps': 1, 'validation_pct': 0.15, 'holdout_pct': 0, },
                { 'cv_method': 'GroupCV', 'partition_key_cols':['x'], 'reps': 1, 'validation_pct': 0.15, 'holdout_pct': 20, },
                { 'cv_method': 'DateCV', 'datetime_col':'i', 'reps': 1, 'time_validation_pct': 0.15, 'time_holdout_pct': 0, },
                { 'cv_method': 'DateCV', 'datetime_col':'i', 'reps': 1, 'time_validation_pct': 0.15, 'time_holdout_pct': 20, },
                ):
            # set up a project with training data
            uid = '532329fcbb51f7015cf64c96'
            self.pid = ObjectId()
            dataset = self.datasets[7]
            pid = self.create_project(dataset,cv_options=cv_options)
            dataset_id = self.tempstore.read(keyname='wsready', index=str(pid), result=[])[0]
            pservice = ProjectService(pid=self.pid,uid=uid)
            ds = pandas.read_csv(os.path.join(self.test_directory,self.datasets[7]['filename']))

            # upload new data
            filename = dataset['filename']
            response = DatasetService().save_metadata(pid, UploadedFile(filename, pid=pid))
            new_dataset_id = response['upserted']

            holdout_pct = pservice.partition['holdout_pct']
            if 'validation_pct' in pservice.partition:
                max_sample = 100 * (1-pservice.partition['validation_pct']) - holdout_pct
            else:
                reps = pservice.partition['reps']
                max_sample = (100 - holdout_pct) * (reps - 1) / reps
            request = {
                'blueprint': {'1': (['NUM'], ['NI'], 'T'), '2': (['1'], ['GLMR'], 'P')},
                'lid': str(ObjectId()),
                'uid': uid,
                'pid': pid,
                'dataset_id': dataset_id,
                'blueprint_id': ObjectId(),
                'qid': 1,
                'total_size': 100.0,
                'icons': [0],
                'bp': 1,
                'new_lid': True,
                'max_folds': 0}

            # test with and without grid search
            for gs_options in ('',' p=[1.1,1.2]'):
                request['blueprint'] = {'1': (['NUM'], ['NI'], 'T'), '2': (['1'], ['GLMR%s' % gs_options], 'P')}
                # test small/medium/large sample sizes
                for ss in (max_sample-1,max_sample+1,99):
                    # predict on training data
                    logger.debug("Testing predicion sample %s gs %s CV %s" % (ss,gs_options,cv_options))
                    request['samplepct'] = ss
                    request['command'] = 'fit'
                    request.pop('scoring_dataset_id',1)
                    if ss>max_sample:
                        request['partitions'] = [[-1,-1]]
                    else:
                        request['partitions'] = [[0,-1]]
                    # Run 1st Fold
                    sw = SecureWorker(request,Mock())
                    lid1 = sw.run()

                    # get predticions from DB
                    query = pservice.get_predictions(lid1, request['dataset_id'])
                    # save predictions to file
                    export_file = os.path.join(self.test_directory,'pred.test')
                    export_predictions(query, export_file, max_sample)
                    # load predictions from file
                    pf = pandas.read_csv(export_file)
                    # predictions should be sequential and 2 * the predictor
                    self.assertTrue(all([pf['Cross-Validation Prediction'][i]<pf['Cross-Validation Prediction'][i+1] for i in range(pf.shape[0]-1)]))
                    self.assertAlmostEqual((pf['Cross-Validation Prediction']/ds['x']).mean(),2)

                    # predict on new data
                    logger.debug("Testing new predicion sample %s gs %s CV %s" % (ss,gs_options,cv_options))
                    request['command'] = 'predict_dataset_id'
                    request['predict'] = 1
                    request['scoring_dataset_id'] = str(new_dataset_id)

                    # Run 1st Fold
                    sw = SecureWorker(request,Mock())
                    lid1 = sw.run()['lid']

                    # get predticions from DB
                    query = pservice.get_predictions(lid1, request['scoring_dataset_id'])
                    # save predictions to file
                    export_file = os.path.join(self.test_directory,'pred.test')
                    export_predictions(query, export_file, max_sample)
                    # load predictions from file
                    pf = pandas.read_csv(export_file)
                    # predictions should be sequential and 2 * the predictor
                    self.assertTrue(all([pf['Prediction'][i]<pf['Prediction'][i+1] for i in range(pf.shape[0]-1)]))
                    self.assertAlmostEqual((pf['Prediction']/ds['x']).mean(),2)


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    #logfile = logging.FileHandler(__file__+'.log',mode='w')
    logger.addHandler(console)
    #logger.addHandler(logfile)

    unittest.main()

