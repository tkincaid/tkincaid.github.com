############################################################################
#
#       app simulation
#
#       Author: Tom DeGodoy
#
#       Copyright DataRobot, Inc. 2013
#
###########################################################################
import unittest
import os
import time
import psutil
import signal
import json
import logging
import numpy as np
import pandas as pd
import pytest

from mock import patch, Mock
from ModelingMachine import worker
from MMApp.entities.db_conn import DBConnections
from MMApp.entities.project import ProjectService
from MMApp.entities.dataset import DatasetService
from common.wrappers import database
from bson import ObjectId
import config.test_config
from tests.IntegrationTests.storage_test_base import StorageTestBase
from ModelingMachine.engine.secure_worker import SecureWorker
from ModelingMachine.engine.worker_request import WorkerRequest
from ModelingMachine.engine.mocks import DB_APIClient
import ModelingMachine
import common.services.eda
from common.services import autopilot
from common.engine import metrics
from common.engine.progress import ProgressSink
from config.engine import EngConfig
import common.io
from common import load_class
from common.services.flippers import FLIPPERS


logger = logging.getLogger('datarobot')

class TestBlueprints(StorageTestBase):

    @classmethod
    def setUpClass(cls):
        super(TestBlueprints, cls).setUpClass()
        cls.dbs = DBConnections()
        cls.tempstore = database.new('tempstore')
        cls.persistent = database.new('persistent')

        cls.get_collection = cls.dbs.get_collection
        cls.collection = cls.dbs.get_collection('project')
        cls.redis_conn = cls.dbs.get_redis_connection()

        cls.datasets = []
        cls.datasets.append({'filename':'credit-sample-200.csv',
                             'target':[u'SeriousDlqin2yrs','Binary'],
                             'metric': metrics.LOGLOSS})
        cls.datasets.append({'filename':'allstate-nonzero-200.csv',
                             'target':['Claim_Amount','Regression'], 'metric': metrics.GINI_NORM})
        cls.datasets.append({'filename':'kickcars-sample-200.csv',
                             'target':['IsBadBuy','Binary'],
                             'metric': metrics.LOGLOSS})
        cls.datasets.append({'filename':'credit-train-small.csv',
                             'target':[u'SeriousDlqin2yrs','Binary'],
                             'metric': metrics.LOGLOSS})
        cls.datasets.append({'filename': 'amazon_fr_text-only-100.csv',
                             'target': ['rating', 'Regression'], 'metric': metrics.RMSE})
        cls.datasets.append({'filename': 'movielense_tiny.csv',
                             'target': ['rating', 'Regression'],
                             'metric': metrics.RMSE, 'recommender': True,
                             'recommender_user_id': 'user_id',
                             'recommender_item_id': 'item_id'})

        cls.test_directory, _ = cls.create_test_files(cls.datasets)

    @classmethod
    def tearDownClass(cls):
        super(TestBlueprints, cls).tearDownClass()
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
        super(TestBlueprints, self).setUp()
        self.clear_tempstore_except_workers()
        self.dbs.destroy_database()

    def tearDown(self):
        super(TestBlueprints, self).tearDown()
        #kill child processes
        children = psutil.Process(os.getpid()).get_children(recursive=True)
        if len(children)>0:
            for child in children:
                try:
                    os.kill(child.pid,signal.SIGUSR1)
                except:
                    continue
            time.sleep(0.5) #some time for process kill signals to work

    @patch.object(ProjectService, 'assert_has_permission')
    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch('ModelingMachine.engine.secure_worker.Progress')
    def test_leaderboard_item_regression(self,*args,**kwargs):
        """ Sets up a project and runs a specific job in order to test what
        ends up in the leaderboard collection.  Feels dirty to copy-paste
        most of this test, but I don't see any other way of how to set up
        the secure_worker to run a job

        See the code below for how to create a new fixture when you have made
        any backwards compatibility tests and code
        """
        dataset = self.datasets[1]  # Allstate 200

        test_worker = worker.Worker(
            worker_id='1', request={'qid':'1', 'pid':'1'}, pipe=None,
            connect=False, tempstore=self.tempstore,
            persistent=self.persistent)
        file_id = dataset['filename']
        targetname = dataset['target'][0]
        metric = dataset['metric']
        ds, controls = common.io.inspect_uploaded_file(
            os.path.join(self.test_directory, file_id), file_id)
        colnames = list(ds.columns)
        ncols = len(colnames)
        uid = str(ObjectId())
        pid = str(self.pid)

        #-- startCSV request -----------------------------------------------
        upload_filename = 'projects/{}/raw/{}'.format(pid, file_id)
        if FLIPPERS.metadata_created_at_upload:
            ds_service = DatasetService(pid=self.pid, uid=uid,
                                        tempstore=self.tempstore,
                                        persistent=self.persistent)
            dataset_id = ds_service.create_metadata('universe', [upload_filename], controls, ds)
            p_service = ProjectService(pid=self.pid, uid=uid,
                                       tempstore=self.tempstore,
                                       persistent=self.persistent)
            p_service.init_project(upload_filename, file_id, dataset_id)

        request = {'originalName': file_id, 'dataset_id': upload_filename,
                   'uid': uid, 'pid': pid}
        response = test_worker.create_project(request)
        self.persistent.update(table='project',condition={'_id':self.pid},values={'version':1.1})

        #wait for workspace ready signal
        reply = self.redis_conn.blpop('wsready:'+pid,10)
        self.assertNotEqual(reply,None)
        check = reply[1]
        dataset_id = check
        #self.assertEqual(check,dataset_id)
        #wait for eda to complete with a timeout of 10 seconds:
        t = time.time()
        while time.time()-t < 10:
            eda = self.redis_conn.hgetall('eda:'+str(pid))
            if len(eda)==ncols:
                break
            time.sleep(0.5)

        #-- select metric-----------------------------------------
        project = ProjectService(pid, uid)
        # store metric in project collection
        project.metric = metric
        project = ProjectService(pid, uid)

        #-- aim request -------------------------------------------------------
        request = {'uid':uid,'pid':pid,'target':targetname,'folds':5,'reps':5,'holdout_pct':20}
        ## FIXME we need to set the mode since we don't use the queue service here
        autopilot_service = autopilot.AutopilotService(pid, uid, progress=ProgressSink(),
                                                       verify_permissions=False)
        autopilot_service.set({'mode': 1})
        response = test_worker.aim(request)
        time.sleep(0.5)

        queue_query = self.redis_conn.lpop('queue:'+str(pid))
        item = json.loads(queue_query)

        # This is how we will reproducibly have the same blueprint
        item['blueprint'] = {'1': [['NUM'], ['NI'], 'T'],
                             '2': [['1'], ['GLMR'], 'P']}
        item['blueprint_id'] = '789530f32030eb86fdb918cf8d5c03c2'
        item['samplepct'] = 32

        #-- fit request ------------------------------------------------------
        self.redis_conn.hmset('inprogress:'+str(pid),{item['qid']: queue_query})
        item['uid']=uid
        item['command'] = 'fit'

        req = WorkerRequest(item)
        sw = SecureWorker(req, Mock())
        sw.run()

        out = self.persistent.read({'_id':ObjectId(req.lid)},
                                    table='leaderboard', result={})
        # If you have added unit tests for whatever changes necessitate a new
        # test fixture, you can create a new one by uncommenting the following
        # lines, and then copying it into the fixtures directory
        #for k,v in out.items():
        #    if str(type(v))=="<class 'bson.objectid.ObjectId'>":
        #        out[k] = repr(v)
        #with open('allstate-leaderboard-item-fixture.json', 'wb') as outfile:
        #    json.dump(out, outfile) #, cls=MongoEncoder)

        reference_ldb_item_path = os.path.join(
            self.testdatadir, 'fixtures',
            'allstate-leaderboard-item-fixture.json')

        with open(reference_ldb_item_path, 'rb') as in_file:
            reference_ldb_item = json.load(in_file)
            self.assert_leaderboard_item_equal(reference_ldb_item, out)

    def assert_leaderboard_item_equal(self, reference, computed):
        can_differ = ['pid', 'uid', 'dataset_id', 'holdout_scoring_time',
                      'task_info', 'time_real', 'partition_stats',
                      '_id', 'ec2', 'finish_time', 'lid', 'metablueprint',
                      'resource_summary', 'time', 'training_dataset_id', 'job_info']


        self.assertEqual(set(reference.keys()), set(computed.keys()))
        for key in reference.keys():
            if key not in can_differ:
                if key == 'lift':
                    self.assert_lift_equal(reference[key], computed[key])
                else:
                    ref = reference[key]
                    comp = computed[key]
                    if isinstance(ref, dict):
                        self.assertDictEqual(ref, comp)
                    else:
                        fail_msg = 'Key {} differs\nReference: {}\nComputed: {}'
                        self.assertEqual(ref, comp, fail_msg.format(key, ref,
                                                                    comp))

    def assert_lift_equal(self, reference, computed):
        self.assertEqual(set(reference.keys()), set(computed.keys()))
        for key in reference.keys():
            np.testing.assert_almost_equal(reference[key]['pred'],
                                           computed[key]['pred'])
            self.assertEqual(reference[key]['rows'], computed[key]['rows'])
            np.testing.assert_almost_equal(reference[key]['act'],
                                           computed[key]['act'])

    @patch.object(ProjectService, 'assert_has_permission')
    @patch.object(ModelingMachine.engine.mocks.ProjectService, 'assert_has_permission')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    @patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
    @patch('ModelingMachine.engine.secure_worker.zmq')
    @patch('ModelingMachine.engine.secure_worker.Progress')
    def run_full_test(self, dataset, *args, **kwargs):
        """ runs all worker requests in sequence on a given dataset
        """
        test_worker = worker.Worker(worker_id='1',
                                    request={'qid':'1', 'pid':'1'},
                                    pipe=None, connect=False,
                                    tempstore=self.tempstore,
                                    persistent=self.persistent)

        n_folds = kwargs.get('n_folds', 5)
        n_reps = kwargs.get('n_folds', 5)
        holdout_pct = kwargs.get('holdout_pct', 20)

        file_id = dataset['filename']
        targetname = dataset['target'][0]
        metric = dataset['metric']

        is_recommender = dataset.get('recommender', False)
        recommender_user_id = dataset.get('recommender_user_id', None)
        recommender_item_id = dataset.get('recommender_item_id', None)

        ds, file_control = common.io.inspect_uploaded_file(
            os.path.join(self.test_directory, file_id), file_id)
        colnames = list(ds.columns)
        ncols = len(colnames)
        uid = str(ObjectId())
        pid = str(self.pid)

        #-- startCSV request -----------------------------------------------
        upload_filename = 'projects/{}/raw/{}'.format(pid, file_id)
        safe_key = upload_filename.replace('.', '_')
        controls = {safe_key: file_control}
        if FLIPPERS.metadata_created_at_upload:
            ds_service = DatasetService(pid=self.pid, uid=uid,
                                        tempstore=self.tempstore,
                                        persistent=self.persistent)
            dataset_id = ds_service.create_metadata('universe', [upload_filename], controls, ds)
            p_service = ProjectService(pid=self.pid, uid=uid,
                                       tempstore=self.tempstore,
                                       persistent=self.persistent)
            p_service.init_project(upload_filename, file_id, dataset_id)

        request = {'originalName': file_id, 'dataset_id': upload_filename,
                   'uid': uid, 'pid': pid}
        response = test_worker.create_project(request)
        self.persistent.update(table='project',condition={'_id':self.pid},values={'version':1.1})

        #wait for workspace ready signal
        reply = self.redis_conn.blpop('wsready:'+pid,10)
        self.assertNotEqual(reply,None)
        check = reply[1]
        dataset_id = check
        #self.assertEqual(check,dataset_id)
        #wait for eda to complete with a timeout of 10 seconds:
        t = time.time()
        while time.time()-t < 10:
            eda = self.redis_conn.hgetall('eda:'+str(pid))
            if len(eda)==ncols:
                break
            time.sleep(0.5)

        #-- select metric-----------------------------------------
        project = ProjectService(pid, uid)
        # store metric in project collection
        project.metric = metric
        project = ProjectService(pid, uid)

        #-- aim request -------------------------------------------------------
        request = {'uid': uid, 'pid': pid, 'target': targetname, 'folds': n_folds,
                   'reps': n_reps, 'holdout_pct': holdout_pct,
                   'is_recommender': is_recommender, 'recommender_item_id': recommender_item_id,
                   'recommender_user_id': recommender_user_id}
        ## FIXME we need to set the mode since we don't use the queue service here
        autopilot_service = autopilot.AutopilotService(pid, uid, progress=ProgressSink(),
                                                       verify_permissions=False)
        autopilot_service.set({'mode': 1})
        response = test_worker.aim(request)
        time.sleep(0.5)

        # make sure aim doesn't mess with the project metric (regression test)
        project = ProjectService(pid, uid)
        self.assertEqual(project.metric, metric)

        #-- execute models selected by the metablueprint-------------------------
        queue_query = self.redis_conn.lpop('queue:'+str(pid))

        min_samplepct = None
        consumed_at_least_one = False
        while queue_query:
            item = json.loads(queue_query)

            # This is a way to shorten the runtime of the test - only let
            # the system test on the samplesizes that it first submits
            if min_samplepct is None:
                min_samplepct = item['samplepct']
            if item['samplepct'] > min_samplepct or item.get('partitions'):
                break

            #-- fit request ------------------------------------------------------
            self.redis_conn.hmset('inprogress:'+str(pid),{item['qid']: queue_query})
            item['uid']=uid
            item['command'] = 'fit'

            req = WorkerRequest(item)
            sw = SecureWorker(req, Mock())
            sw.run()

            out = self.persistent.read({'_id':ObjectId(req.lid)}, table='leaderboard', result={})
            metric = out['test']['Gini Norm'][0]

            logger.warn('lid= %s, Gini Norm= %s'%(req.lid,metric))

            self.assertNotEqual(metric,None)

            #-- next
            queue_query = self.redis_conn.lpop('queue:'+str(pid))
            if not queue_query:
                response = test_worker.next_steps({'pid':str(pid), 'uid':uid})
                time.sleep(1)
                queue_query = self.redis_conn.lpop('queue:'+str(pid))
            consumed_at_least_one = True

        self.assertTrue(consumed_at_least_one)

        leaderboard1 = ProjectService(pid=pid).get_leaderboard(UI_censoring=False)

        logger.critical('leaderboard uncensored content length = %s', len(str(leaderboard1)))

        leaderboard2 = ProjectService(pid=pid).get_leaderboard(UI_censoring=True)

        logger.critical('leaderboard censored content length = %s'%len(str(leaderboard2)))
        return leaderboard1

    def make_a_text_only_dataset(self, nrows=100):
        rng = np.random.RandomState(12)
        y = rng.randn(nrows)
        words = ['cat', 'dog', 'jump', 'run', 'hamiltonian', 'eigenfunction']
        words_vec = [' '.join(rng.choice(words, 10)) for i in range(nrows)]
        data = pd.DataFrame({'x': words_vec, 'y': y})

    @patch.dict(EngConfig, {'metablueprint_classname':
                            EngConfig['original_metablueprint_classname']})
    def test_on_credit_data(self):
        """Run DummyMB on credit dataset using LogLoss as metric.

        WARN: this test depends on blueprint numbering.
        """
        lb = self.run_full_test(self.datasets[0])
        lb = sorted(lb, key=lambda d: d['bp'])
        glmb = [l for l in lb if l['model_type'] ==
                'Generalized Linear Model (Bernoulli Distribution)']
        d = glmb[0]
        self.assertEqual(d['model_type'], 'Generalized Linear Model (Bernoulli Distribution)')
        score = d['test'][metrics.LOGLOSS][0]
        np.testing.assert_almost_equal(score, 0.37548, decimal=3)

    @patch.dict(EngConfig, {'metablueprint_classname':
                            EngConfig['original_metablueprint_classname']})
    def test_on_allstate_data(self):
        """Run DummyMB on allstate dataset using Gini Norm as metric

        WARN: this test depends on blueprint numbering.
        """
        lb = self.run_full_test(self.datasets[1])
        lb = sorted(lb, key=lambda d: int(d['bp']))
        d = lb[2]
        self.assertEqual(d['bp'], 3)
        self.assertEqual(d['model_type'], 'Ridge Regression')
        gini_norm = d['test'][metrics.GINI_NORM][0]
        np.testing.assert_almost_equal(gini_norm, 0.33309, decimal=3)

    @patch.dict(EngConfig, {'metablueprint_classname':
                            EngConfig['original_metablueprint_classname']})
    def test_on_amazon_fr_text_only(self):
        # Let's get a better test for this
        self.run_full_test(self.datasets[4])

    def test_movielense(self):
        """Recommender system test. """
        lb = self.run_full_test(self.datasets[5], n_folds=2, n_reps=1, holdout_pct=0)
        lb = sorted(lb, key=lambda d: int(d['bp']))
        d = lb[0]
        self.assertEqual(d['bp'], 1)
        self.assertEqual(d['model_type'], 'Most Popular Items Recommender')
        rmse = d['test'][metrics.RMSE][0]
        np.testing.assert_almost_equal(rmse, 1.08086, decimal=3)
        ndcg = d['test'][metrics.NDCG][0]
        np.testing.assert_almost_equal(ndcg, 0.87503, decimal=3)
        Metablueprint = load_class(EngConfig['recommender_mb_classname'])
        self.assertEqual(d['metablueprint'], [Metablueprint.__name__, Metablueprint.version])


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    logfile = logging.FileHandler(__file__+'.log',mode='w')
    logfile.setLevel(logging.WARN)
    logger.addHandler(logfile)

    unittest.main()
