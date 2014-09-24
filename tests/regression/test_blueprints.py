"""
Depends on DB_APIClient
Depends on the production master mongo server address
Depends on production storage config values
"""
############################################################################
#
#       Copyright DataRobot, Inc. 2014
#
###########################################################################

import numpy as np
import sys
import unittest
import os
import random
import time
import shutil
import datetime
from cStringIO import StringIO
import logging
from copy import deepcopy
import json
from bson.objectid import ObjectId
import argparse

from mock import patch, Mock, PropertyMock

from config.engine import EngConfig
from config.test_config import db_config
from common.wrappers import database
from common.wrappers.dbs.mongo_db import MongoDB
from common.storage import FileObject, S3Storage
from ModelingMachine.engine.secure_worker import SecureWorker
from ModelingMachine.engine.mocks import DB_APIClient
from ModelingMachine.engine.monitor import FakeMonitor
from common.engine.progress import ProgressSink
from common.entities.job import DataRobotJob

logger = logging.getLogger('bptest')
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)
drlogger = logging.getLogger('datarobot')
drlogger.setLevel(logging.ERROR)

mm = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(mm,'../..'))

class BlueprintRegression(object):
    def __init__(self):
        pass

    def setUp(self):
        self.patchers = []
        #self.addCleanup(self.stopPatching)

        tests_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.test_directory = os.path.join(tests_dir,'testworkspacetemp')
        try:
            shutil.rmtree(self.test_directory)
        except OSError:
            pass

        file_storage_patcher = patch.dict(EngConfig, {
            'LXC_CONTEXT_BASE': os.path.join(self.test_directory, 'lxc'),
            'LOCAL_FILE_STORAGE_DIR' : os.path.join(self.test_directory, 'local_file_storage'),
            'CACHE_DIR' : os.path.join(self.test_directory,'cache'),
            'CACHE_LIMIT': 100*1024*1024,
            'DATA_DIR' : os.path.join(self.test_directory,'temp'),
            'FILE_STORAGE_TYPE': "s3",
            'FILE_STORAGE_PREFIX': "datarobot/production/"
            }, clear = False)

        prod_master = 'mongo-2.prod.aws.datarobot.com'
        db_patcher = patch.dict(db_config['persistent'], {
            'host': prod_master,
            'port': 27017,
            'dbname': "MMApp"
        }, clear = False)

        db_patcher2 = patch.dict(db_config['tempstore'], {
                'use_sentinel': True,
                'host': 'redis-0.prod.aws.datarobot.com',
                'port': 6379,
                'default_master_name': 'redis0',
                'sentinels': [('mongo-1.prod.aws.datarobot.com', 26379), ('mongo-0.prod.aws.datarobot.com', 26379), ('mongo-2.prod.aws.datarobot.com', 26379)]
            }, clear=False)

        self.dp_api_patch = patch('ModelingMachine.engine.data_processor.APIClient', DB_APIClient)
        self.sw_api_patch = patch('ModelingMachine.engine.secure_worker.APIClient', DB_APIClient)
        self.sw_monitor_patch = patch('ModelingMachine.engine.secure_worker.Monitor', FakeMonitor)
        self.sw_progress_patch = patch('ModelingMachine.engine.secure_worker.Progress', ProgressSink)

        self.patchers.append(file_storage_patcher)
        self.patchers.append(db_patcher)
        self.patchers.append(db_patcher2)
        self.patchers.append(self.dp_api_patch)
        self.patchers.append(self.sw_api_patch)
        self.patchers.append(self.sw_monitor_patch)
        self.patchers.append(self.sw_progress_patch)
        file_storage_patcher.start()
        db_patcher.start()
        db_patcher2.start()
        self.dp_api = self.dp_api_patch.start()
        self.sw_api = self.sw_api_patch.start()
        self.sw_monitor_patch.start()
        self.sw_progress_patch.start()

        logger.info("db:, %s", db_config)

        os.mkdir(self.test_directory)
        os.mkdir(EngConfig['LXC_CONTEXT_BASE'])
        os.mkdir(EngConfig['DATA_DIR'])

        # self.persistent = database.new("persistent", host=db_config['persistent']['host'],
        #         port=db_config['persistent']['port'], dbname=db_config["persistent"]["dbname"])

        prod_master = 'mongo-2.prod.aws.datarobot.com'
        self.production_db = MongoDB("persistent", host=prod_master,
                port=27017, dbname="MMApp")

    def init_project(self):
        """Before running this, create a production project with the dataset to use
        and specify the ids for that project below
        """
        #self.target = 'numeric'
        #self.target_type = 'Regression'
        self.target = 'binary'
        self.target_type = 'Binary'
        header = ["binary", "numeric", "num1", "num2", "num3", "cat1", "cat2", "txt1", "txt2", "user1", "item1"]

        self.uid = "534f82e81331270647c3d0a7"
        self.pid = "53baa5751331277d5092bc12"
        self.dsid = "53baa578e6d3f95c856de422"
        self.filename = "378c1be5-265a-4d12-9a4e-4db48e64b5c3"
        self.storage_filename = 'projects/'+self.pid+'/raw/'+self.filename
        #self.create_data(self.storage_filename)

        #changing as little as possible to allow migrations to fix issues
        #this will ideally only change the fields necessary for the test
        self.dataset = {"columns": [[i,col,0] for i,col in enumerate(header)],
                        "varTypeString" : self.vartypestring_from_header(header)}
        self.project = {"default_dataset_id" : self.dsid,
                        "target" : {
                            "type" : self.target_type,
                            "name" : self.target,
                            "size" : 160
                        },
                        "target_options" : {
                            "positive_class" : None,
                            "name" : self.target,
                            "missing_maps_to" : None
                        }
                    }

        self.production_db.update({'_id': ObjectId(self.pid)}, table="project", values=self.project)
        self.production_db.update({'_id': ObjectId(self.dsid)}, table="metadata", values=self.dataset)

    def stopPatching(self):
        for patcher in self.patchers:
            if patcher:
                patcher.stop()

    def clean(self, bps):
        """Remove all vertex files and remove leaderboard records
        """
        self.production_db.conn.leaderboard.remove({'pid': ObjectId(self.pid)})
        storage = S3Storage()
        key_query = storage.bucket.list(prefix="datarobot/production/projects/"+self.pid+"/vertex/")
        keys = [k for k in key_query]
        for k in keys:
            k.delete()

    def valid_bp(self, bp):
        """For filtering blueprints post-query
        """
        if bp is None:
            return False
        valid = True
        for v in bp.values():
            for i in v[0]:
                if not (i in ['NUM','CAT','TXT'] or i.isdigit()):
                    valid = False
        return valid

    def get_bps(self):
        """Blueprints collection fields: bp, timestamp, lid, supported, failed
        """
        bp_query = self.production_db.conn.blueprints.find()
        bps = [r for r in bp_query]

        return bps

    def get_new_bps(self, since=None):
        if since is None:
            since = datetime.datetime(2014,4,1)

        bps_query = self.production_db.conn.leaderboard.aggregate(
                        [{'$match': {'_id': {'$gt': ObjectId.from_datetime(since)}}},
                         {'$group': {'_id': '$blueprint'}}])

        bp_list = [b['_id'] for b in bps_query['result']]

        bps = []
        for b in bp_list:
            if self.valid_bp(b):
                bps.append(b)

        return bps

    def get_bp_list(self, bps, new_bps, skip_completed=False):
        bp_list = [b['bp'] for b in bps]
        new_bp_list = [b for b in new_bps if b not in bp_list]

        if skip_completed:
            supported_bps = [b['bp'] for b in bps if b['supported'] and (b.get('fit_failed', True) or b['failed'])]
        else:
            supported_bps = [b['bp'] for b in bps if b['supported']]

        return supported_bps + new_bp_list

    def supported_models(self, bps, models):
        lids = [b['lid'] for b in bps if b['supported'] and not b.get('fit_failed', False)]
        #logger.info("Supported lids: %s", lids)
        supported_model_list = []
        for m in models:
            if m['_id'] in lids:
                logger.info("Including lid: %s", m['_id'])
                supported_model_list.append(m)
            else:
                logger.info("Skipping lid: %s", m['_id'])
        return supported_model_list

    def get_leaderboard(self):
        #get leaderboard
        #exclude roc, lift, extras
        lb_query = self.production_db.conn.leaderboard.find({'pid': ObjectId(self.pid)},
                    fields={'roc': False,
                            'lift': False,
                            'extras': False})
        lb = [r for r in lb_query]

        return lb

    def vartypestring_from_header(self, header):
        vts= []
        for col in header:
            if col.startswith("user"):
                vt = "N"
            elif col.startswith("item"):
                vt = "N"
            elif col.startswith("cat"):
                vt = 'C'
            elif col.startswith('txt'):
                vt = 'T'
            else:
                vt = 'N'
            vts.append(vt)
        return ''.join(vts)

    def create_data(self, storage_filename):
        """
        """
        random.seed(1)
        np.random.seed(1)

        size = 200
        words = "a string of words to choose from to create another random string with no particular meaning"
        binary = np.random.randint(2, size=size)
        numeric = np.random.randint(200, size=size)
        num1 = np.random.normal(size=size)
        num2 = np.random.uniform(size=size)
        num3 = [random.choice([1,2,3,4,5]) for i in range(size)]
        cat1 = [random.choice(['a','b','c']) for i in range(size)]
        cat2 = [random.choice(['d','e','f']) for i in range(size)]
        txt1 = [' '.join([random.choice(words.split()) for j in range(np.random.randint(2,7))]) for i in range(size)]
        txt2 = [' '.join([random.choice(words.split()) for j in range(np.random.randint(2,7))]) for i in range(size)]
        user1 = [random.choice([1,2,3,4,5]) for i in range(size)]
        item1 = [random.choice([1,2,3,4,5]) for i in range(size)]

        # make a csv to use the same pandas method as the app, probably not necessary
        header = ["binary", "numeric", "num1", "num2", "num3", "cat1", "cat2", "txt1", "txt2", "user1", "item1"]
        rows = [header] + zip(binary, numeric, num1, num2, num3, cat1, cat2, txt1, txt2, user1, item1)
        rowstrings = [','.join(map(str,row))+"\n" for row in rows]
        # f = StringIO()
        # f.writelines(rowstrings)
        # f.seek(0)
        raw_file = os.path.join(self.test_directory,'test.csv')
        with open(raw_file, "w") as f:
            f.writelines(rowstrings)

        FileObject(storage_filename).put(raw_file)

    def get_request(self, bp):
        return {'blueprint': bp,
                'samplepct': 50,
                'partitions': [[0,-1]],
                'max_folds': 0,
                'blender': {},
                'command':'fit',
                'dataset_id': self.dsid,
                'pid': self.pid,
                'uid': self.uid,
                'qid': 'testqid',
                'blueprint_id': 'testbp_id',
                'lid': str(ObjectId()),
                'new_lid': False}

    def predict(self):
        self.lb = self.get_leaderboard()
        self.bps = self.get_bps()
        self.models = self.supported_models(self.bps, self.lb)

        success = True
        for l in self.models:
            logger.info("Predicting: %s", l['blueprint'])
            l['predict'] = 1
            l['scoring_dataset_id'] = self.dsid
            jobs = DataRobotJob(l).to_joblist()

            if not jobs:
                success = False
                all_jobs = False
            else:
                all_jobs = True
                for j in jobs:
                    j['command'] = 'predict'
                    worker = SecureWorker(j, Mock())
                    out = worker.run()
                    if not out:
                        logger.info("Job failed: %s", j)
                        all_jobs = False
                        success = False

            record = {'bp': l['blueprint'],
                      'timestamp': datetime.datetime.utcnow(),
                      'failed': False}

            if not all_jobs:
                logger.info("Blueprint failed: %s", l['blueprint'])
                record['failed'] = True

            self.production_db.update({'bp': record['bp']}, table="blueprints", values=record)

        return success

    def fit(self, skip_completed=False):
        """Creates blueprint records
        The field "supported" is used to manually disable a blueprint
        """
        logger.info("Getting existing list of blueprints")
        self.bps = self.get_bps()
        logger.info("Getting new blueprints")
        self.new_bps = self.get_new_bps()
        logger.info("Checking blueprints to exclude")
        self.bp_list = self.get_bp_list(self.bps, self.new_bps, skip_completed=skip_completed)

        if not skip_completed:
            logger.info("Removing existing data")
            #clear out everything that is about to be recreated
            self.clean(self.bps)

        #self.create_data(self.storage_filename)

        success = True
        for i,bp in enumerate(self.bp_list):
            logger.info("Adding blueprint: %s", bp)
            request = self.get_request(bp)
            worker = SecureWorker(request, Mock())
            out = worker.run()

            record = {'bp': bp,
                      'supported': True,
                      'lid': ObjectId(request['lid']),
                      'timestamp': datetime.datetime.utcnow(),
                      'failed': False,
                      'fit_failed': False}

            if not out:
                logger.info("Fit failed: %s", request)
                success = False
                record['fit_failed'] = True

            self.production_db.update({'bp': bp}, table="blueprints", values=record)

        #TODO: try failed blueprints with a regression or poisson/gamma distributed response

        return success

class TestBlueprints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.bptest = BlueprintRegression()
        self.bptest.setUp()
        self.bptest.init_project()
        self.addCleanup(self.bptest.stopPatching)

    def tearDown(self):
        pass

    def test_blueprints(self):
        #predict existing leaderboard items
        pred = self.bptest.predict()
        self.assertTrue(pred)

if __name__ == '__main__':
    filt = logging.Filter("bptest")
    sh = logging.StreamHandler()
    sh.addFilter(filt)
    logger.addHandler(sh)
    drlogger.addHandler(logging.StreamHandler())

    parser = argparse.ArgumentParser(description='Blueprint regression test')
    parser.add_argument('--fit', action='store_true')
    parser.add_argument('--skip', action='store_true')

    args = parser.parse_args()
    try:
        if args.fit:
            bptest = BlueprintRegression()
            bptest.setUp()
            bptest.init_project()
            bptest.fit(skip_completed=args.skip)
            bptest.stopPatching()
        else:
            bptest = BlueprintRegression()
            bptest.setUp()
            bptest.init_project()
            pred = bptest.predict()
            logger.info("Prediction result: %s", pred)
            bptest.stopPatching()
            if not pred:
                sys.exit(1)
    except Exception:
        logger.error("", exc_info=True)
        # unittest.main()
