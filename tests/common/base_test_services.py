import unittest
import pytest
from bson.objectid import ObjectId
from mock import patch, Mock

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

from common.services.eda import EdaService

class TestServiceBase(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.tempstore = database.new("tempstore", host=config['tempstore']['host'],
                port=config['tempstore']['port'])
        self.persistent = database.new("persistent", host=config['persistent']['host'],
                port=config['persistent']['port'], dbname=config['persistent']['dbname'])

        self.pid = ObjectId()

    @classmethod
    def tearDownClass(self):
        self.tempstore.conn.flushdb()
        for table in ['users','project','eda','eda_map','metadata']:
            self.persistent.destroy(table=table)

    def setUp(self):
        self.tempstore.conn.flushdb()
        for table in ['users','project','eda','eda_map','metadata']:
            self.persistent.destroy(table=table)

    def create_eda(self):
        col_a_key = 'a'
        col_a_eda = {
            'profile': {
                'info': 0.1,
                'plot': [],
                'name': 'a',
                'miss_count': 0,
                'y': 'b',
                'plot2': [],
                'type': 'N',
                'miss_mean': 0},
            'transform_args': [],
            'low_info': {},
            'metric_options': {},
            'summary': [],
            'transform_id': 0,
            'id': 0,
            'raw_variable_index': 0}
        col_b_key = 'b'
        col_b_eda =  {
            'profile': {
                'info': 0.2,
                'plot': [],
                'name': 'b',
                'miss_count': 0,

                'y': 'b',
                'plot2': [],
                'type': 'N',
                'miss_mean': 0},
            'transform_args': [],
            'low_info': {},

            'metric_options': {},
            'summary': [],
            'transform_id': 0,
            'id': 1,
            'raw_variable_index': 1}
        out = {
            'eda' : {
                col_a_key: col_a_eda,
                col_b_key: col_b_eda,
            },
            'pid':self.pid}
        _id = self.persistent.create(out, table='eda')
        self.persistent.create({'pid':self.pid,'dataset_id':'universe','block_contents':{str(_id):['a','b']}}, table='eda_map')

    def create_project(self):
        username = 'a@asdf.com'
        userservice = UserService(username)
        userservice.create_account('asdfasdf')
        uid = userservice.uid

        dataset_id = ObjectId()

        self.persistent.create({'_id':self.pid,'target':{'name':'a', 'size':100, 'type':'Regression'},
            'partition': {'reps':5, 'folds':1, 'total_size':100, 'holdout_pct':20, 'seed':0},
            'holdout_pct': 20, 'version':1.1, 'cv_method':'RandomCV',
            'default_dataset_id':str(dataset_id)}, table='project')

        self.persistent.create({'shape':[100,2], 'varTypeString':'NN', 'columns':[(1,'a',1),(2,'b',2)],
            'pct_min_y': 0.01, 'pct_max_y': 0.01, 'nunique_y': 90,
            'pid':self.pid, '_id':dataset_id}, table='metadata')

        role_provider = RoleProvider()
        role_provider.set_roles(uid, self.pid, [Roles.OWNER])
        #old way of initializing queue settings
        queue = QueueService(str(self.pid),ProgressSink())
        queue.set_autopilot({'workers':2, 'mode':1})
        #eda
        self.create_eda()
        return username, uid, self.pid, str(dataset_id)

    def fake_leaderboard_item(self, updates=None):
        out = {'pid':self.pid, '_id':ObjectId(None), 'model_type':'Test Model', 'bp':'5', 'samplepct':50,
                'dataset_id':str(ObjectId(None)), 'blueprint_id': 'fake-bp-id', 'uid':ObjectId(None),
                'max_folds':1,
                'blueprint':{'1':[['NUM'], ['NI'], 'T'], '2':[['1'], ['GLMB'], 'P']},
                'partition_stats':dict((str((i,-1)),'testvalue') for i in range(5))}
        if updates:
            out.update(updates)
        return out



