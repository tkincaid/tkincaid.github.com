import unittest
import json
from bson.objectid import ObjectId
from mock import patch, Mock

from config.engine import EngConfig
from config.test_config import db_config as config
from common.wrappers import database

from common.services.project import ProjectServiceBase as ProjectService
from common.services.autopilot import AutopilotService, DONE, MANUAL, SEMI, AUTO
from common.engine.progress import ProgressSink
import common
from common import load_class

from MMApp.entities.user import UserService
from MMApp.entities.roles import RoleProvider
from MMApp.entities.jobqueue import QueueService
from MMApp.entities.permissions import Roles, Permissions

class TestProjectClass(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.tempstore = database.new("tempstore", host=config['tempstore']['host'],
                port=config['tempstore']['port'])
        self.persistent = database.new("persistent", host=config['persistent']['host'],
                port=config['persistent']['port'], dbname=config['persistent']['dbname'])

    @classmethod
    def tearDownClass(self):
        self.tempstore.conn.flushdb()
        self.persistent.destroy(table='users')
        self.persistent.destroy(table='project')

    def setUp(self):
        self.tempstore.conn.flushdb()
        self.persistent.destroy(table='users')
        self.persistent.destroy(table='project')

    def create_project(self):
        username = 'a@asdf.com'
        userservice = UserService(username)
        userservice.create_account('asdfasdf')
        uid = userservice.uid

        dataset_id = ObjectId()

        pid = self.persistent.create({'target':{'name':'a', 'size':100, 'type':'Regression'},
            'partition': {'reps':5, 'folds':1, 'total_size':100, 'holdout_pct':20, 'seed':0},
            'holdout_pct': 20,
            'default_dataset_id':str(dataset_id)}, table='project')

        self.persistent.create({'shape':[100,2], 'varTypeString':'NN', 'columns':[(1,'a',1),(2,'b',2)],
            'pct_min_y': 0.01, 'pct_max_y': 0.01, 'nunique_y': 90,
            'pid':pid, '_id':dataset_id}, table='metadata')

        role_provider = RoleProvider()
        role_provider.set_roles(uid, pid, [Roles.OWNER])
        #old way of initializing queue settings
        queue = QueueService(str(pid),ProgressSink())
        queue.set_autopilot({'workers':2, 'mode':1})
        return username, uid, pid, str(dataset_id)

    def test_get_settings(self):
        username, uid, pid, dataset_id = self.create_project()

        autopilot = AutopilotService(pid, uid)

        out = autopilot.get_settings()
        print out

        expected = {'dataset_id':dataset_id, 'workers':2, 'mode':1, 'done':False}
        self.assertEqual(out, expected)

        queue = QueueService(str(pid),ProgressSink())
        queue.set_autopilot({'mode':2})
        expected.update({'mode':2})
        out = autopilot.get_settings()
        self.assertEqual(out, expected)

        queue = QueueService(str(pid),ProgressSink())
        queue.set_autopilot({'mode':0})
        expected.update({'mode':0})
        out = autopilot.get_settings()
        self.assertEqual(out, expected)

    def test_set_done(self):
        username, uid, pid, dataset_id = self.create_project()

        autopilot = AutopilotService(pid, uid, ProgressSink())
        autopilot.set_done()
        expected = {'dataset_id':dataset_id, 'workers':2, 'mode':1, 'done':True}
        out = autopilot.get_settings()
        self.assertEqual(out, expected)

        self.assertEqual(autopilot.settings, out)

    def test_permissions(self):
        username, uid, pid, dataset_id = self.create_project()

        with self.assertRaises(Exception):
            autopilot = AutopilotService('invalid', 'invalid', ProgressSink())

        with self.assertRaises(Exception):
            autopilot = AutopilotService('invalid', uid, ProgressSink())

        with self.assertRaises(Exception):
            autopilot = AutopilotService(pid, 'invalid', ProgressSink())

        with patch('common.services.autopilot.raise_invalid_perms') as mock_raise:
            #valid permissions
            autopilot = AutopilotService(pid, uid)
            self.assertEqual(mock_raise.call_count, 0)
            #invalid permissions
            autopilot = AutopilotService(pid, str(ObjectId()))
            self.assertEqual(mock_raise.call_count, 1)

    def test_set(self):
        username, uid, pid, dataset_id = self.create_project()

        autopilot = AutopilotService(pid, uid, ProgressSink())

        for mode in [0,1,2]:
            request = {'mode':mode}
            autopilot.set(request)
            expected = {'dataset_id':dataset_id, 'workers':2, 'mode':mode, 'done':False}
            out = autopilot.get_settings()
            print out
            self.assertEqual(out, expected)

    def test_set_does_not_change_eng_config(self):

        serialized_config = json.dumps(EngConfig)

        username, uid, pid, dataset_id = self.create_project()

        autopilot = AutopilotService(pid, uid, ProgressSink())

        for mode in [0,1,2]:
            request = {'mode':mode}
            autopilot.set(request)
            expected = {'dataset_id':dataset_id, 'workers':2, 'mode':mode, 'done':False}
            out = autopilot.get_settings()
            self.assertEqual(out, expected)

        after_the_fact_config = json.dumps(EngConfig)

        self.assertEqual(serialized_config, after_the_fact_config, 'EngConfig should remain immutable')


    def test_apply_settings_on_restart(self):
        username, uid, pid, dataset_id = self.create_project()

        autopilot = AutopilotService(pid, uid, ProgressSink())

        args = autopilot.get_settings()
        request = {'restart':True, 'dataset_id':'12341234'}

        out = autopilot._apply_settings(request, args)
        expected = {'dataset_id':'12341234', 'mode':SEMI, 'done':False}

        print out
        self.assertEqual(out, expected)

    def test_restart(self):
        username, uid, pid, dataset_id = self.create_project()

        autopilot = AutopilotService(pid, uid, ProgressSink())

        expected = autopilot.get_settings()

        q = QueueService(pid, ProgressSink(), uid)

        out = q.get()
        self.assertEqual(len(out),1)

        #initialize queue with default dataset_id
        self.call_mb_next_steps(pid, uid)

        out = q.get()
        self.assertGreater(len(out),1)
        number_of_models = 0
        for i in out:
            if i['status'] == 'settings':
                continue
            number_of_models +=1
            self.assertEqual(i['dataset_id'], dataset_id)

        print 'mb added %s models with dataset_id = %s'%(number_of_models, dataset_id)

        #test testings are updated on restart request
        newdataset_id = self.persistent.create({'shape':[100,2], 'varTypeString':'NN', 'columns':[(1,'a',1),(2,'b',2)],
            'pid':pid}, table='metadata')

        #'pct_min_y': 0.01, 'pct_max_y': 0.01, 'nunique_y': 90,

        autopilot.set_done()
        out = autopilot.get_settings()
        self.assertEqual(out['done'], True)

        request = {'restart':True, 'dataset_id':str(newdataset_id)}
        with patch('common.services.autopilot.MMClient') as mock_mmclient:
            autopilot.set(request)
            self.assertEqual(mock_mmclient.call_count, 1)

        out = autopilot.get_settings()
        expected.update({'dataset_id':str(newdataset_id), 'done':False})
        self.assertEqual(out, expected)

        #test metablueprint can restart on new dataset_id
        self.call_mb_next_steps(pid, uid)

        out = q.get()
        self.assertEqual(len(out), 1+2*number_of_models)
        print set([i.get('dataset_id') for i in out])
        for i in out[number_of_models+1:]:
            self.assertEqual(i['dataset_id'], str(newdataset_id))


    def call_mb_next_steps(self, pid, uid):
        Metablueprint = load_class(EngConfig['metablueprint_classname'])
        mb = Metablueprint(pid, uid, progressInst=ProgressSink(), reference_models=True,
                tempstore=self.tempstore, persistent=self.persistent)

        mb.get_recommended_metrics = Mock()
        metrics_ = {'accuracy': 'Gini',
                    'ranking': 'Gini'}
        mb.get_recommended_metrics.return_value = metrics_

        mb()

if __name__ == '__main__':
    unittest.main()

