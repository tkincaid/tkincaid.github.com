"""This is a unittest for the Base Metablueprint.  I'm
pretty sure that if we decide to follow PEP3119 that his
won't work, because then we couldn't get an instance
of the base class
"""


# ########################################################
#
# Unit Test for Base Metablueprint
#
#       Author: Tom de Godoy, Dallin Akagi
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

import unittest
import json
import copy
import random
import string
import pytest

from mock import patch

from config.test_config import db_config as config
from common.wrappers import database

import ModelingMachine.metablueprint.ref_model_mixins as rmm
from ModelingMachine.engine.metrics import ALL_METRICS
from ModelingMachine.metablueprint.base_metablueprint import BaseMetablueprint
from ModelingMachine.metablueprint.base_metablueprint import job_signature
import ModelingMachine.metablueprint.base_metablueprint as basemb_module
import common.entities.blueprint as blueprint_module
import common.services.eda

from config.engine import EngConfig

from bson.objectid import ObjectId

import common.services.flippers

FLIPPERS = common.services.flippers.GlobalFlipper()


class BaseTestMB(unittest.TestCase):
    """ Base Class for Meta-Blueprint Unit Tests
    """

    @classmethod
    def setUpClass(self):
        self.tempstore = database.new("tempstore",
                                      host=config['tempstore']['host'],
                                      port=config['tempstore']['port'])
        self.persistent = database.new("persistent",
                                       host=config['persistent']['host'],
                                       port=config['persistent']['port'],
                                       dbname=config['persistent']['dbname'])

    @classmethod
    def tearDownClass(cls):
        cls.persistent.destroy(table='project')
        cls.persistent.destroy(table='metadata')
        cls.persistent.destroy(table='leaderboard')
        cls.tempstore.conn.flushdb()

    def setUp(self):
        self.persistent.destroy(table='project')
        self.persistent.destroy(table='metadata')
        self.persistent.destroy(table='leaderboard')
        self.tempstore.conn.flushdb()

    def tearDown(self):
        pass

    class ConstructableBaseMB(BaseMetablueprint):
        """Used to test the methods of BaseMetablueprint.  Necessary because
        BaseMetablueprint is an Abstract Base Class
        """

        def next_steps(self, metric):
            pass

    class QueueKey(object):
        QUEUE = 'queue'
        ONHOLD = 'onhold'
        ERRORS = 'errors'
        INPROGRESS = 'inprogress'

    def one_fake_ldb_item(self, pid, samplepct=50, total_size=2000,
                          max_reps=1, bp=0, blend=0, bpoffset=0,
                          scoreoffset=0, blueprint=None,
                          blueprint_id=None, dataset_id='52b6121e076329407cb2c88d'):
        fake = {}
        if blueprint is None:
            blueprint = {
                '1': [['NUM'], ['NI'], 'T'],
                '2': [['1'], ['RFI'], 'P']
            }
        fake['blueprint'] = blueprint
        fake['features'] = [self.randomword(10) for i in xrange(3)]
        fake['total_size'] = total_size
        fake['samplepct'] = samplepct
        fake['parts'] = None
        fake['vertex_cnt'] = 1
        fake['task_cnt'] = 2
        fake['model_type'] = self.randomword(10)
        fake['test_size'] = random.randint(100, fake['total_size'])
        fake['pid'] = pid
        fake['uid'] = self.randomword(32)
        fake['dataset_id'] = dataset_id
        fake['originalName'] = 'A-fake-originalName.csv'

        if blueprint_id is None:
            blueprint_id = self.randomword(32)
        fake['blueprint_id'] = blueprint_id
        fake['partition_stats'] = {
            str((i, -1)): 'test' for i in range(max_reps)}
        fake['max_reps'] = max_reps
        if max_reps == 5:
            fake['test'] = {metric: [random.random() + scoreoffset,
                                     random.random() + scoreoffset]
                            for metric in ALL_METRICS}
        else:
            fake['test'] = {metric: [random.random() + scoreoffset]
                            for metric in ALL_METRICS}
        fake['test']['metrics'] = [metric for metric in ALL_METRICS]
        fake['max_folds'] = 0
        fake['bp'] = bp
        fake['blend'] = blend
        fake['_id'] = '5223deadbeefdeadbeef{0:04}'.format(bp + bpoffset)
        return fake

    def one_fake_queue_item(self, pid, samplepct=50, total_size=2000,
                            rep=1, bp=0, blueprint_id=None):
        """Describes a job in the queue or in progress.  Rep should be
        in {1, 2, 3, 4, 5}.

        TODO I still need to check out how blends are specified in the queue
        """
        fake = {}
        fake['blueprint'] = {'1': [['NUM'], ['NI'], 'T'],
                             '2': [['1'], ['RFI'], 'P']}
        if blueprint_id is None:
            blueprint_id = self.randomword(32)
        fake['blueprint_id'] = blueprint_id
        fake['bp'] = bp
        fake['dataset_id'] = '52b6121e076329407cb2c88d'
        fake['features'] = [self.randomword(10) for i in xrange(3)]
        fake['icons'] = [1]
        fake['lid'] = None  # how is this already set on a new model?
        fake['max_folds'] = 0
        fake['model_type'] = self.randomword(10)
        fake['new_lid'] = rep == 1
        fake['pid'] = str(pid)
        fake['qid'] = None  # TODO
        fake['reference_model'] = False  # Make this an arg? Care to test?
        fake['samplepct'] = round(float(samplepct) / total_size)
        fake['samplepct'] = samplepct
        fake['total_size'] = total_size
        if rep == 1:
            fake['max_reps'] = 1
        else:
            fake['partitions'] = [[rep - 1, -1]]
            fake['runs'] = rep
            fake['s'] = 1  # What is this?
        return fake

    def produce_leaderboard_items(self,
                                  n=100,
                                  pid='AFakePidString',
                                  fixed_data={},
                                  random_choices={}):
        """Generate ``n`` leaderboard items.  Defaults to creating
        random values for all the items.  Every element in the
        fixed_data dict will overwrite the randomly created values,
        so if you want to fix any number of leaderboard item parameters
        while generating random values for the others, you can do that
        here.

        Sometimes you may want to have your random items choose from
        a set of values instead of being actually random.  You can
        do that by passing a dictionary to random_choices where the
        key is the key in the job (i.e. blueprint, features, etc.)
        and the value is a list of possible alternatives

        Parameters
        ----------
        n : int
            The number of leaderboard examples to generate
        pid : string or ObjectId
            The pid to associate with the leaderboard item
        fixed_data : dict
            The values that will be constant
        random_choices : dict
            Provides a selection for the given keys to randomly assume

        Returns
        -------
        leaderboard : list
            A list of items to put into the leaderboard.

        """
        leaderboard = []
        for i in xrange(n):
            f = self.one_fake_ldb_item(pid=pid, bp=i)
            for k in fixed_data:
                f[k] = fixed_data[k]
            for k in random_choices:
                f[k] = random.choice(random_choices[k])
            leaderboard.append(f)
        return leaderboard

    def fake_leaderboard_item1(self):
        fake = {'blueprint': {'1': (['NUM'], ['LS', 'LR1'], 'P')},
                'features': [],
                'total_size': 10000,
                'test': {'metrics': ['Gini'], 'Gini': [0.16]},
                'train_size': 2000,
                'samplepct': 16,
                'parts': None,
                'vertex_cnt': 1,
                'task_cnt': 2,
                'model_type': 'Imaginary Testing Model 1',
                'test_size': 500,
                'pid': None,
                'uid': 'userid123',
                'dataset_id': '52b6121e076329407cb2c88d',
                'originalName': 'UserSubmittedData.csv',
                'blueprint_id': '4444-eeee-blueid',
                'partition_stats': {str((0, -1)): 'test'},
                'bp': 1,
                'max_folds': 0,
                'max_reps': 1}
        return fake

    def fake_leaderboard_item2(self):
        fake = {'blueprint': {'1': (['NUM'], ['NI', 'RFI'], 'P')},
                'features': [],
                'total_size': 10000,
                'test': {'metrics': ['Gini'], 'Gini': [0.15]},
                'train_size': 2000,
                'samplepct': 1000,
                'parts': None,
                'vertex_cnt': 1,
                'task_cnt': 2,
                'model_type': 'Imaginary Testing Model 2',
                'test_size': 500,
                'pid': None,
                'uid': 'userid123',
                'dataset_id': '52b6121e076329407cb2c88d',
                'originalName': 'UserSubmittedData.csv',
                'blueprint_id': '3333-dddd-blueid',
                'partition_stats': {str((0, -1)): 'test'},
                'bp': 2,
                'max_folds': 0,
                'max_reps': 1
        }
        return fake

    def check_job(self, item):
        self.assertIsInstance(item, dict)
        self.assertGreater(set(item.keys()), set(['blueprint', 'pid']))
        self.assertIsInstance(item['blueprint'], dict)

    def check_joblist(self, joblist):
        for item in joblist:
            self.check_job(item)

    @classmethod
    def randomword(cls, n=10):
        return ''.join([random.choice(string.ascii_lowercase)
                        for i in xrange(n)])

    def default_add_data(self):
        pid = self.make_fake_project()
        self.add_default_test_data(pid)
        return pid

    def add_default_test_data(self, pid):
        f1 = self.fake_leaderboard_item1()
        f1['pid'] = pid
        f2 = self.fake_leaderboard_item2()
        f2['pid'] = pid
        self.persistent.create(values=f1, table='leaderboard')
        self.persistent.create(values=f2, table='leaderboard')
        return

    def add_data(self, pid, items, remember_signatures=True):
        if not isinstance(items, list):
            items = [items]
        for item in items:
            if not isinstance(item, dict):
                self.assertTrue(False,
                                'You\'re using the add_data function wrong')
            self.persistent.create(values=item, table='leaderboard')
        if remember_signatures:
            signatures = [job_signature(item) for item in items]
            self.persistent.update(table='metablueprint',
                                   condition={'pid': pid},
                                   values={'submitted_jobs': signatures})

    def insert_in_queue(self, pid, items, key=QueueKey.QUEUE):
        """Use this to directly insert an item into the redis keyed queues

        The metablueprint will read all of keyed queues when it calls q.get(),
        and the queue service base takes care of adding 'status' to each of
        them (i.e., queue, inprogress, error, onhold)

        Make sure items is a list! Otherwise, I think it will call the reids
        ``set`` command, which will be no bueno
        """
        if key == self.QueueKey.QUEUE:
            values = [json.dumps(item) for item in items]
            self.tempstore.update(keyname=key, index=str(pid), values=values)
        elif key in [self.QueueKey.ONHOLD, self.QueueKey.INPROGRESS, 'errors']:
            # The queue is a list, but the others are hashes.  Roll with it
            values = {item['qid']: json.dumps(item) for item in items}
            self.tempstore.update(keyname=key, index=str(pid), values=values)
        else:
            raise ValueError('The queue type {} is unknown'.format(key))
        return

    def preload_mb_state_data(self, pid, data):
        """Some MBs are written to save state via mongo in between runs.

        pid is an ObjectId
        data is a dict.  Put whatever you want into it
        """
        if 'pid' not in data:
            data['pid'] = pid
        self.persistent.create(table='metablueprint', values=data)

    def make_fake_project(self,
                          dataset_id='52b6121e076329407cb2c88d',
                          task_type='Binary',
                          varTypeString='NNNN', targetname='V0',
                          train_rows=10000, metric='Gini', project_reps=5,
                          holdout=20, pct_min_y=0.01, pct_max_y=0.01,
                          min_y=0, max_y=1, nunique_y=None,
                          weight_col_name=None):
        if nunique_y is None:
            nunique_y = train_rows

        project_document = {
            'target': {'type': task_type, 'name': targetname,
                       'size': train_rows},
            'default_dataset_id': dataset_id,
            'partition': {'reps': project_reps, 'holdout_pct': holdout},
            'holdout_pct': holdout}
        # For project_reps=1 we need a validation set
        if project_reps == 1:
            project_document['partition']['validation_pct'] = 0.16
        if metric is not None:  # This is what the tests were previously
            project_document['metric'] = metric
        if weight_col_name is not None:
            project_document['weights'] = {'weight': weight_col_name}

        pid = self.persistent.create(values=project_document,
                                     table='project')
        metadata_doc = {
            'pid': pid, 'dataset_id': dataset_id,
            'varTypeString': varTypeString,
            'shape': [train_rows, len(varTypeString)],
            'columns': [[i, 'V' + str(i), 0]
                        for i in range(len(varTypeString))],
            'pct_min_y': pct_min_y,
            'pct_max_y': pct_max_y,
            'min_y': min_y,
            'max_y': max_y,
            'nunique_y': nunique_y,
        }

        self.persistent.update(
            condition={"_id": ObjectId(dataset_id)},
            values=metadata_doc,
            table='metadata')
        self.tempstore.create(keyname='queue_settings', index=str(pid),
                              values={'mode': '0', 'parallel': '2'})
        return pid

    def randstring(self, size=10):
        return ''.join(random.choice(string.lowercase) for i in range(size))


@pytest.mark.db
class TestBaseMbJobDuplicates(BaseTestMB):
    """Tests scenarios when the MB attempts to submit multiple copies of the
    same job
    """
    regressors = ['RR', 'LR1', 'RFR', 'SGDR', 'SGDRA', 'SVMR', 'SVRR', 'GBR']

    class ConstructableBaseMB(BaseMetablueprint):
        def next_steps(self, metric):
            plan = {}
            plan['1'] = [['NUM'], ['NI'], 'T']
            plan['2'] = [['1'], ['RFR'], 'P']
            return [{'blueprint': plan,
                     'samplepct': 32,
                     'max_reps': 1,
                     'max_folds': 0,
                     'bp': 1}]

    class DuplicatorMB(BaseMetablueprint):
        def next_steps(self, metric):
            plan = {}
            plan['1'] = [['NUM'], ['NI'], 'T']
            plan['2'] = [['1'], ['RFR'], 'P']
            job_item = {'blueprint': plan,
                        'samplepct': 32,
                        'max_reps': 1,
                        'max_folds': 0,
                        'bp': 1}
            return [job_item, job_item]

    def test_call_records_jobs(self):
        pid = self.make_fake_project()
        mbp = self.ConstructableBaseMB(pid, None)

        #Act
        mbp()

        #Assert
        mb_state = self.persistent.read(table='metablueprint',
                                        condition={'pid': pid},
                                        result={})
        notes = mb_state.get('submitted_jobs', [])
        self.assertEqual(len(notes), 1)

    def test_multiple_calls_do_not_cram_queue(self):
        pid = self.make_fake_project()
        mbp = self.ConstructableBaseMB(pid, None)

        #Act
        mbp()

        # Call it right afterwards to attempt to insert the same job
        mbp()

        #Assert that the second job doesn't make it into the queue
        queue = self.tempstore.read(keyname='queue', index=str(pid),
                                    result=[])
        self.assertEqual(1, len(queue))

    def test_mbs_that_submit_same_job_twice_in_round_only_one_gets_to_queue(
            self):
        pid = self.make_fake_project()
        mbp = self.DuplicatorMB(pid, None)

        #Act
        mbp()

        #Assert that the second job doesn't make it into the queue
        queue = self.tempstore.read(keyname='queue', index=str(pid),
                                    result=[])
        self.assertEqual(1, len(queue))


    def test_running_on_empty_joblist_doesnot_crash(self):
        class DumbMB(BaseMetablueprint):
            def next_steps(self, metric):
                return []

        pid = self.make_fake_project()
        mb = DumbMB(pid, None)

        mb()
        #Passes if no error


@pytest.mark.db
class TestBaseMb(BaseTestMB):
    """
    Tests for base_metablueprint
    """

    def test_manual_get_metadata(self):
        check = self.persistent.read(table='metadata', result={})
        self.assertEqual(check, {})
        pid = self.make_fake_project()
        metadata = self.persistent.read(condition={'pid': pid},
                                        table='metadata', result={})
        self.assertIsInstance(metadata, dict)
        expected_keys = {'_id', 'pid', 'dataset_id', 'varTypeString', 'shape',
                         'columns', 'nunique_y', 'min_y', 'max_y', 'pct_min_y',
                         'pct_max_y'}
        self.assertEqual(set(metadata.keys()), expected_keys)
        mbp = self.ConstructableBaseMB(pid, None)
        self.assertEqual(mbp._metadata, {})
        check = mbp.metadata
        self.assertEqual(check, metadata)

    def test_support_functions(self):
        dataset_id = '52b6121e076329407cb2c88a'
        pid = self.make_fake_project(dataset_id=dataset_id)
        fixed_data = {'dataset_id': dataset_id}
        ldb_items = self.produce_leaderboard_items(20, pid=pid,
                                                   fixed_data=fixed_data)
        self.add_data(pid, ldb_items)
        db_ldb = self.persistent.read(condition={'pid': pid},
                                      table='leaderboard', limit=(0, 0),
                                      result=[])
        self.assertEqual(len(db_ldb), len(ldb_items))
        mbp = self.ConstructableBaseMB(pid, None)
        ldb = mbp.leaderboard
        self.assertEqual(len(ldb), len(ldb_items))

    def test_get_project(self):
        check = self.persistent.read(table='project')
        self.assertEqual(check, '')
        pid = self.default_add_data()
        project = self.persistent.read(condition={'_id': pid}, table='project',
                                       result={})
        self.assertIsInstance(project, dict)
        self.assertEqual(set(project.keys()), set(
            ['default_dataset_id', 'target', '_id', 'holdout_pct', 'partition',
             'metric']))
        mbp = self.ConstructableBaseMB(pid, None)
        self.assertIsNone(mbp._project)
        check = mbp.project
        for key in project:
            self.assertEqual(check[key], project[key])
        self.assertIn('metric', check)

    def test_get_metadata(self):
        check = self.persistent.read(table='metadata', result={})
        self.assertEqual(check, {})
        pid = self.default_add_data()
        metadata = self.persistent.read(condition={'pid': pid},
                                        table='metadata', result={})
        self.assertIsInstance(metadata, dict)
        expected_keys = set(['_id', 'pid', 'dataset_id', 'varTypeString',
                             'shape', 'columns', 'nunique_y', 'min_y',
                             'max_y', 'pct_min_y', 'pct_max_y'])
        self.assertEqual(set(metadata.keys()), expected_keys)
        mbp = self.ConstructableBaseMB(pid, None)
        self.assertEqual(mbp._metadata, {})
        check = mbp.metadata
        self.assertEqual(check, metadata)

    def test_add_to_queue_dupes_are_not_accepted(self):
        pid = self.default_add_data()
        mbp = self.ConstructableBaseMB(pid, None)
        joblist = self.produce_leaderboard_items(n=25, pid=pid)
        mbp.add_to_queue(joblist)
        self.assertEqual(
            int(self.tempstore.read(keyname='qid_counter', index=str(pid))),
            len(joblist))
        mbp.add_to_queue(joblist)
        check = self.tempstore.read(keyname='queue', index=str(pid), result=[])
        self.assertEqual(
            int(self.tempstore.read(keyname='qid_counter', index=str(pid))),
            len(joblist))
        self.assertEqual(len(check), len(joblist))
        for n, item in enumerate(check):
            self.assertEqual(item['qid'], n + 1)

    def test_add_to_queue_multiple_calls_still_keep_adding_jobs(self):
        pid = self.default_add_data()
        mbp = self.ConstructableBaseMB(pid, None)
        joblist = self.produce_leaderboard_items(n=10, pid=pid)
        mbp.add_to_queue(joblist)
        self.assertEqual(
            int(self.tempstore.read(keyname='qid_counter', index=str(pid))),
            len(joblist))
        joblist2 = self.produce_leaderboard_items(n=10, pid=pid)
        mbp.add_to_queue(joblist2)
        check = self.tempstore.read(keyname='queue', index=str(pid), result=[])
        self.assertEqual(
            int(self.tempstore.read(keyname='qid_counter', index=str(pid))),
            len(joblist) * 2)
        self.assertEqual(len(check), 2 * len(joblist))
        for n, item in enumerate(check):
            self.assertEqual(item['qid'], n + 1)

    @pytest.mark.db
    def test_call_generates_diagrams(self):
        """Generating diagrams will now be something the metablueprint does
        when creating jobs.  It is a side effect of creating the menu of
        jobs.  The value stored inside 'diagram' is a json that the front
        end can use to draw the blueprint diagram
        """

        blueprint = {'1': [['NUM'], ['NI'], 'T'],
                     '2': [['1'], ['RFC'], 'P']}

        class PredictableMB(BaseMetablueprint):
            def next_steps(self, metric):
                return [{'blueprint': blueprint,
                         'samplepct': 50,
                         'max_reps': 1,
                         'max_folds': 0,
                         'bp': 1}]

        pid = self.make_fake_project()
        uid = ObjectId()
        mb = PredictableMB(pid, uid)

        mb()

        blueprint_id = basemb_module.blueprint_module.blueprint_id(blueprint)
        stored_data = mb._data['menu'][blueprint_id]
        self.assertIn('diagram', stored_data)
        json.loads(stored_data['diagram'])

    def test_add_to_menu(self):
        pid = self.default_add_data()
        mbp = self.ConstructableBaseMB(pid, None)
        mbp.add_to_menu([{'blueprint': 'a', 'blueprint_id': 1}])
        self.assertEqual(len(mbp._data['menu']), 1)
        mbp.add_to_menu([{'blueprint': 'a', 'blueprint_id': 1}])
        self.assertEqual(len(mbp._data['menu']), 1)
        mbp.add_to_menu([{'blueprint': 'b', 'blueprint_id': 2}])
        self.assertEqual(len(mbp._data['menu']), 2)

    def test_sort_1(self):
        pid = self.make_fake_project()
        f1 = self.fake_leaderboard_item1()
        f2 = self.fake_leaderboard_item2()
        f1['pid'] = pid
        f2['pid'] = pid
        f1['test']['metrics'] = ['Gini', 'LogLoss']
        f1['blend'] = 2
        f1['test']['LogLoss'] = [0.5]
        f1['test']['Gini'] = [0.1]
        f2['test']['metrics'] = ['Gini', 'LogLoss']
        f2['test']['LogLoss'] = [0.3]
        f2['test']['Gini'] = [0.3]
        f2['max_reps'] = 5
        f2['blend'] = 1
        self.add_data(pid, [f1, f2])
        mbp = self.ConstructableBaseMB(pid, None)
        ldb = mbp.sorted_leaderboard('Gini')
        self.assertTrue(ldb[0]['blend'] < ldb[1]['blend'],
                        'Higher reps should always outrank lower reps')

    def test_sort_2(self):
        pid = self.make_fake_project()
        f1 = self.fake_leaderboard_item1()
        f2 = self.fake_leaderboard_item2()
        f1['pid'] = pid
        f2['pid'] = pid
        f1['test']['metrics'] = ['Gini', 'LogLoss']
        f1['test']['LogLoss'] = [0.5]
        f1['test']['Gini'] = [0.1]
        f2['test']['metrics'] = ['Gini', 'LogLoss']
        f2['test']['LogLoss'] = [0.3]
        f2['test']['Gini'] = [0.3]
        f2['samplepct'] = 2 * f1['samplepct']
        f1['blend'] = 2
        f2['blend'] = 1
        self.add_data(pid, [f1, f2])
        mbp = self.ConstructableBaseMB(pid, None)
        ldb = mbp.sorted_leaderboard('Gini')
        self.assertTrue(ldb[0]['blend'] < ldb[1]['blend'],
                        'Higher samplepct should always outrank smaller')

    def test_sort_3(self):
        pid = self.make_fake_project()
        f1 = self.fake_leaderboard_item1()
        f2 = self.fake_leaderboard_item2()
        f1['pid'] = pid
        f2['pid'] = pid
        f1['test']['metrics'] = ['AUC', 'LogLoss']
        f1['test']['LogLoss'] = [0.5]
        f1['test']['AUC'] = [0.1]
        f2['test']['metrics'] = ['AUC', 'LogLoss']
        f2['test']['LogLoss'] = [0.3]
        f2['test']['AUC'] = [0.3]
        f1['blend'] = 2
        f2['blend'] = 1
        self.add_data(pid, [f1, f2])
        mbp = self.ConstructableBaseMB(pid, None)
        ldb = mbp.sorted_leaderboard('AUC')
        #print ldb
        self.assertTrue(ldb[0]['blend'] < ldb[1]['blend'],
                        'Sorting AUC should favor higher scores')

    def test_sort_4(self):
        pid = self.make_fake_project()
        f1 = self.fake_leaderboard_item1()
        f2 = self.fake_leaderboard_item2()
        f1['pid'] = pid
        f2['pid'] = pid
        f1['test']['metrics'] = ['Gini', 'LogLoss']
        f1['test']['LogLoss'] = [0.5]
        f1['test']['Gini'] = [0.1]
        f2['test']['metrics'] = ['Gini', 'LogLoss']
        f2['test']['LogLoss'] = [0.3]
        f2['test']['Gini'] = [0.3]
        f2['samplepct'] = 2 * f1['samplepct']
        f1['blend'] = 2
        f2['blend'] = 1
        self.add_data(pid, [f1, f2])
        mbp = self.ConstructableBaseMB(pid, None)
        ldb = mbp.sorted_leaderboard('Gini')
        self.assertTrue(ldb[0]['blend'] < ldb[1]['blend'],
                        'Sorting LogLoss should favor lower scores')

    def test_sort_5(self):
        pid = self.make_fake_project()
        models = []
        for i in xrange(100):
            f = self.fake_leaderboard_item1()
            f['pid'] = pid
            f['test']['Gini'] = [random.random()]
            f['blueprint_id'] = self.randstring(10)
            f['blueprint'] = {'1': random.randint(1, 1000)}
            models.append(f)
        self.add_data(pid, models)
        mbp = self.ConstructableBaseMB(pid, None)
        ldb = mbp.sorted_leaderboard('Gini')
        prior = 10E9
        for l in ldb:
            self.assertLessEqual(l['test']['Gini'][0], prior)
            prior = l['test']['Gini'][0]
        for i in xrange(len(ldb) - 1):
            self.assertTrue(
                ldb[i]['test']['Gini'] > ldb[i + 1]['test']['Gini'],
                'Sorting by metric does not put the best models on top')

    def test_get_leaderboard(self):
        pid = self.make_fake_project()
        models = []
        for i in xrange(100):
            f = self.fake_leaderboard_item1()
            f['pid'] = pid
            f['test']['Gini'] = [random.random()]
            f['blueprint_id'] = ''.join(
                random.choice(string.lowercase) for i in range(10))
            f['blueprint'] = {'1': random.randint(1, 1000)}
            models.append(f)
        self.add_data(pid, models)
        mbp = self.ConstructableBaseMB(pid, None)
        ldb = mbp.sorted_leaderboard('Gini')
        self.assertEqual(len(models),
                         len(ldb),
                         'Not all models are coming back '
                         '(returned %d, expected %d'
                         % (len(ldb), len(models)))

    def test_sorted_leaderbaord(self):
        pid = self.make_fake_project()
        models = []
        for i in range(12):
            f = self.fake_leaderboard_item1()
            f['pid'] = pid
            f['test']['Gini'] = [i / 12.0]
            f['blueprint_id'] = 'asdfqwerzxcv'[i]
            models.append(f)
        self.add_data(pid, models)
        mbp = self.ConstructableBaseMB(pid, None)
        top = mbp.sorted_leaderboard('Gini')
        check = ''
        check_ss = 0
        check_reps = 0
        for i in top:
            check += i['blueprint_id']
            check_ss += i['samplepct']
            check_reps += i['max_reps']
        self.assertEqual(check_ss / len(top), 16)
        self.assertEqual(check_reps, len(top))
        self.assertEqual(check, 'vcxzrewqfdsa')

    def test_leaderboard_sort_with_errored_jobs(self):
        pid = self.make_fake_project()
        models = []
        for i in range(12):
            f = self.fake_leaderboard_item1()
            f['pid'] = pid
            f['test']['Gini'] = [i / 12.0]
            models.append(f)
        f2 = self.fake_leaderboard_item1()
        f2['pid'] = pid

        # These lines are the conditions we see for failed jobs
        for metric in f2['test']['metrics']:
            f2['test'][metric] = [None]
        f2['no_finish'] = 'Errored'

        models.append(f2)
        self.add_data(pid, models)
        mbp = self.ConstructableBaseMB(pid, None)
        top = mbp.sorted_leaderboard('Gini')

    def test_leaderboard_sort_with_missing_metric(self):
        """ Skip models where the metric does not exist
        """
        pid = self.make_fake_project()
        models = []
        for i in range(12):
            f = self.fake_leaderboard_item1()
            f['pid'] = pid
            f['test']['RMSLE'] = [i / 12.0]
            f['test']['Gini'] = [i / 12.0]
            models.append(f)
        f2 = self.fake_leaderboard_item1()
        # Create one without RMSLE
        f2['pid'] = pid
        f2['test']['Gini'] = [1]

        models.append(f2)
        self.add_data(pid, models)
        mbp = self.ConstructableBaseMB(pid, None)
        top = mbp.sorted_leaderboard('RMSLE')

    def test_top_models(self):
        pid = self.make_fake_project()
        models = []
        for i in range(12):
            f = self.fake_leaderboard_item1()
            f['pid'] = pid
            f['test']['Gini'] = [i / 12.0]
            f['blueprint_id'] = 'asdfqwerzxcv'[i]
            models.append(f)
        self.add_data(pid, models)
        mbp = self.ConstructableBaseMB(pid, None)
        top = mbp.top_models('Gini')
        check = ''
        check_ss = 0
        check_reps = 0
        for i in top:
            check += i['blueprint_id']
            check_ss += i['samplepct']
            check_reps += i['max_reps']
        self.assertEqual(check_ss / len(top), 16)
        self.assertEqual(check_reps, len(top))
        self.assertEqual(check, 'vcxzrewqfdsa')

    def test_top_models_errored_items(self):
        """ Test to make sure that autopilot does not fail
        if a model errors
        """
        pid = self.make_fake_project()
        models = self.produce_leaderboard_items(n=11, pid='11')
        f = self.fake_leaderboard_item1()
        f['pid'] = pid
        del (f['test'])
        f['blueprint_id'] = 'v'
        models.append(f)
        f = self.fake_leaderboard_item1()
        f['pid'] = pid
        f['test'] = None
        f['blueprint_id'] = 't'
        models.append(f)
        self.add_data(pid, models)
        mbp = self.ConstructableBaseMB(pid, None)
        top = mbp.top_models('Gini')

    def test_add_metadata(self):
        #test base class
        pid = self.make_fake_project()
        mb = self.ConstructableBaseMB(pid, None)
        self.assertIsInstance(mb.computeMetadata(None), dict)
        out = mb.persistent.read({'pid': pid, 'dataset_id': mb.dataset_id},
                                 table='metadata', result={})
        self.assertIsInstance(out, dict)
        self.assertGreater(set(out.keys()), {'pid', 'dataset_id'})
        mb.addMetadata(None)
        out2 = mb.persistent.read({'pid': pid, 'dataset_id': mb.dataset_id},
                                  table='metadata', result={})
        self.assertEqual(out, out2)

        #test with non-trivial computeMetadata function
        class newMB(self.ConstructableBaseMB):
            def computeMetadata(self, ds):
                return {'newtestkey': 'newtestvalue'}

        pid = self.make_fake_project()
        newmb = newMB(pid, None)
        out0 = newmb.persistent.read({'pid': pid, 'dataset_id': mb.dataset_id},
                                     table='metadata', result={})
        self.assertGreater(set(out.keys()), {'pid', 'dataset_id'})
        newmb.addMetadata(None)
        out = newmb.persistent.read({'pid': pid, 'dataset_id': mb.dataset_id},
                                    table='metadata', result={})
        out0.update({'newtestkey': 'newtestvalue'})
        self.assertEqual(out0, out)

    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    def test_eda(self, m):
        pid = self.make_fake_project()
        uid = '52b6121e076329407cb2c88d'
        mb = self.ConstructableBaseMB(pid, uid)
        es = common.services.eda.EdaService(pid, uid)
        es.update({'testkey': {'ss': 'testvalue'}})
        out = mb.eda
        self.assertIsInstance(out, list)
        self.assertEqual(set(out[0].keys()), {'ss', 'id'})

    def test_get_metric_backwards_compatible(self):
        pid = self.make_fake_project()
        mb = self.ConstructableBaseMB(pid, None, reference_models=False)
        self.assertIsNotNone(mb.project['metric'])

    def test_get_computed_models(self):
        pid = self.make_fake_project()

        models = []
        for i in range(6):
            f = self.fake_leaderboard_item1()
            f['pid'] = pid
            f['test']['Gini'] = [i / 12.0]
            f['blueprint_id'] = 'asdfqwerzxcv'[i]
            models.append(f)
        self.add_data(pid, models[:3])

        mb = self.ConstructableBaseMB(pid, None, reference_models=True)
        mb.add_to_queue(models[3:])
        out = mb.computed_models

        self.assertIsInstance(out, set)
        self.assertEqual(len(out), 6)
        for i in out:
            self.assertIsInstance(i, tuple)
            self.assertEqual(len(i), 4)
            self.assertIn(i[0], [i['blueprint_id'] for i in models])

    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    @patch('common.services.eda.assert_valid_id')
    def test_eda_knows_only_one_eda_per_pid(self, *args):
        """Formerly, we had planned to support multiple datasets by having
        more than one EDA per project, where the dataset_id would identify
        which eda to fetch.  Now there is only one EDA document per pid, with
        additional datasets being added to that document (I believe).

        """
        pid = self.make_fake_project()
        uid = '52b6121e076329407cb2c88c'

        # n.b., if the layout of the EDA collection changes, so should this var
        some_eda = {'dataset_id': 'universe',
                    'pid': pid,
                    'eda': {'SomeColumn': {},
                            'AnotherColumn': {}}}
        es = common.services.eda.EdaService(pid, uid, 'universe')
        es.update(some_eda['eda'])
        mbp = self.ConstructableBaseMB(pid, None, "universe")
        eda = mbp.eda
        # expected output is a list with one element per column
        # (len=2 in this test case)
        self.assertEqual(len(eda), 2)

    def test_add_to_queue_with_multiple_subblueprints(self):
        bps = []
        bps += [{'1': (['NUM'], ['NI'], 'T'),
                 '2': (['1'], ['RFC 50;5'], 'T'),
                 '3': (['2'], ['GLMB'], 'P')}]
        bps += [{'1': (['NUM'], ['NI'], 'T'),
                 '2': (['1'], ['RFC 50;5'], 'T'),
                 '3': (['TXT'], ['TM2'], 'T'),
                 '4': (['3'], ['LR1'], 'S'),
                 '5': (['2', '4'], ['GLMB'], 'P')}]
        pid = self.default_add_data()
        mbp = self.ConstructableBaseMB(pid, None)
        joblist = []
        for bp in bps:
            item = {'max_reps': 1, 'samplepct': 50, 'blueprint': bp}
            joblist.append(item)

        mbp.add_to_queue(joblist, subblueprints=True)
        self.assertEqual(
            int(self.tempstore.read(keyname='qid_counter', index=str(pid))), 4)
        check = self.tempstore.read(keyname='queue', index=str(pid), result=[])
        self.assertEqual(check[0].get('require'), None)
        self.assertEqual(check[1].get('require'), ['1'])
        self.assertEqual(check[2].get('require'), None)
        self.assertEqual(check[3].get('require'), ['1', '3'])

    def test_add_to_queue_with_one_sub_should_add_subblueprint(self):
        bps = []
        bps += [{'1': (['NUM'], ['NI'], 'T'),
                 '2': (['1'], ['RFC 50;5'], 'T'),
                 '3': (['2'], ['GLMB'], 'P')}]
        pid = self.default_add_data()
        mbp = self.ConstructableBaseMB(pid, None)
        joblist = []
        for bp in bps:
            item = {'max_reps': 1, 'samplepct': 50, 'blueprint': bp}
            joblist.append(item)

        mbp.add_to_queue(joblist, subblueprints=True)
        self.assertEqual(
            int(self.tempstore.read(keyname='qid_counter', index=str(pid))), 2)
        check = self.tempstore.read(keyname='queue', index=str(pid), result=[])
        self.assertEqual(check[0].get('require'), None)
        self.assertEqual(check[1].get('require'), ['1'])

    def test_add_to_queue_with_two_subs_should_add_subblueprint(self):
        bps = []
        bps += [{'1': (['NUM'], ['NI'], 'T'),
                 '2': (['1'], ['RFC 50;5'], 'T'),
                 '3': (['TXT'], ['TM2'], 'T'),
                 '4': (['3'], ['LR1'], 'S'),
                 '5': (['2', '4'], ['GLMB'], 'P')}]
        pid = self.default_add_data()
        mbp = self.ConstructableBaseMB(pid, None)
        joblist = []
        for bp in bps:
            item = {'max_reps': 1, 'samplepct': 50, 'blueprint': bp}
            joblist.append(item)

        mbp.add_to_queue(joblist, subblueprints=True)
        self.assertEqual(
            int(self.tempstore.read(keyname='qid_counter', index=str(pid))), 3)
        check = self.tempstore.read(keyname='queue', index=str(pid), result=[])
        self.assertEqual(check[0].get('require'), None)
        self.assertEqual(check[1].get('require'), None)
        self.assertEqual(check[2].get('require'), ['1', '2'])

    def test_get_preliminary_jobs_one_detection(self):
        bp = {'1': (['NUM'], ['NI'], 'T'),
              '2': (['1'], ['RFC 50;5'], 'T'),
              '3': (['2'], ['GLMB'], 'P')}
        pid = self.default_add_data()
        mbp = self.ConstructableBaseMB(pid, None)
        jobitem = {'max_reps': 1, 'samplepct': 50, 'blueprint': bp}
        out = mbp.get_preliminary_jobs(mbp.prepackage_job(jobitem), False)
        self.assertEqual(len(out), 1)

    def test_get_preliminary_jobs_two_detections(self):
        bp = {'1': (['NUM'], ['NI'], 'T'),
              '2': (['1'], ['RFC 50;5'], 'T'),
              '3': (['TXT'], ['TM2'], 'T'),
              '4': (['3'], ['LR1'], 'S'),
              '5': (['2', '4'], ['GLMB'], 'P')}
        pid = self.default_add_data()
        mbp = self.ConstructableBaseMB(pid, None)
        jobitem = {'max_reps': 1, 'samplepct': 50, 'blueprint': bp}
        out = mbp.get_preliminary_jobs(mbp.prepackage_job(jobitem), False)
        self.assertEqual(len(out), 2)

    def test_prepackage_includes_target_size(self):
        pid = self.make_fake_project()
        mb = self.ConstructableBaseMB(pid, None, reference_models=False)
        fake_job_desc = self.one_fake_ldb_item(pid)

        # jobs that a MBP writer generates are not responsible for generating
        # the 'total_size' field.  That (probably) is put into mongo by the
        # "report" method of the worker
        del fake_job_desc['total_size']
        packaged = mb.prepackage_job(fake_job_desc)

        self.assertIn('total_size', packaged)

    @pytest.mark.integration
    def test_logy_transform_is_in_features_not_title(self):
        pid = self.make_fake_project()
        mb = self.ConstructableBaseMB(pid, ObjectId())

        blueprint = {'1': [['NUM'], ['NI'], 'T'],
                     '2': [['1'], ['RFC logy'], 'P']}

        blueprint_id = blueprint_module.blueprint_id(blueprint)
        job = dict(blueprint=blueprint, blueprint_id=blueprint_id)

        packaged = mb.prepackage_job(job)
        self.assertNotIn('Log Transformed Response', packaged['model_type'])
        self.assertIn('Log Transformed Response', packaged['features'])

    def test_logy_transform_is_in_features_even_if_not_in_stored_features(
            self):
        """We do some lying about which features are in a blueprint in order
        to retain some of our secret sauce.  We will now make sure that
        'Log Transformed Response' shows up in the features list regardless
        of which graybox features have been generated"""
        pid = self.make_fake_project()
        mb = self.ConstructableBaseMB(pid, ObjectId())

        blueprint = {'1': [['NUM'], ['NI'], 'T'],
                     '2': [['1'], ['RFC logy'], 'P']}

        blueprint_id = blueprint_module.blueprint_id(blueprint)
        job = dict(blueprint=blueprint, blueprint_id=blueprint_id,
                   features=['No one', 'cares'])

        packaged = mb.prepackage_job(job)
        self.assertNotIn('Log Transformed Response', packaged['model_type'])
        self.assertIn('Log Transformed Response', packaged['features'])

    def test_text_mining_appends_name_of_col_to_model_label(self):
        """ Test for the code to display the variable name for the new text
        mining insight tasks
        """
        pid = self.make_fake_project(varTypeString='NNTT')
        mb = self.ConstructableBaseMB(pid, ObjectId())

        blueprint = {'1': [['NUM'], ['SCTXT cn=0'], 'T'],
                     '2': [['1'], ['WNGR'], 'P']}

        blueprint_id = blueprint_module.blueprint_id(blueprint)
        job = dict(blueprint=blueprint, blueprint_id=blueprint_id)

        packaged = mb.prepackage_job(job)
        self.assertIn('Auto-Tuned Word N-Gram Text Modeler - V2',
                      packaged['features'])
        self.assertEqual('Auto-Tuned Word N-Gram Text Modeler - V2',
                         packaged['model_type'])

    def test_text_mining_appends_name_of_col_too_model_label_cn_1(self):
        """ Test for the code to display the variable name for the new text
        mining insight tasks
        """
        pid = self.make_fake_project(varTypeString='NNTT')
        mb = self.ConstructableBaseMB(pid, ObjectId())

        blueprint = {'1': [['NUM'], ['SCTXT cn=1'], 'T'],
                     '2': [['1'], ['WNGR'], 'P']}

        blueprint_id = blueprint_module.blueprint_id(blueprint)
        job = dict(blueprint=blueprint, blueprint_id=blueprint_id)

        packaged = mb.prepackage_job(job)
        self.assertIn('Auto-Tuned Word N-Gram Text Modeler - V3',
                      packaged['features'])
        self.assertEqual('Auto-Tuned Word N-Gram Text Modeler - V3',
                         packaged['model_type'])

    @pytest.mark.unit
    def test_label_bp_features_text_NUM_NI_RFC(self):
        """Test that label_bp returns the descriptive textual name for the
        bp NI, RFC
        """
        pid = self.make_fake_project()
        mb = self.ConstructableBaseMB(pid, None, reference_models=False)
        blueprint = {'1': [['NUM'], ['NI'], 'T'],
                     '2': [['1'], ['RFC e=0;c=1'], 'P']}
        out = mb.label_bp(blueprint, 1)
        self.assertEqual(out['features_text'],
                         'Transformed numeric data using Missing Values Imputed, predicted the response using a RandomForest Classifier (Entropy).')

    @pytest.mark.skip('Feature not complete')
    def test_label_bp_features_text_TXT_TF_IDF_SGDR(self):
        """Test that label_bp returns the descriptive textual name for the
        bp TXT, TM2, SGDR
        """
        pid = self.make_fake_project()
        mb = self.ConstructableBaseMB(pid, None, reference_models=False)
        blueprint = {'1': [['TXT'], ['TM2'], 'T'],
                     '2': [['1'], ['SGDR'], 'T'],
                     '3': [['2'], ['RDT2'], 'T'],
                     '4': [['3'], ['SGDR'], 'P']}
        out = mb.label_bp(blueprint, 1)
        self.assertEqual(out['features_text'],
                         'Transformed text data using TF-IDF Text Transform '
                         'and Ridit transform,  performed variable selection '
                         'using Stochastic Gradient Descent Regressor, '
                         'predicted the response using Stochastic Gradient '
                         'Descent Regressor.')

    @pytest.mark.unit
    def test_get_textual_description_blueprint(self):
        """Test that get_textual_description_blueprint returns a proper description
        """
        pid = self.make_fake_project()
        mb = self.ConstructableBaseMB(pid, None, reference_models=False)
        blueprint = {'1': [['NUM'], ['NI'], 'T'],
                     '2': [['1'], ['RFC e=0;c=1'], 'P']}
        feats = ['Missing Values Imputed', 'RandomForest Classifier (Entropy)']
        out = mb.get_textual_description_blueprint(feats, blueprint)
        self.assertEqual(out,
                         'Transformed numeric data using Missing Values '
                         'Imputed, predicted the response using a RandomForest'
                         ' Classifier (Entropy).')

    @pytest.mark.unit
    def test_get_task_role_single_branch(self):
        """ Test that get_task_role correctly identifies
        the roles in a single branch
        """
        pid = self.make_fake_project()
        mb = self.ConstructableBaseMB(pid, None, reference_models=False)
        branch = {'1': {'root': ['NUM'], 'node': 'response',
                        'tasks': ['1', '2']}}
        role1 = mb.get_task_role('1', branch['1'])
        self.assertEqual(role1, 't')
        role2 = mb.get_task_role('2', branch['1'])
        self.assertEqual(role2, 'p')

    @pytest.mark.unit
    def test_get_task_role_multiple_branch(self):
        """ Test that get_task_role correctly identifies the roles in a
        single branch
        """
        pid = self.make_fake_project()
        mb = self.ConstructableBaseMB(pid, None, reference_models=False)
        branch = {'1': {'root': ['NUM'], 'node': '3',
                        'tasks': ['1']},
                  '2': {'root': ['CAT'], 'node': '3',
                        'tasks': ['2']},
                  '3': {'root': ['1', '2'], 'node': 'response',
                        'tasks': ['3', '4']}}
        role1 = mb.get_task_role('1', branch['1'], )
        self.assertEqual(role1, 't')
        role2 = mb.get_task_role('2', branch['2'])
        self.assertEqual(role2, 't')
        role3 = mb.get_task_role('3', branch['3'], 'ELNR')  # Test a model
        self.assertEqual(role3, 'vs')
        role4 = mb.get_task_role('4', branch['3'])
        self.assertEqual(role4, 'p')

    @pytest.mark.unit
    def test_get_task_text_transformer(self):
        """Test that get_task_text works properly for a transformer
        """
        pid = self.make_fake_project()
        mb = self.ConstructableBaseMB(pid, None, reference_models=True)
        out = mb.get_task_text('FooBar', 't', ['NUM'])
        self.assertEqual(out, 'transformed numeric data using FooBar')

    @pytest.mark.unit
    def test_get_task_text_predictor(self):
        """Test that get_task_text works properly for a predictor
        """
        pid = self.make_fake_project()
        mb = self.ConstructableBaseMB(pid, None, reference_models=True)
        out = mb.get_task_text('FooBar', 'p')
        self.assertEqual(out, 'predicted the response using a FooBar')

    @pytest.mark.unit
    def test_get_task_text_variable_selector(self):
        """Test that get_task_text works properly for a variable selector
        """
        pid = self.make_fake_project()
        mb = self.ConstructableBaseMB(pid, None, reference_models=True)
        out = mb.get_task_text('FooBar', 'vs')
        self.assertEqual(out, 'selected variables using a FooBar')

    def test_mb_call_submits_blends_correctly(self):
        """Going to change the way that blends are handled, having the
        __call__ method be in charge of submitting them, and the next_steps
        in charge of creating them
        """

        class BlenderMB(BaseMetablueprint):
            def next_steps(self, metric):
                return [{'blender_items': ['fake_blend_ids'],
                         'blender_method': 'GLM',
                         'blender_args': ['logitx'],
                         'blender_family': 'binomial',
                         'blender_tweedie_p': 1.5}]

        pid = self.make_fake_project()
        mb = BlenderMB(pid, None)

        with patch.object(mb, 'add_blends_to_queue') as fake_q_blender:
            mb()
            self.assertTrue(fake_q_blender.called)

    def test_none_jobs_doesnt_break_metablueprint(self):
        """Github issue #1435.  Sometimes next_steps would return ``None``
        instead of empty list, which was breaking things
        """

        class NoneMB(BaseMetablueprint):
            def next_steps(self, metric):
                return None

        pid = self.make_fake_project()
        mb = NoneMB(pid, None)

        mb()

    def test_testing_helper_function_puts_mb_state_data(self):
        pid = self.make_fake_project()
        state_data = {'hidden': 'secrets'}
        self.preload_mb_state_data(pid, state_data)

        mb = self.ConstructableBaseMB(pid, None)
        self.assertIn('hidden', mb._data)
        self.assertEqual(mb._data['hidden'], 'secrets')


class TestMetablueprintMenuCreation(BaseTestMB):
    """So, in addition to the default manner of generating jobs (which is
    through next_steps), we have created an alternate method of creating
    jobs that does _not_ automatically get fed into the autopilot.  This
    is for computationally long tasks, like neural networks, that kind of
    impede the normal user experience but which are models that we would
    like our users to be able to easily compare
    """

    class ManualOnlyMB(BaseMetablueprint):
        def next_steps(self, metric):
            return []

        def create_menu_jobs(self):
            jobs = [
                {'1': [['NUM'], ['NI'], 'T'],
                 '2': [['1'], [task], 'P']} for task in ['RFC', 'LR1', 'SVMC']
            ]
            return jobs

    @pytest.mark.db
    def test_generate_menu(self):
        pid = self.make_fake_project()
        mb = self.ManualOnlyMB(pid, None)

        mb.generate_menu()

        data = self.persistent.read(table='metablueprint',
                                    condition={'pid': pid},
                                    result={})
        menu = data['menu']
        self.assertEqual(3, len(menu))
        for job in menu.values():
            self.assertIn('diagram', job)
            self.assertIn('blueprint', job)
            self.assertIn('blueprint_id', job)


class TestManualModeBehavior(BaseTestMB):
    class SimpleMB(BaseMetablueprint):
        def next_steps(self, metric):
            return [{'blueprint': {
                '1': [['NUM'], ['NI'], 'T'],
                '2': [['1'], ['RFC'], 'P']},
                     'samplepct': 64,
                     'max_reps': 1,
                     'max_folds': 0,
                     'dataset_id': 'AFakeDatasetId'}]

    @pytest.mark.db
    def test_manual_mode_doesnt_remember_jobs_as_submitted(self):
        pid = self.make_fake_project()
        # Set the mode to manual, somehow
        mb = self.SimpleMB(pid, None)

        mb._settings = {'mode': basemb_module.autopilot.MANUAL,
                        'dataset_id': 'AFakeDatasetId'}
        with patch.object(mb, '_metadata') as fake_metadata:
            mb()

        self.assertEqual(len(mb._data['submitted_jobs']), 0)


class MBAssumesNothing(BaseMetablueprint):
    def next_steps(self, metric):
        plan = {}
        plan['1'] = [['NUM'], ['NI'], 'T']
        plan['2'] = [['1'], ['RFR'], 'P']
        return [{'blueprint': plan,
                 'samplepct': 32,
                 'max_reps': 1,
                 'max_folds': 0,
                 'bp': 1}]


class MBAssumesStorage(MBAssumesNothing, rmm.DefaultReferenceModelsMixin):
    def __init__(self, *args, **kwargs):
        super(MBAssumesStorage, self).__init__(*args, **kwargs)
        self._data['flags'] = {
            basemb_module.MBFlags.SUBMITTED_JOBS_STORED: True
        }

    def get_recommended_metrics(self):
        return {'default': {'short_name': 'LogLoss'},
                'recommender': {'short_name': 'AUC'},
                'weighted': {'short_name': 'Weighted LogLoss'},
                'weight+rec': {'short_name': 'Weighted AUC'}}


class TestMetablueprintJobMemories(BaseTestMB):
    """This feature of making the metablueprint only recognize jobs that it
    'remembers' submitting can (did) cause backwards compatibility problems.

    These tests are to cover the various scenarios of pre and post
    recognizing jobs

    """

    BLENDER_ITEM_FIXTURE = {
        u'_id': ObjectId('538d00018bd88f02c9209b6c'),
        u'blend': 0,
        u'blender': {u'inputs': [{u'blender': {},
                                  u'blueprint': {
                                      u'1': [[u'NUM'], [u'GS'], u'T'],
                                      u'2': [[u'CAT'], [u'DM2 sc=25;cm=500'],
                                             u'T'],
                                      u'3': [[u'1', u'2'], [u'LR1 p=1'],
                                             u'S']},
                                  u'dataset_id': u'538cf2268bd88f21f07b6517',
                                  u'samplepct': 64},
                                 {u'blender': {},
                                  u'blueprint': {
                                      u'1': [[u'CAT'], [u'CRED1b1 cmin=1'],
                                             u'TS'],
                                      u'2': [[u'CAT'], [u'CCAT cmin=33'],
                                             u'T'],
                                      u'3': [[u'NUM'], [u'NI'], u'T'],
                                      u'4': [[u'3'], [u'DIFF ivr=0.01'], u'T'],
                                      u'5': [[u'3'],
                                             [u'RATIO dist=2;ivr=0.01'],
                                             u'T'],
                                      u'6': [[u'3', u'4', u'5'], [u'BIND'],
                                             u'T'],
                                      u'7': [[u'1', u'2', u'6'],
                                             [u'RFC e=0;mf=0.3'],
                                             u'T t=0.001'],
                                      u'8': [[u'7'],
                                             [
                                                 u'RFC e=1;t_a=2;t_n=1;t_f=0.15;ls=[5, 10, 20];mf=[0.2, 0.3, 0.4, 0.5];t_m=LogLoss'],
                                             u'S']},
                                  u'dataset_id': u'538cf2268bd88f21f07b6517',
                                  u'samplepct': 64},
                                 {u'blender': {},
                                  u'blueprint': {
                                      u'1': [[u'CAT'], [u'CRED1b1 cmin=1'],
                                             u'TS'],
                                      u'2': [[u'CAT'], [u'CCAT cmin=33'],
                                             u'T'],
                                      u'3': [[u'NUM'], [u'NI'], u'T'],
                                      u'4': [[u'3'], [u'DIFF ivr=0.01'], u'T'],
                                      u'5': [[u'3'],
                                             [u'RATIO dist=2;ivr=0.01'],
                                             u'T'],
                                      u'6': [[u'3', u'4', u'5'], [u'BIND'],
                                             u'T'],
                                      u'7': [[u'1', u'2', u'6'],
                                             [u'RFC e=0;mf=0.3'],
                                             u'T t=0.001'],
                                      u'8': [[u'7'],
                                             [
                                                 u'GBC lr=0.05;n=1000;mf=None;'
                                                 u'md=[1, 3, 5];t_m=LogLoss'],
                                             u'S']},
                                  u'dataset_id': u'538cf2268bd88f21f07b6517',
                                  u'samplepct': 64}]},
        u'blueprint': {u'1': [[u'a898c9aa53025b319892c1526fac0aed'],
                              [u'GLMB logitx'],
                              u'P']},
        u'blueprint_id': u'982cd2d53cae75c90506f321fa060d0e',
        u'bp': u'36+39+42',
        u'dataset_id': u'538cf2268bd88f21f07b6517',
        u'dataset_name': u'Informative Features',
        u'features': [],
        u'holdout': {u'AUC': 0.69231,
                     u'Gini': 0.12474,
                     u'Gini Norm': 0.38462,
                     u'Ians Metric': 0.1754,
                     u'LogLoss': 0.60863,
                     u'RMSE': 0.45259,
                     u'Rate@Top10%': 0.5,
                     u'Rate@Top5%': 1.0,
                     u'metrics': [u'LogLoss',
                                  u'AUC',
                                  u'Ians Metric',
                                  u'Gini',
                                  u'Gini Norm',
                                  u'Rate@Top10%',
                                  u'Rate@Top5%',
                                  u'RMSE']},
        u'holdout_size': 37,
        u'icons': [0],
        u'lid': u'538d00018bd88f02c9209b6c',
        u'max_folds': 0,
        u'max_reps': 1,
        u'metablueprint': [u'Metablueprint', u'8.6.7b'],
        u'partition_stats': {u'(0, -1)': {u'test_size': 35,
                                          u'time_real': u'0.0084',
                                          u'train_size': 123},
                             u'(1, -1)': {u'test_size': 33,
                                          u'time_real': u'0.00582',
                                          u'train_size': 125},
                             u'(2, -1)': {u'test_size': 30,
                                          u'time_real': u'0.00586',
                                          u'train_size': 128},
                             u'(3, -1)': {u'test_size': 30,
                                          u'time_real': u'0.00633',
                                          u'train_size': 128},
                             u'(4, -1)': {u'test_size': 30,
                                          u'time_real': u'0.00572',
                                          u'train_size': 128}},
        u'parts': [[u'1', 3], [u'3', 3], [u'2', 3], [u'5', 3], [u'4', 3]],
        u'parts_label': [u'partition', u'NonZeroCoefficients'],
        u'pid': ObjectId('538cf2248bd88f5010189057'),
        u'qid': 57,
        u'reference_model': False,
        u'samplepct': 64,
        u'task_cnt': 1,
        u'task_parameters': u'{"1": null, "GLMB logitx": null}',
        u'test': {u'AUC': [0.784, 0.8125],
                  u'Gini': [0.20286, 0.21756],
                  u'Gini Norm': [0.568, 0.625],
                  u'Ians Metric': [0.31036, 0.3035],
                  u'LogLoss': [0.42677, 0.45681],
                  u'RMSE': [0.36659, 0.38314],
                  u'Rate@Top10%': [1.0, 0.875],
                  u'Rate@Top5%': [1.0, 1.0],
                  u'labels': [u'(0,-1)', u'(.,-1)'],
                  u'metrics': [u'LogLoss',
                               u'AUC',
                               u'Ians Metric',
                               u'Gini',
                               u'Gini Norm',
                               u'Rate@Top10%',
                               u'Rate@Top5%',
                               u'RMSE']},
        u'total_size': 158,
        u'training_dataset_id': u'538cf2268bd88f21f07b6517',
        u'training_dataset_name': u'default',
        u'uid': ObjectId('5359d6cb8bd88f5cddefd3a8')}

    def test_without_assume_stored_submissions_thinks_ldb_nonempty(self):
        dataset_id = '52b6121e076329407cb2c88a'
        pid = self.make_fake_project(dataset_id=dataset_id)
        ldb_item = copy.deepcopy(self.BLENDER_ITEM_FIXTURE)
        ldb_item['dataset_id'] = dataset_id
        ldb_item['pid'] = pid
        self.add_data(pid, [ldb_item], remember_signatures=False)

        mb = MBAssumesNothing(pid, None)
        self.assertEqual(1, len(mb.leaderboard))

    def test_with_assume_stored_submissions_thinks_ldb_empty(self):
        dataset_id = '52b6121e076329407cb2c88a'
        pid = self.make_fake_project(dataset_id=dataset_id)
        ldb_item = copy.deepcopy(self.BLENDER_ITEM_FIXTURE)
        ldb_item['dataset_id'] = dataset_id
        ldb_item['pid'] = pid
        self.add_data(pid, [ldb_item], remember_signatures=False)

        mb = MBAssumesStorage(pid, None)
        self.assertEqual(0, len(mb.leaderboard))


class TestMBMetricValidation(BaseTestMB):
    class ConstructableBaseMB(BaseMetablueprint):
        def next_steps(self, metric):
            self.called_with = metric

            plan = {}
            plan['1'] = [['NUM'], ['NI'], 'T']
            plan['2'] = [['1'], ['RFR'], 'P']
            return [{'blueprint': plan,
                     'samplepct': 32,
                     'max_reps': 1,
                     'max_folds': 0,
                     'bp': 1}]

    def test_no_panic_mb_passed_a_nonexistent_metric_and_metric_stored(self):
        pid = self.make_fake_project(metric='LogLoss')
        mb = self.ConstructableBaseMB(pid, None)
        mb()
        self.assertEqual(mb.called_with, 'LogLoss')

    def test_no_panic_mb_passed_a_nonexistent_metric_and_metric_not_stored(
            self):
        """We have been storing the metric for a really long time now,
        but just in case, let's cover this case
        """
        pid = self.make_fake_project(metric=None)
        mb = self.ConstructableBaseMB(pid, None)
        mb()
        self.assertEqual(mb.called_with, EngConfig['DEFAULT_METRIC'])


class TestMBIfOnlyTextPredictors(BaseTestMB):
    def setUp(self):
        self.pid = self.make_fake_project(varTypeString='NT')

    def test_only_text_mining_insights_produced(self):
        mb = MBAssumesStorage(self.pid, None)
        insights_models = mb.add_insight_models({'max_reps': 1,
                                                 'samplepct': 16,
                                                 'max_folds': 0})

        # Two models run (word ngram for each var, ridge on all)
        self.assertEqual(len(insights_models), 1)
        self.assertIn('insights', insights_models[0])
        self.assertEqual(insights_models[0]['insights'], 'text')

    def test_no_reference_models_attempted(self):
        mb = MBAssumesStorage(self.pid, None, reference_models=True)
        reference_models = mb.add_reference_models(
            {'max_reps': 1, 'samplepct': 16, 'max_folds': 0})

        self.assertEqual(len(reference_models), 0)


if __name__ == '__main__':
    unittest.main()
