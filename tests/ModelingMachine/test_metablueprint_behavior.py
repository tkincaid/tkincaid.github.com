'''A test of the default metablueprint's logic

Basically we need to ensure that it

    1. Doesn't submit the same jobs twice
    2. Eventually submits a job with 5CV runs
    3. Eventually terminates

We bypass all of the actual machine learning tasks - those need to be
tested elsewhere.  This is just to ensure the above three happen.

We could also consider writing one of these for every metablueprint, and
keeping it inside the test file for that MB, but in a separate testcase

'''

import unittest
from mock import patch, Mock
import pytest

import bson

import random

from common.wrappers import database
from common import load_class
from config.test_config import db_config
from config.engine import EngConfig
from common.entities.blueprint import blueprint_id


class TestMetablueprintBehavior(unittest.TestCase):
    '''Duplicated jobs, never running 5CV jobs, and looping infinitely
    are hands down the worst sins that a metablueprint can commit

    We may want to test for other behaviors, but this basically comes up
    with some random results for every single job that is requested.
    Perhaps at some point we can substitute random results for real ones,
    I'm not sure that it matters.

    As metablueprints become more complicated, faking the data that the MBP
    expects to be in the database may become more complicated and this
    approach may not be feasible anymore.

    One glaringly false assumption this makes is that the samplesize that goes
    into the queue is the same as what goes into the leaderboard.  In reality
    there is logic that caps that samplesize to be the the full size of the
    non-holdout set.
    '''

    @classmethod
    def setUpClass(self):
        random.seed(0)

        self.tempstore = database.new('tempstore', db_config=db_config)
        self.persistent = database.new('persistent', db_config=db_config)
        self.pid = self.make_fake_project()
        self.make_fake_metadata(self.pid)

        Metablueprint = load_class(EngConfig['metablueprint_classname'])
        self.mb = Metablueprint(self.pid, None, progressInst=Mock(), reference_models=True,
                    tempstore=self.tempstore, persistent=self.persistent)
        self.mb._queue_settings = {'mode': '0'}

    @classmethod
    def tearDownClass(self):
        pass

    @classmethod
    def setUp(self):
        pass

    @classmethod
    def tearDown(self):
        pass

    def run_metablueprint(self, max_rounds=100):
        n_rounds_run = 0

        while self.fake_one_round():
            n_rounds_run += 1
            if n_rounds_run > max_rounds:
                raise RuntimeError('Looks like we got a runaway')
        return n_rounds_run

    @classmethod
    def make_fake_project(self):
        pid = self.persistent.create(table='project',
                values={'active':1,'default_dataset_id':'52b6121e076329407cb2c88b','holdout_pct':20,
                    'holdout_unlocked':False, 'partition':{'folds':5, 'reps':5},
                    'stage':'modeling', 'target':{'name':'1','size':800,'type':'Binary'},
                    'target_options':{'missing_maps_to':None,'name':'1',
                                      'positive_class':None},
                    'uid':'a_user_id'})
        return pid

    @classmethod
    def make_fake_metadata(self, pid):
        self.persistent.update(table='metadata',
            values={'columns':[[i, str(i), 0] for i in xrange(25)],
                    'created':'not-important',
                    'dataset_id':'52b6121e076329407cb2c88b',
                    'originalName':'an-original-name',
                    'pid':pid,
                    'shape':[1000,25],
                    'varTypeString':''.join(['N']*25),
                    'xavTypeString':''.join(['N']*25)},
            condition={"_id": bson.ObjectId("52b6121e076329407cb2c88b")})

    def make_fake_eda(self):
        pass

    @patch('common.services.queue_service_base.ProjectService.assert_has_permission')
    def fake_one_round(self,mock_object):
        '''Run next_steps, see if any jobs ended up in the queue,
        give fake results for all of them.  If no jobs ended up in
        the queue, return false, else return true

        '''
        self.mb._leaderboard = None
        with patch.object(self.mb, 'get_metric_for_models') as fake_metric_rec:
            fake_metric_rec.return_value = 'RMSE'
            jobs = self.mb.next_steps(metric='Gini')

            #TODO - can you eliminate redis from this test?
            self.mb.add_to_queue(jobs)

        new_jobs = self.tempstore.read(keyname='queue',
                                       index=str(self.pid),
                                       limit=(0,-1),
                                       result=[])

        ldb = self.mb.leaderboard
        print 'Leaderboard has {} items'.format(len(ldb))

        if len(new_jobs) > 0:
            print 'Generated {} jobs'.format(len(new_jobs))
            for job in new_jobs:
                if job['samplesize'] > 800:
                    raise RuntimeError('Asked for illegal samplesize')

                fake_result = self.fake_fit_results(job)
                if job.get('lid')=='new':
                    self.persistent.create(table='leaderboard', values=fake_result)
                else:
                    lid = database.ObjectId(job.pop('lid'))
                    self.persistent.update(condition={'_id':lid},table='leaderboard', values=fake_result)

            # Normally we pop the jobs one at a time so we don't have to delete
            # the key.  This is not the case here, so we'll just delete the queue
            self.tempstore.destroy(keyname='queue',index=str(self.pid))
            return True
            #TODO -make fake results for all of them and put them in the DB
        else:
            return False

    def check_for_duplicates(self, ldb):
        entries = [(l['blueprint_id'], l['samplesize'], l['max_reps']) for l in ldb]
        for i, entry in enumerate(entries):
            for j, entry2 in enumerate(entries[i+1:]):
                self.assertNotEqual(entry, entry2, 'Leaderboard items {} and {} are identical'.format(i,j))

    @pytest.mark.skip('This test is deprecated - unit testing has improved')
    def test_test(self):
        rounds_until_finish  = self.run_metablueprint(max_rounds=20)
        print 'We went {} rounds'.format(rounds_until_finish)
        for i,l in enumerate(self.mb.leaderboard):
            print i, ':', l['samplesize'], l['max_reps'], l['blueprint_id']


        self.mb._leaderboard = None # Clear the cache
        ldb = self.mb.leaderboard
        self.assertGreater(rounds_until_finish, 0, 'No rounds of metablueprint took place')
        self.assertGreater(len(ldb), 0, 'No blueprints were fit')
        self.assertEqual(max([len(l.get('partition_stats',[])) for l in ldb]), 5)
        self.check_for_duplicates(ldb)

    def fake_fit_results(self, job_item):
        ldb = {}
        for key in job_item:
            ldb[key] = job_item[key]

        print job_item
        ldb['pid'] = bson.ObjectId(job_item['pid'])
        ldb['total_size'] = 800

        #Fake the performance
        metrics = ['AUC', 'Gini', 'Gini Norm', 'Ians Metric', 'LogLoss',
                    'Rate@Top10%', 'Rate@Top5%']
        test = {'metrics':metrics}
        for metric in metrics:
            if job_item.get('max_reps')==1:
                test[metric] = [random.random()]
            else:
                test[metric] = [random.random(),random.random()]
        ldb['test'] = test
        ldb['blueprint_id'] = blueprint_id(ldb['blueprint'])
        if job_item.get('max_reps')==1:
            ldb['partition_stats'] = {str((0,-1)):'test'}
        else:
            ldb['partition_stats'] = {
                str((0,-1)):'test',
                str((1,-1)):'test',
                str((2,-1)):'test',
                str((3,-1)):'test',
                str((4,-1)):'test',
            }
        if job_item.get('partitions') and 'partition_stats' not in ldb:
            for p in job_item.get('partitions'):
                ldb['partition_stats.'+str((p[0],p[1]))] = 'test'

        return ldb



