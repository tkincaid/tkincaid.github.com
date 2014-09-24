import unittest
import pytest
import copy

from mock import patch, DEFAULT
from bson import ObjectId

import ModelingMachine.metablueprint.ref_model_mixins as rmm
from ModelingMachine.metablueprint.base_autopilot_mb import BaseAutopilotMB, _has_blend
import ModelingMachine.metablueprint.base_autopilot_mb as autopilot
from tests.ModelingMachine.test_base_mb import BaseTestMB

from common.services.flippers import FLIPPERS
from common.services.metablueprint import MBFlags

import ModelingMachine

@pytest.mark.unit
class TestBaseAutopilotMBsamplepcts(unittest.TestCase):

    def setUp(self):
        # This would be the result of a 20 pct holdout, and 5 partition reps
        self.sample_steps = [16, 32, 48, 64]

    def test_initial_samplepcts_160(self):
        calculated_pct = autopilot.initial_samplepct(160, self.sample_steps)
        self.assertEqual(64, calculated_pct)

    def test_initial_samplepcts_1600(self):
        calculated_pct = autopilot.initial_samplepct(1600, self.sample_steps)
        self.assertEqual(48, calculated_pct)

    def test_initial_samplepcts_100k(self):
        calculated_pct = autopilot.initial_samplepct(100000, self.sample_steps)
        self.assertEqual(10, calculated_pct)

    def test_initial_samplepcts_1072(self):
        '''The AutoBI case'''
        calculated_pct = autopilot.initial_samplepct(1072, self.sample_steps)
        self.assertEqual(48, calculated_pct)

    def test_next_samplepct_8k_after_25_pct(self):
        targetsize = 8000
        samplepcts = [16]*8
        ns = autopilot.get_next_samplepct_median(samplepcts, self.sample_steps)
        self.assertEqual(ns, 32)

    def test_next_samplepct_8k_after_50_pct(self):
        targetsize = 8000
        samplepcts = [32]*8
        ns = autopilot.get_next_samplepct_median(samplepcts, self.sample_steps)
        self.assertEqual(ns, 48)

    def test_next_samplepct_8k_after_75_pct(self):
        targetsize = 8000
        samplepcts = [48]*8
        ns = autopilot.get_next_samplepct_median(samplepcts, self.sample_steps)
        self.assertEqual(ns, 64)


# Doesn't actually use the db, just relies on BaseTestMB, for object
# generation, and that has some DB connections.
@pytest.mark.db
class TestAutopilotStages(BaseTestMB):

    def setUp(self):
        self.sample_steps = [16, 32, 48, 64]  # Assumes 20% holdout, 5 folds

    def test_basic_case_start_next_samplepct(self):
        target_size = 4000
        max_models = 8
        open_positions = 6  # i.e. 8 models continue, but 2 are in progress
        leaderboard = [self.one_fake_ldb_item('apid', 16, target_size, bp=i)
                       for i in range(32)]
        submitted_jobs = {'16': [autopilot.job_signature(l)
                                 for l in leaderboard]}
        for l in leaderboard[-2:]:
            l['test']['Gini'] = [None]

        jobs = autopilot.autopiloto(
            'Gini', submitted_jobs, leaderboard, target_size, open_positions,
            max_models, self.sample_steps)

        self.assertGreater(len(jobs), 0)
        self.assertLessEqual(len(jobs), open_positions)
        for job in jobs:
            self.assertEqual(job['samplepct'], 32)

    def test_end_of_round_2_or_higher_when_round_incomplete(self):
        '''N = MAX_MODELS, K = enqueued_models.  At end of round 1, only N-K
        jobs were submitted.  So when we come to the end of round 2, we need
        to check that the best N models from round 1 were submitted at
        samplepct 2
        '''
        target_size = 4000
        max_models = 8
        open_positions = 6
        leaderboard_round1 = [
            self.one_fake_ldb_item('apid', 25, target_size, bp=i)
            for i in range(32)]
        leaderboard_round2 = [
            self.one_fake_ldb_item('apid', 50, target_size, bp=i,
                                   bpoffset=32) for i in range(6)]
        for l in leaderboard_round2[-2:]:
            l['test']['Gini'] = [None]

        for i in range(6):
            leaderboard_round2[i]['blueprint_id'] = \
                leaderboard_round1[i]['blueprint_id']

        submitted_jobs = {'25': [autopilot.job_signature(l)
                                 for l in leaderboard_round1],
                          '50': [autopilot.job_signature(l)
                                 for l in leaderboard_round2]}

        leaderboard = leaderboard_round1 + leaderboard_round2[:-2]

        for i, l in enumerate(leaderboard):
            l['test']['Gini'] = [i*0.01]

        jobs = autopilot.autopiloto(
            'Gini', submitted_jobs, leaderboard, target_size, open_positions,
            max_models, self.sample_steps)

        self.assertGreater(len(jobs), 0)
        self.assertLessEqual(len(jobs), open_positions)
        for job in jobs:
            self.assertEqual(job['samplepct'], 50)
            sig = autopilot.job_signature(job)
            self.assertNotIn(sig, submitted_jobs['50'])

    def test_end_of_round_2_or_higher_when_round_complete(self):
        '''When we find that all of the jobs for a round have been
        submitted, it is safe to move on to the next samplepct
        '''
        target_size = 4000
        max_models = 8
        open_positions = 6
        leaderboard_round1 = [
            self.one_fake_ldb_item('apid', 16, target_size, bp=i)
            for i in range(32)]
        leaderboard_round2 = [
            self.one_fake_ldb_item('apid', 32, target_size, bp=i,
                                   bpoffset=32)
            for i in range(8)]

        for i in range(8):
            leaderboard_round2[i]['blueprint_id'] = \
                leaderboard_round1[i]['blueprint_id']

        submitted_jobs = {'16': [autopilot.job_signature(l)
                                 for l in leaderboard_round1],
                          '32': [autopilot.job_signature(l)
                                 for l in leaderboard_round2]}

        # Simulate condition when two jobs from current samplepct
        # have not yet finished - they will have been filtered out
        # by the caller of "autopilot_cruising"
        leaderboard = leaderboard_round1 + leaderboard_round2
        for i, l in enumerate(leaderboard):
            l['test']['Gini'] = [i*0.01]
        for l in leaderboard[-2:]:
            l['test']['Gini'] = [None]

        jobs = autopilot.autopiloto(
            'Gini', submitted_jobs, leaderboard, target_size, open_positions,
            max_models, self.sample_steps)

        self.assertGreater(len(jobs), 0)
        self.assertLessEqual(len(jobs), open_positions)
        for job in jobs:
            self.assertEqual(job['samplepct'], 48)

    def test_when_fullsize_examples_exist_but_not_all_done(self):
        target_size = 4000
        max_models = 8
        open_positions = 6
        leaderboard_round1 = [
            self.one_fake_ldb_item('apid', 25, target_size, bp=i)
            for i in range(32)]
        leaderboard_round2 = [
            self.one_fake_ldb_item('apid', 50, target_size, bp=i,
                                   bpoffset=32)
            for i in range(8)]
        leaderboard_round3 = [
            self.one_fake_ldb_item('apid', 75, target_size, bp=i,
                                   bpoffset=40)
            for i in range(8)]
        leaderboard_round4 = [
            self.one_fake_ldb_item('apid', 100, target_size, bp=i,
                                   bpoffset=48)
            for i in range(8)]

        for i in range(8):
            blueprint_id = leaderboard_round1[i]['blueprint_id']
            leaderboard_round2[i]['blueprint_id'] = blueprint_id
            leaderboard_round3[i]['blueprint_id'] = blueprint_id
            leaderboard_round4[i]['blueprint_id'] = blueprint_id

        submitted_jobs = {'25': [autopilot.job_signature(l)
                                 for l in leaderboard_round1],
                          '50': [autopilot.job_signature(l)
                                 for l in leaderboard_round2],
                          '75': [autopilot.job_signature(l)
                                 for l in leaderboard_round3],
                          '100': [autopilot.job_signature(l)
                              for l in leaderboard_round4[:-2]]}

        # Simulate condition when two jobs from current samplepct
        # have not yet finished - they will have been filtered out
        # by the caller of "autopilot_cruising"
        leaderboard = leaderboard_round1 + leaderboard_round2 \
            + leaderboard_round3 + leaderboard_round4[:-2]

        for i, l in enumerate(leaderboard):
            l['test']['Gini'] = [i*0.01]
        for l in leaderboard[-2:]:
            l['test']['Gini'] = [None]

        jobs = autopilot.autopiloto(
            'Gini', submitted_jobs, leaderboard, target_size, open_positions,
            max_models, self.sample_steps)
        for j in jobs:
            print 'Job: ', autopilot.job_signature(j)

        self.assertGreater(len(jobs), 0)
        self.assertLessEqual(len(jobs), open_positions)
        old_bp_ids = [i['blueprint_id'] for i in leaderboard_round3]
        for job in jobs:
            self.assertEqual(job['samplepct'], 100)
            js = autopilot.job_signature(job)
            self.assertNotIn(js, submitted_jobs['100'])
            self.assertIn(job['blueprint_id'], old_bp_ids)

    def test_when_all_fullsize_examples_have_been_submitted(self):
        target_size = 4000
        max_models = 8
        open_positions = 6
        leaderboard_round1 = [
            self.one_fake_ldb_item('apid', 1000, target_size, bp=i)
            for i in range(32)]
        leaderboard_round2 = [
            self.one_fake_ldb_item('apid', 2000, target_size, bp=i,
                                   bpoffset=32)
            for i in range(8)]
        leaderboard_round3 = [
            self.one_fake_ldb_item('apid', 3000, target_size, bp=i,
                                   bpoffset=40)
            for i in range(8)]
        leaderboard_round4 = [
            self.one_fake_ldb_item('apid', 4000, target_size, bp=i,
                                   bpoffset=48)
            for i in range(8)]

        for i in range(8):
            blueprint_id = leaderboard_round1[i]['blueprint_id']
            leaderboard_round2[i]['blueprint_id'] = blueprint_id
            leaderboard_round3[i]['blueprint_id'] = blueprint_id
            leaderboard_round4[i]['blueprint_id'] = blueprint_id

        submitted_jobs = {'1000': [autopilot.job_signature(l)
                                 for l in leaderboard_round1],
                          '2000': [autopilot.job_signature(l)
                                 for l in leaderboard_round2],
                          '3000': [autopilot.job_signature(l)
                                 for l in leaderboard_round3],
                          '4000': [autopilot.job_signature(l)
                              for l in leaderboard_round4]}

        # Simulate condition when two jobs from current samplepct
        # have not yet finished - they will have been filtered out
        # by the caller of "autopilot_cruising"
        leaderboard = leaderboard_round1 + leaderboard_round2 \
            + leaderboard_round3 + leaderboard_round4

        for i, l in enumerate(leaderboard):
            l['test']['RMSE'] = [i*0.01]
        for l in leaderboard[-2:]:
            l['test']['RMSE'] = [None]

        for l in leaderboard:
            print 'Leaderboard: ', l['blueprint_id'], l['samplepct']

        jobs = autopilot.autopiloto(
            'RMSE', submitted_jobs, leaderboard, target_size, open_positions,
            max_models, self.sample_steps)
        for j in jobs:
            print 'Job: ', autopilot.job_signature(j)

        self.assertGreater(len(jobs), 0)
        self.assertLessEqual(len(jobs), open_positions)
        old_bp_ids = [i['blueprint_id'] for i in leaderboard_round4]
        for job in jobs:
            self.assertEqual(job['samplepct'], 4000)
            self.assertEqual(job['max_reps'], 5)
            self.assertIn(job['blueprint_id'], old_bp_ids)


@pytest.mark.integration
class TestBaseAutopilotMBAutopilot(BaseTestMB):

    class UnnecessaryMB(BaseAutopilotMB, rmm.DefaultReferenceModelsMixin):
        def initial_blueprints(self):
            return []

        def get_recommended_metrics(self):
            return {'default': {'short_name': 'RMSE' },
                    'ranking': {'short_name': 'Gini' }}

        @property
        def flags(self):
            mbf = MBFlags()
            flags =  mbf.default_new_flags()
            return MBFlags.from_dict(flags)

    def test_initial_joblist_at_least_adds_reference_models(self):
        if FLIPPERS.use_new_autopilot:
            return  # This suite's setup was too brittle to change
        pid = self.make_fake_project(train_rows=200)
        mb = self.UnnecessaryMB(pid, ObjectId())

        with patch.object(mb, 'get_metric_for_models') as fake_metric:
            fake_metric.return_value = 'RMSE'
            newjobs = mb.next_steps('Gini')
            self.assertGreater(len(newjobs), 0)
    def test_advances_best_mbs_of_early_round_large(self):
        if FLIPPERS.use_new_autopilot:
            return  # This suite's setup was too brittle to change
        pid = self.make_fake_project(train_rows=20000)

        fake_ldb = [self.one_fake_ldb_item(pid, 16, 2000, bp=i)
                    for i in range(40)]
        self.add_data(pid, fake_ldb)
        self.preload_mb_state_data(pid,
            {'submitted_jobs':{'16':[autopilot.job_signature(l)
                                     for l in fake_ldb]}})

        mb = self.UnnecessaryMB(pid, ObjectId())
        newjobs = mb.next_steps('Gini')

        self.assertEqual(len(newjobs), 16)
        for job in newjobs:
            self.assertEqual(job['samplepct'], 32)
            self.assertEqual(job['lid'], 'new')
            self.assertEqual(job['max_reps'], 1)

    def test_advances_best_mbs_of_early_round_small(self):
        if FLIPPERS.use_new_autopilot:
            return  # This suite's setup was too brittle to change
        pid = self.make_fake_project(train_rows=2000)

        fake_ldb = [self.one_fake_ldb_item(pid, 32, 2000, bp=i)
                    for i in range(10)]
        self.add_data(pid, fake_ldb)
        self.preload_mb_state_data(pid,
            {'submitted_jobs':{'32':[autopilot.job_signature(l)
                                     for l in fake_ldb]}})

        mb = self.UnnecessaryMB(pid, ObjectId())
        newjobs = mb.next_steps('Gini')

        self.assertEqual(len(newjobs), 10)
        for job in newjobs:
            self.assertEqual(job['samplepct'], 48)
            self.assertEqual(job['lid'], 'new')
            self.assertEqual(job['max_reps'], 1)

    def test_advances_best_mbs_correct_samplepct_multiple_featurelists(self):
        """ Regression test to make sure autopilot works correctly if multiple
        feature lists are used.
        """
        if FLIPPERS.use_new_autopilot:
            return  # This suite's setup was too brittle to change
        pid = self.make_fake_project(train_rows=2000)
        # Create some jobs stored in the database with a different dataset_id
        fake_ldb = [self.one_fake_ldb_item(pid, 48, 2000, bp=i, dataset_id='52b6121e076329407cb2c88c')
                    for i in range(10)]
        self.add_data(pid, fake_ldb)
        self.preload_mb_state_data(pid,
            {'submitted_jobs': [autopilot.job_signature(l)
                                     for l in fake_ldb]})
        mb = self.UnnecessaryMB(pid, ObjectId(), dataset_id='52b6121e076329407cb2c88c')
        newjobs = mb()

        self.tempstore.conn.delete('queue:' + str(pid))
        mb1 = self.UnnecessaryMB(pid, ObjectId(), dataset_id='52b6121e076329407cb2c88d')
        mb1()
        enqueued = [job for job in mb1.q.get()
                    if job['status'] in ['queue', 'inprogress']]
        lb = []
        for item in enqueued:
            item['test'] = {'Gini': [0.1], 'metrics': ['Gini']}
            item['partition_stats'] = {str((0, -1)): 'test'}
            item['_id'] = 'fake-lid'

        #    self.add_data(pid, item, True)
            lb.append(item)
        mb1._leaderboard = lb
        self.tempstore.conn.delete('queue:' + str(pid))
        newjobs = mb1.next_steps('Gini')

        self.assertEqual(len(newjobs), 1)
        for job in newjobs:
            self.assertEqual(job['samplepct'], 48)
            self.assertEqual(job['lid'], 'new')
            self.assertEqual(job['max_reps'], 1)

    def test_advances_best_mbs_of_early_round_correct_samplepct_when_higher_is_worse(self):
        if FLIPPERS.use_new_autopilot:
            return  # This suite's setup was too brittle to change
        pid = self.make_fake_project(train_rows=2000)
        fake_ldb2 = [self.one_fake_ldb_item(pid, 16, 2000, bp=i, scoreoffset=10)
                    for i in range(15)]

        fake_ldb = [self.one_fake_ldb_item(pid, 32, 2000, bp=i, bpoffset=20)
                    for i in range(10)]
        ldb = copy.deepcopy(fake_ldb)
        ldb.extend(fake_ldb2)

        self.add_data(pid, ldb)

        mb = self.UnnecessaryMB(pid, ObjectId())
        newjobs = mb.next_steps('Gini')

        self.assertGreater(len(newjobs), 10)
        for job in newjobs:
            self.assertEqual(job['samplepct'], 48)
            self.assertEqual(job['lid'], 'new')
            self.assertEqual(job['max_reps'], 1)

    def test_autopilot_maintains_referenceness(self):
        ''' Regressioin test: see issue #3386
        '''
        if FLIPPERS.use_new_autopilot:
            return  # This suite's setup was too brittle to change
        pid = self.make_fake_project(train_rows=2000)
        proto_lb = self.fake_leaderboard_item1()
        ldb_list = self.produce_leaderboard_items(n=5, pid=pid, fixed_data={'reference_model': True, 'samplepct': 16})
        self.add_data(pid, ldb_list)
        self.preload_mb_state_data(pid,
                {'submitted_jobs': {'16': [autopilot.job_signature(l)
                                           for l in ldb_list]}})
        mb = self.UnnecessaryMB(pid, ObjectId())
        newjobs = mb.next_steps('Gini')
        print newjobs
        self.assertEqual(len(newjobs), 5)
        for job in newjobs:
            self.assertTrue(job['reference_model'])

    def test_autopilot_maintains_insightfulness(self):
        ''' Regression test: see issue #3386
        '''
        if FLIPPERS.use_new_autopilot:
            return  # This suite's setup was too brittle to change
        pid = self.make_fake_project(train_rows=2000)
        proto_lb = self.fake_leaderboard_item1()
        ldb_list = self.produce_leaderboard_items(n=5, pid=pid, fixed_data={'insights': 'text', 'samplepct': 16})
        self.add_data(pid, ldb_list)
        self.preload_mb_state_data(pid,
                {'submitted_jobs': {'16': [autopilot.job_signature(l)
                                           for l in ldb_list]}})
        mb = self.UnnecessaryMB(pid, ObjectId())
        newjobs = mb.next_steps('Gini')
        print newjobs
        self.assertEqual(len(newjobs), 5)
        for job in newjobs:
            self.assertEqual(job['insights'], 'text')

    def test_submit_8jobs_final_percentage(self):
        # For now we have a limit of 8 models for the final percentage,
        # This is because of the 5-cv runs for these models


        if FLIPPERS.use_new_autopilot:
            return  # This suite's setup was too brittle to change
        pid = self.make_fake_project(train_rows=2000)

        fake_ldb1 = [self.one_fake_ldb_item(pid, 16, 2000, bp=i)
                    for i in range(30)]
        self.add_data(pid, fake_ldb1)

        fake_ldb2 = [self.one_fake_ldb_item(pid, 32, 2000, bp=i, bpoffset=30, scoreoffset=0.3)
                    for i in range(15)]
        self.add_data(pid, fake_ldb2)

        fake_ldb3 = [self.one_fake_ldb_item(pid, 48, 2000, bp=i, bpoffset=45, scoreoffset=1)
                    for i in range(12)]

        self.add_data(pid, fake_ldb3)
        self.preload_mb_state_data(pid,
            {'submitted_jobs':{'16':[autopilot.job_signature(l)
                                     for l in fake_ldb1]},
            'submitted_jobs':{'32':[autopilot.job_signature(l)
                                     for l in fake_ldb2]},
            'submitted_jobs':{'48':[autopilot.job_signature(l)
                                     for l in fake_ldb3]},

                                     })

        mb = self.UnnecessaryMB(pid, ObjectId())
        newjobs = mb.next_steps('Gini')
        print newjobs
        self.assertEqual(len(newjobs), 8)
        for job in newjobs:
            self.assertEqual(job['samplepct'], 64)
            self.assertEqual(job['lid'], 'new')
            self.assertEqual(job['max_reps'], 1)

    def test_recognizes_to_run_5cv_jobs_float(self):
        """
        Also checks that it works if holdout is a float
        """
        if FLIPPERS.use_new_autopilot:
            return  # This suite's setup was too brittle to change
        pid = self.make_fake_project(train_rows=2000, holdout=19.5)
        fake_ldb = [self.one_fake_ldb_item(pid, 64, 2000, bp=i)
                    for i in range(10)]
        self.add_data(pid, fake_ldb)

        mb = self.UnnecessaryMB(pid, ObjectId())
        newjobs = mb.next_steps('Gini')
        self.assertGreater(len(newjobs), 0)
        for job in newjobs:
            print job
            self.assertEqual(job['samplepct'], 64)
            self.assertNotEqual(job['lid'], 'new')
            self.assertNotEqual(job['max_reps'], 1)

    def test_recognizes_to_run_5cv_jobs_int(self):
        """
        Checks that it works if holdout is a int
        """
        if FLIPPERS.use_new_autopilot:
            return  # This suite's setup was too brittle to change
        pid = self.make_fake_project(train_rows=2000, holdout=20)
        fake_ldb = [self.one_fake_ldb_item(pid, 64, 2000, bp=i)
                    for i in range(10)]
        self.add_data(pid, fake_ldb)

        mb = self.UnnecessaryMB(pid, ObjectId())
        newjobs = mb.next_steps('Gini')
        self.assertGreater(len(newjobs), 0)
        for job in newjobs:
            print job
            self.assertEqual(job['samplepct'], 64)
            self.assertNotEqual(job['lid'], 'new')
            self.assertNotEqual(job['max_reps'], 1)

    def test_recognizes_to_not_run_5cv_jobs(self):
        ''' For datasets >=50000 don't run 5 cv partitions
        '''
        if FLIPPERS.use_new_autopilot:
            return  # This suite's setup was too brittle to change
        pid = self.make_fake_project(train_rows=150000)
        fake_ldb = [self.one_fake_ldb_item(pid, 64, 150000, bp=i)
                    for i in range(10)]
        self.add_data(pid, fake_ldb)

        mb = self.UnnecessaryMB(pid, ObjectId())
        newjobs = mb.next_steps('Gini')
        self.assertGreater(len(newjobs), 0)
        for job in newjobs:
            self.assertTrue('blender_items' in job)
            self.assertGreater(len(job['blender_items']), 0)
            self.assertTrue('blender_method' in job)

    def test_no_infinite_spam_for_40k(self):
        '''Because of the weird way that the samplepct code was written,
        having 40k test rows caused a bad bag.  Once we refactor, this
        test can probably go away
        '''
        if FLIPPERS.use_new_autopilot:
            return  # This suite's setup was too brittle to change
        pid = self.make_fake_project(train_rows=40000)
        mb = self.UnnecessaryMB(pid, ObjectId())

        fake_ldb = [self.one_fake_ldb_item(pid, 25, 40000, bp=i)
                    for i in range(10)]
        self.add_data(pid, fake_ldb)

        newjobs = mb.next_steps('Gini')
        self.assertGreater(len(newjobs), 0)
        for job in newjobs:
            self.assertNotEqual(job['samplepct'], 25)

    def test_launches_blend_if_all_5_cvs_run(self):
        '''Auto blending shows off that feature
        '''
        if FLIPPERS.use_new_autopilot:
            return  # This suite's setup was too brittle to change
        pid = self.make_fake_project(train_rows=40000)

        fake_ldb = [self.one_fake_ldb_item(pid, 64, 40000, bp=i, max_reps=5)
                    for i in range(10)]
        self.add_data(pid, fake_ldb)

        mb = self.UnnecessaryMB(pid, ObjectId())
        with patch.object(mb, 'set_autopilot_done') as fake_done_signal:
            newjobs = mb.next_steps('Gini')
            self.assertGreater(len(newjobs), 0)
            for job in newjobs:
                self.assertIn('blender_method', job)
            self.assertTrue(fake_done_signal.called)

    def test_launches_blend_makes_has_blend_true(self):
        '''Test to make sure that has blends detects
        blends submitted
        '''
        if FLIPPERS.use_new_autopilot:
            return  # This suite's setup was too brittle to change
        pid = self.make_fake_project(train_rows=40000)

        fake_ldb = [self.one_fake_ldb_item(pid, 64, 40000, bp=i, max_reps=5)
                    for i in range(10)]
        self.add_data(pid, fake_ldb)

        mb = self.UnnecessaryMB(pid, ObjectId())
        # No blends should be on the leaderboard
        self.assertFalse(_has_blend(mb.leaderboard))

        with patch.object(mb, 'set_autopilot_done') as fake_done_signal:
            with patch.object(mb, 'q') as fake_queue:
                fake_queue.blend.return_value = {'blueprint_id': "7db1b4847802bec666327c85292807a6",
                                                 'dataset_id': "52b6121e076329407cb2c88d",
                                                 'samplepct': "64", 'max_reps': "5", 'max_folds': '0'}
                mb()
                # Need to add it to the leaderboard as well to simulate the model finishing
                ldb_item = [{'lid': 'fake_lid', 'pid': ObjectId(pid), 'blender': 'fake_blender',
                                    'blueprint_id': "7db1b4847802bec666327c85292807a6",
                                    'blueprint': "fake_bp",
                                    'dataset_id': "52b6121e076329407cb2c88d",
                                    'samplepct': "64", 'max_reps': "5", 'max_folds': '0'}]

                # Don't store it in automatically in submitted jobs to test that the mb
                # actually stores it
                self.add_data(pid, ldb_item, False)

                mb._leaderboard = None

        # Make sure that submitted_jobs_stored is True so filtering in effect
        self.assertTrue(mb.flags.submitted_jobs_stored)
        # Check that _has_blend now returns True
        self.assertTrue(_has_blend(mb.leaderboard))

    def test_launches_blend_even_if_one_5cv_missing(self):
        '''Auto blending shows off that feature
        '''
        if FLIPPERS.use_new_autopilot:
            return  # This suite's setup was too brittle to change
        pid = self.make_fake_project(train_rows=40000)

        fake_ldb = [self.one_fake_ldb_item(pid, 64, 40000, bp=i, max_reps=5)
                    for i in range(6)]
        broken_item = self.one_fake_ldb_item(pid, 64, 40000, bp=20, max_reps=5)
        for metric_name in set(broken_item['test']['metrics']):
            broken_item['test'][metric_name].pop(-1)
        broken_item['partition_stats'] = {
            str((i, -1)): 'test' for i in [0, 1, 3, 4]}
        fake_ldb.append(broken_item)
        self.add_data(pid, fake_ldb)

        mb = self.UnnecessaryMB(pid, ObjectId())
        with patch.object(mb, 'set_autopilot_done') as fake_done_signal:
            newjobs = mb.next_steps('Gini')
            self.assertGreater(len(newjobs), 0)
            for job in newjobs:
                self.assertIn('blender_method', job)
            self.assertTrue(fake_done_signal.called)

    def test_launches_blends_if_max_reps_is_unity_float(self):
        ''' We can have max_reps = 1, this might break the old logic

        This also checks that it then works even if holdout is a float
        '''
        if FLIPPERS.use_new_autopilot:
            return  # This suite's setup was too brittle to change
        pid = self.make_fake_project(train_rows=40000, project_reps=1, holdout=19.5)

        fake_ldb = [self.one_fake_ldb_item(pid, 40, 40000, bp=30+i, max_reps=1, scoreoffset=-1)
                    for i in range(6)]
        forty_lids = [i['_id'] for i in fake_ldb]
        fake_ldb.extend([self.one_fake_ldb_item(pid, 64, 40000, bp=i, max_reps=1)
                    for i in range(8)])
        good_models =  [self.one_fake_ldb_item(pid, 64, 40000, bp=10+i, max_reps=1, scoreoffset=1000)
                    for i in range(6)]
        more_bad_models = [self.one_fake_ldb_item(pid, 64, 40000, bp=20+i, max_reps=1)
                    for i in range(6)]

        good_models_lids = [i['_id'] for i in good_models]
        broken_item = self.one_fake_ldb_item(pid, 64, 40000, bp=20, max_reps=1)
        for metric_name in set(broken_item['test']['metrics']):
            broken_item['test'][metric_name].pop(-1)
        broken_item['partition_stats'] = {
            str((i, -1)): 'test' for i in [0]}
        fake_ldb.extend(good_models)
        fake_ldb.extend(more_bad_models)
        self.add_data(pid, fake_ldb)

        mb = self.UnnecessaryMB(pid, ObjectId())
        with patch.object(mb, 'set_autopilot_done') as fake_done_signal:
            newjobs = mb.next_steps('Gini')
            self.assertGreater(len(newjobs), 0)
            for job in newjobs:
                self.assertIn('blender_method', job)
                # Should not include any 40 pct models
                for item in job['blender_items']:
                    self.assertTrue(item not in fake_ldb)
                # If not backwards stepwise, should only blend good models
                if 'sw_b=2' not in job['blender_args']:
                    for item in job['blender_items']:
                        self.assertIn(item, good_models_lids)

            self.assertTrue(fake_done_signal.called)

    def test_launches_blends_if_max_reps_is_unity_int(self):
        ''' We can have max_reps = 1, this might break the old logic

        This also checks that it then works even if holdout is a int
        '''
        if FLIPPERS.use_new_autopilot:
            return  # This suite's setup was too brittle to change
        pid = self.make_fake_project(train_rows=40000, project_reps=1, holdout=20)

        fake_ldb = [self.one_fake_ldb_item(pid, 40, 40000, bp=30+i, max_reps=1, scoreoffset=-1)
                    for i in range(6)]
        forty_lids = [i['_id'] for i in fake_ldb]
        fake_ldb.extend([self.one_fake_ldb_item(pid, 64, 40000, bp=i, max_reps=1)
                    for i in range(8)])
        good_models =  [self.one_fake_ldb_item(pid, 64, 40000, bp=10+i, max_reps=1, scoreoffset=1000)
                    for i in range(6)]
        more_bad_models = [self.one_fake_ldb_item(pid, 64, 40000, bp=20+i, max_reps=1)
                    for i in range(6)]

        good_models_lids = [i['_id'] for i in good_models]
        broken_item = self.one_fake_ldb_item(pid, 64, 40000, bp=20, max_reps=1)
        for metric_name in set(broken_item['test']['metrics']):
            broken_item['test'][metric_name].pop(-1)
        broken_item['partition_stats'] = {
            str((i, -1)): 'test' for i in [0]}
        fake_ldb.extend(good_models)
        fake_ldb.extend(more_bad_models)
        self.add_data(pid, fake_ldb)

        mb = self.UnnecessaryMB(pid, ObjectId())
        with patch.object(mb, 'set_autopilot_done') as fake_done_signal:
            newjobs = mb.next_steps('Gini')
            self.assertGreater(len(newjobs), 0)
            for job in newjobs:
                self.assertIn('blender_method', job)
                # Should not include any 40 pct models
                for item in job['blender_items']:
                    self.assertTrue(item not in fake_ldb)
                # If not backwards stepwise, should only blend good models
                if 'sw_b=2' not in job['blender_args']:
                    for item in job['blender_items']:
                        self.assertIn(item, good_models_lids)

            self.assertTrue(fake_done_signal.called)
    def test_finishes_if_blend_present(self):
        '''Specifically, launches no new jobs and calls ``set_autopilot_done``

        '''
        if FLIPPERS.use_new_autopilot:
            return  # This suite's setup was too brittle to change
        pid = self.make_fake_project(train_rows=40000)
        fake_ldb = [self.one_fake_ldb_item(pid, 64, 40000, bp=i, max_reps=5,
                                           blend=1) for i in range(10)]
        self.preload_mb_state_data(pid,
                {'submitted_jobs':[
                    autopilot.job_signature(l) for l in fake_ldb]})
        self.add_data(pid, fake_ldb)
        mb = self.UnnecessaryMB(pid, ObjectId())

        with patch.object(mb, 'set_autopilot_done') as mock_finish:
            with patch('ModelingMachine.metablueprint.base_autopilot_mb._has_blend') as mock_has_blend:
                mock_has_blend.return_value=True
                newjobs = mb.next_steps('Gini')
                self.assertEqual(len(newjobs), 0)
                self.assertTrue(mock_finish.called)

@pytest.mark.integration
class TestAutopilotBlending(BaseTestMB):

    class UnnecessaryMB(BaseAutopilotMB):
        def initial_blueprints(self):
            return []

    def test_make_blends_creates_GLM(self):
        pid = self.make_fake_project(train_rows=10000)
        mb = self.UnnecessaryMB(pid, ObjectId())

        fake_ldb = [self.one_fake_ldb_item(pid, 10000, 10000, bp=i, max_reps=5)
                    for i in range(10)]

        # All of those had the same blueprint, we can make some up
        models = ['LR1', 'RFR', 'RFE', 'SGDC', 'SVMC', 'LSVC', 'KNNC', 'GBC',
                  'ESGBC', 'RGBC']
        for i, m in enumerate(models):
            bp = fake_ldb[i]['blueprint']
            fake_ldb[i]['blueprint'][str(len(bp))][1] = [m]

        blends = autopilot.make_blends(fake_ldb, 'Gini', 'Regression')
        methods = [i['blender_method'] for i in blends]
        self.assertIn('GLM', methods)
        for blend in blends:
            self.assertIn('blender_items', blend)

@pytest.mark.integration
class TestAutopilotCalledAnytime(BaseTestMB):
    """We want to be able to submit more jobs to the queue before a round ends,
    which means that the metablueprint needs to be able to be called at any
    time and still return a sensible set of jobs.  Might require adding some
    state to the metablueprint.
    """
    @classmethod
    def setUpClass(cls):
        super(TestAutopilotCalledAnytime, cls).setUpClass()
        cls._blueprint_ids = {}
        for i in range(100):
            cls._blueprint_ids[i] = cls.randomword(32)

    class UnnecessaryMB(BaseAutopilotMB):
        def initial_blueprints(self):
            return []

        def get_recommended_metrics(self):
            return {'default': {'short_name': 'RMSE'}}

    class OneModelMB(BaseAutopilotMB):
        def initial_blueprints(self):
            bp = {
                '1': [['NUM'], ['NI'], 'T'],
                '2': [['1'], ['RFR'], 'P']
            }
            return [bp]

        def get_recommended_metrics(self):
            return {'default': {'short_name': 'RMSE'}}

    def test_when_called_multiple_times_initially_doesnt_get_ahead(self):
        pid = self.make_fake_project()

        outcomes = []
        for i in range(4):
            mb = self.OneModelMB(pid, ObjectId())
            jobs_it_will_submit = mb.next_steps()
            outcomes.append(jobs_it_will_submit)
            mb()

        queue_status = self.tempstore.read(keyname=self.QueueKey.QUEUE,
                                           index=str(pid),
                                           result=[])
        self.assertEqual(len(queue_status), 2)
        self.assertEqual(len(outcomes[0]), 2)
        self.assertEqual(len(outcomes[1]), 0)
        self.assertEqual(len(outcomes[2]), 0)
        self.assertEqual(len(outcomes[3]), 0)

    def test_when_called_early_doesnt_get_ahead(self):
        """Inital jobs populated, and one has finished
        """
        pid = self.make_fake_project()

        fake_ldb = [self.one_fake_ldb_item(pid, 16, 10000, bp=0,
                                           max_reps=1)]
        fake_queue = [self.one_fake_queue_item(pid, 16, 10000, bp=i, rep=1)
                      for i in range(1, 20)]
        self.add_data(pid, fake_ldb)
        self.insert_in_queue(pid, fake_queue, key=self.QueueKey.QUEUE)

        mb = self.UnnecessaryMB(pid, ObjectId())
        jobs = mb.next_steps('Gini')
        self.assertEqual(len(jobs), 0)

    def test_when_called_prematurely_returns_no_jobs(self):
        """Even though it should only be called if there are enough jobs
        finished to know that at least one of them should advance, let's
        make it handle that case.  Simulate this by having only some jobs
        actually finished in the leaderboard, none in progress, and many
        in the queue.
        """
        pid = self.make_fake_project(train_rows=10000)
        fake_ldb = [self.one_fake_ldb_item(pid, 2500, 10000, bp=i, max_reps=1)
                    for i in range(10)]
        fake_queue = [self.one_fake_queue_item(pid, 2500, 10000, bp=i, rep=1)
                      for i in range(10, 20)]
        self.add_data(pid, fake_ldb, remember_signatures=True)
        self.insert_in_queue(pid, fake_queue, key='queue')

        mb = self.UnnecessaryMB(pid, ObjectId())
        jobs = mb.next_steps('Gini')
        self.assertEqual(len(jobs), 0)

    def test_errored_jobs_dont_stop_mb_from_creating(self):
        if FLIPPERS.use_new_autopilot:
            return  # Old autopilot test
        pid = self.make_fake_project(train_rows=10000)
        fake_ldb = [self.one_fake_ldb_item(pid, 25, 10000, bp=i, max_reps=1)
                    for i in range(10)]
        fake_errors = [self.one_fake_queue_item(pid, 25, 10000, bp=i, rep=1)
                       for i in range(10, 20)]
        self.add_data(pid, fake_ldb)
        self.insert_in_queue(pid, fake_errors, key='errors')
        self.preload_mb_state_data(pid, {'submitted_jobs':
            {'25': [autopilot.job_signature(j) for j in fake_ldb]}})

        mb = self.UnnecessaryMB(pid, ObjectId())

        jobs = mb.next_steps('Gini')
        self.assertGreater(len(jobs), 0)

    def test_when_called_doesnt_automatically_advance_unfinished_jobs(self):
        '''Some combos of metric direction and metric sorting can result in
        unfinished jobs being the top models.  That is no good
        '''
        if FLIPPERS.use_new_autopilot:
            return  # Old autopilot test
        pid = self.make_fake_project(train_rows=10000)

        # Create a leaderboard with RMSE as the metric under test
        fake_ldb = [self.one_fake_ldb_item(pid, 25, 10000, bp=i, max_reps=1)
                    for i in range(10)]
        for i, ldb in enumerate(fake_ldb):
            ldb['test']['RMSE'] = [i * 0.1]

        # Make some of the leaderboard items be in progress and not have "test"
        # in the leaderboard (I'm not sure, but I think that's the way you can
        # tell if it has finished. Wait, what about higher CV jobs?...
        # hmmm...)
        last_guy = fake_ldb[-1]
        fake_in_progress = [
            self.one_fake_queue_item(pid, 25, 10000, bp=last_guy['bp'],
                                     rep=1)]
        fake_in_progress[0]['blueprint_id'] = last_guy['blueprint_id']

        for metric in last_guy['test']['metrics']:
            last_guy['test'][metric] = [None]

        # If these are part of the new jobs, we are in trouble
        inprogress_bpids = [i['blueprint_id'] for i in fake_in_progress]

        self.add_data(pid, fake_ldb)
        self.insert_in_queue(pid, fake_in_progress, 'inprogress')

        mb = self.UnnecessaryMB(pid, ObjectId())
        jobs = mb.next_steps('RMSE')
        self.assertEqual(len(jobs), 0)


class TestAutopilotShyMode(BaseTestMB):
    '''We are going to assign names to the preprocessing steps.  The idea is
    to be able to switch between full blueprint descriptions and minimized
    blueprint descriptions ("shy mode") with just a flipper.  Since the
    methods in preprocessing_list are called inside of initial_blueprints
    of mb8.6 and mb8.7 (the current and next, at time of writing), we will
    need to allow those methods to add the descriptions
    '''

    class MetablueprintTest(BaseAutopilotMB, rmm.DefaultReferenceModelsMixin):
        def initial_blueprints(self):
            return [{'blueprint': {'1': [['NUM'], ['NI'], 'T'],
                                   '2': [['1'], ['RFR'], 'P']},
                     'features': ['TEST Missing Values Treated']
                     }]

        def get_recommended_metrics(self):
            return {'default': {'short_name': 'RMSE'}}

    def test_features_in_initial_blueprints_doesnt_break_it(self):
        pid = self.make_fake_project()
        mb = self.MetablueprintTest(pid, ObjectId())

        jobs = mb.next_steps()

        self.assertTrue(any('features' in job for job in jobs))
        self.assertTrue(any(job.get('features') == ['TEST Missing Values Treated']
                            for job in jobs))


    @pytest.mark.integration
    def test_features_in_initial_blueprints_dont_get_lost(self):
        pid = self.make_fake_project(metric='RMSE')
        mb = self.MetablueprintTest(pid, ObjectId())

        # Put the jobs into the database
        mb()

        the_queue = self.tempstore.read(keyname='queue',
                                        index=str(pid),
                                        result=[])

        for i, qitem in enumerate(the_queue):
            print 'Queue item {}'.format(i)
            for k, v in qitem.iteritems():
                print '\t{}={}'.format(k, v)

        self.assertTrue(any(job['features'] == ['TEST Missing Values Treated']
                            for job in the_queue))



