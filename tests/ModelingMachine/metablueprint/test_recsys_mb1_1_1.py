import unittest
import pytest
from mock import patch

import common.entities.blueprint as bpm

from tests.ModelingMachine.test_base_mb import BaseTestMB
from ModelingMachine.metablueprint.recsys_mb1_1_2 import RecommenderMetaBlueprint


class TestRecsysMB111(BaseTestMB):

    def test_reference_models(self):
        pid = self.make_fake_project()
        mb = RecommenderMetaBlueprint(pid, None)
        args = {'max_folds': 0, 'max_reps': 5, 'samplepct': 16}
        ref_models = mb.add_reference_models(args)
        self.assertGreater(len(ref_models), 0)
        for model in ref_models:
            self.assertTrue(model['reference_model'])

    def test_initial_jobs_all_at_max_percent(self):
        pid = self.make_fake_project()
        mb = RecommenderMetaBlueprint(pid, None)
        with patch.object(mb, 'get_metric_for_models') as fake_m:
            fake_m.return_value = 'RMSE'
            jobs = mb.initial_joblist()
            self.assertGreater(len(jobs), 0)
            for job in jobs:
                self.assertEqual(job['samplepct'], mb.max_sample)

    def test_reference_models_all_num_output_regression(self):
        pid = self.make_fake_project(varTypeString='NNN')
        mb = RecommenderMetaBlueprint(pid, None)
        args = {'max_folds': 0, 'max_reps': 5, 'samplepct': 16}
        ref_models = mb.add_reference_models(args)
        self.assertEqual(len(ref_models), 5)
        bp_ids = [bpm.blueprint_id(model['blueprint'])
                  for model in ref_models]
        self.assertEqual(bp_ids, ['4bb8d053ca8202c50030e4f90c11cb19',
                                  '137391f842d1dc7b48e8476b5c855141',
                                  '5a9bbdca80699b6cec974135a207690b',
                                  '7db1b4847802bec666327c85292807a6',
                                  'ff7ad2cb05b1dcb0e75e6e1ffea1ef85',])

    def test_reference_models_all_cat_output_regression(self):
        pid = self.make_fake_project(varTypeString='CCC')
        mb = RecommenderMetaBlueprint(pid, None)
        args = {'max_folds': 0, 'max_reps': 5, 'samplepct': 16}
        ref_models = mb.add_reference_models(args)
        self.assertEqual(len(ref_models), 5)
        bp_ids = [bpm.blueprint_id(model['blueprint'])
                  for model in ref_models]
        self.assertEqual(bp_ids, ['4bb8d053ca8202c50030e4f90c11cb19',
                                  '137391f842d1dc7b48e8476b5c855141',
                                  '5a9bbdca80699b6cec974135a207690b',
                                  '7db1b4847802bec666327c85292807a6',
                                  'ff7ad2cb05b1dcb0e75e6e1ffea1ef85',])

    def test_insight_models(self):
        pid = self.make_fake_project()
        mb = RecommenderMetaBlueprint(pid, None)
        args = {'max_folds': 0, 'max_reps': 5, 'samplepct': 16}
        with patch.object(mb, 'get_metric_for_models') as fake_m:
            fake_m.return_value = 'RMSE'
            models = mb.add_insight_models(args)

    def test_just_default_insight_models_when_only_text_available(self):
        pid = self.make_fake_project(varTypeString='TTT',
                                     task_type='Regression')
        mb = RecommenderMetaBlueprint(pid, None)
        args = {'max_folds': 0, 'max_reps': 5, 'samplepct': 16}
        with patch.object(mb, 'get_metric_for_models') as fake_m:
            fake_m.return_value = 'RMSE'
            models = mb.add_insight_models(args)
            self.assertEqual(len(models), 0)
