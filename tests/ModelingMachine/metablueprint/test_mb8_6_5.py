import unittest
import pytest
from mock import patch

import common.entities.blueprint as bpm

from tests.ModelingMachine.test_base_mb import BaseTestMB
from ModelingMachine.metablueprint.mb8_6_5 import Metablueprint


class TestMB8_6_5(BaseTestMB):

    def test_reference_models(self):
        pid = self.make_fake_project()
        mb = Metablueprint(pid, None, reference_models=True)
        args = {'max_folds': 0, 'max_reps': 5, 'samplepct': 16}
        ref_models = mb.add_reference_models(args)
        self.assertGreater(len(ref_models), 0)
        for model in ref_models:
            self.assertTrue(model['reference_model'])

    def test_reference_models_all_num_output_regression(self):
        pid = self.make_fake_project(varTypeString='NNN')
        mb = Metablueprint(pid, None, reference_models=True)
        args = {'max_folds': 0, 'max_reps': 5, 'samplepct': 16}
        ref_models = mb.add_reference_models(args)
        self.assertEqual(len(ref_models), 11)
        bp_ids = [bpm.blueprint_id(model['blueprint'])
                  for model in ref_models]
        self.maxDiff = None
        expected_bp_ids = set(['d4c06a5c23cf1d917019720bceba32c8',
                               'bf31ed01c39a3686b5a75ff486097aea',
                               '70e6521b809c7c04d02f53e520815227',
                               '67c42e9227c2d54eb540ec4781c8925e',
                               '82a07cfe033f6d4496522dc01e316671',
                               '2bf563186fa8d83fabe71e1dd86f78a8',
                               '33121f57a51cc68e4ecfe74a43bad439',
                               '6ee139c9b97b2b397c472e2cd04b2211',
                               '38ea275c225db179833a98918c304e6d',
                               '89e08076a908e859c07af49bd4aa6a0f',
                               '304452be31ffad5ebfc9b716b28f9a23',
                               ])
        self.assertEqual(expected_bp_ids, set(bp_ids))

    def test_reference_models_all_cat_output_regression(self):
        pid = self.make_fake_project(varTypeString='CCC')
        mb = Metablueprint(pid, None, reference_models=True)
        args = {'max_folds': 0, 'max_reps': 5, 'samplepct': 16}
        ref_models = mb.add_reference_models(args)
        self.assertEqual(len(ref_models), 11)
        bp_ids = [bpm.blueprint_id(model['blueprint'])
                  for model in ref_models]
        expected_bp_ids = set(['3010cf2f2d8f7e64f2665c0223ae6bd7',
                               '84e873fd23202fd3979cf62b7ed2e1a7',
                               '1a88e3f969acf8f4b6c8304241974f9d',
                               'eb2faf22a6bcf8f37fa705ef321c1e39',
                               'de34b774738e6e0a1b17890fab40676c',
                               '15696229b1a73d530c87bd0f12b0db3f',
                               'd1d7c0b34e0b537bfc6ebf598949783d',
                               'fd712fa7fc44655df6c98e62495425ba',
                               '6911843177d937f7a30ae328f75d8c12',
                               '89e08076a908e859c07af49bd4aa6a0f',
                               'bc937d1de3412d78557c86bea3a81b23',
                               ])
        self.assertEqual(expected_bp_ids, set(bp_ids))

    def test_reference_models_num_and_cat_output_regression(self):
        pid = self.make_fake_project(varTypeString='CCCNNN')
        mb = Metablueprint(pid, None, reference_models=True)
        args = {'max_folds': 0, 'max_reps': 5, 'samplepct': 16}
        ref_models = mb.add_reference_models(args)
        self.assertEqual(len(ref_models), 11)
        bp_ids = [bpm.blueprint_id(model['blueprint'])
                  for model in ref_models]
        self.maxDiff = None
        expected_bp_ids = set(['08f15f00ed64dfd34fbae67b442228e5',
                               'ae3e2379a7f851d827ae3759234787b1',
                               'e839218c0eff338b14a708e81cdec46e',
                               'e27d0e503e6324730d773a2e2e4dded1',
                               '49e0d62fca6f42350fc65249b7c58e2e',
                               '2ca07ec2136f238383fdfbf1999bf2f6',
                               '1ac393161bed15a780903f4ac74b4d14',
                               '41b7fb41264bee8550dd042a464e67be',
                               '0aee8b32ddc14a2dd2c2e940097ffcae',
                               '89e08076a908e859c07af49bd4aa6a0f',
                               'c07a3342d5fab1d2e913a2d460a70ecf'])
        self.assertEqual(expected_bp_ids, set(bp_ids))

    def test_insight_models(self):
        pid = self.make_fake_project()
        mb = Metablueprint(pid, None, reference_models=True)
        args = {'max_folds': 0, 'max_reps': 5, 'samplepct': 16}
        with patch.object(mb, 'get_metric_for_models') as fake_m:
            fake_m.return_value = 'RMSE'
            models = mb.add_insight_models(args)

    def test_just_default_insight_models_when_only_text_available(self):
        pid = self.make_fake_project(varTypeString='TTT',
                                     task_type='Regression')
        mb = Metablueprint(pid, None)
        args = {'max_folds': 0, 'max_reps': 5, 'samplepct': 16}
        with patch.object(mb, 'get_metric_for_models') as fake_m:
            fake_m.return_value = 'RMSE'
            models = mb.add_insight_models(args)
            self.assertEqual(len(models), 2)
            model = models[0]
            self.assertEqual(bpm.blueprint_id(model['blueprint']),
                             '2c61f324beec2fd3cd752ae6d773432b')
            model1 = models[1]
            self.assertEqual(bpm.blueprint_id(model1['blueprint']),
                             'c2d6f258f309486db663d663b2c76ab4')
