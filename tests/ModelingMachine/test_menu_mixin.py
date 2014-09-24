import unittest
from mock import patch, DEFAULT

import ModelingMachine.metablueprint.ref_model_mixins as rmm
from ModelingMachine.metablueprint.mb8_7 import MBSeriesEight
from tests.ModelingMachine.test_base_mb import BaseTestMB
import common.engine.metrics as metrics
import ModelingMachine.metablueprint.preprocessing_list_V4 as ppl

class TestMenuMixin(BaseTestMB):

    class TestMB(MBSeriesEight, rmm.DefaultReferenceModelsMixin):
        '''MBSeriesEight uses the SeriesEightMenuMixin, which is what
        we actually want to test
        '''

        def initial_blueprints(self):
            m_params = self.get_modeling_parameters()
            return [ppl.DR_NN_VANILLA_BLUEPRINT(m_params.rtype,
                                                m_params.Y_transf,
                                                m_params.available_types)]

    def test_mixin_generates_a_menu(self):
        dataset_id = '5223deadfeedbeefdeed1234'
        varTypeString = 'NNNNCCCC'
        pid = self.make_fake_project(dataset_id=dataset_id,
                                     varTypeString=varTypeString)
        uid = None
        mb = self.TestMB(pid, uid)

        with patch.multiple(mb, get_recommended_metrics=DEFAULT,
                            _metadata=DEFAULT,
                            get_available_types=DEFAULT) as fakes:
            fakes['get_recommended_metrics'].return_value = {
                  'default': {'short_name': metrics.LOGLOSS},
                  'recommender': {'short_name': metrics.AUC},
                  'weighted': {'short_name': metrics.LOGLOSS_W},
                  'weight+rec': {'short_name': metrics.AUC_W}
            }
            fakes['get_available_types'].return_value = set(
                    [i for i in varTypeString])
            fakes['_metadata'] = {dataset_id:
                {
                    'pct_min_y': 0.01,
                }
            }
            mb.generate_menu()

        data = self.persistent.read(table='metablueprint',
                                    condition={'pid':pid},
                                    result={})
        menu = data['menu']
        self.assertGreater(len(menu), 0)
        for menu_item in menu.values():
            self.assertIn('manual', menu_item)
            self.assertEqual(menu_item['manual'], 1)

    def test_autopilot_jobs_will_overwrite_menu_and_not_be_manual(self):
        dataset_id = '5223deadfeedbeefdeed1234'
        varTypeString = 'NNNNCCCC'
        pid = self.make_fake_project(dataset_id=dataset_id,
                                     varTypeString=varTypeString)
        uid = None
        mb = self.TestMB(pid, uid)

        with patch.multiple(mb, get_recommended_metrics=DEFAULT,
                            _metadata=DEFAULT,
                            get_available_types=DEFAULT) as fakes:
            fakes['get_recommended_metrics'].return_value = {
                  'default': {'short_name': metrics.LOGLOSS},
                  'recommender': {'short_name': metrics.AUC},
                  'weighted': {'short_name': metrics.LOGLOSS_W},
                  'weight+rec': {'short_name': metrics.AUC_W}
            }
            fakes['get_available_types'].return_value = set(
                    [i for i in varTypeString])
            fakes['_metadata'] = {dataset_id:
                {
                    'pct_min_y': 0.01,
                }
            }
            mb.generate_menu()
            mb()

        data = self.persistent.read(table='metablueprint',
                                    condition={'pid':pid},
                                    result={})
        menu = data['menu']
        self.assertGreater(len(menu), 0)
        model_types = [i['model_type'] for i in menu.values()]
        blueprint_ids = [i['blueprint_id'] for i in menu.values()]
        self.assertIn('Neural Net Classifier', model_types)
        nn_idx = model_types.index('Neural Net Classifier')
        self.assertNotIn('manual', menu[blueprint_ids[nn_idx]])



