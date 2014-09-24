import json
import unittest

from mock import patch
import pytest

import ModelingMachine.metablueprint.preprocessing_list_V2 as pre
from common.services.flippers import FLIPPERS
import common.entities.blueprint as bp_mod

class TestPreprocessingDecorator(unittest.TestCase):

    def test_without_flipper_just_returns_blueprint(self):
        # TODO: Make the return consistent, even if it is just
        # a dictionary with a single value
        if FLIPPERS.graybox_enabled:
            return True

        value = pre.DRT_NI_x_OE_COUNT_x_TF(
            model='RFC', available_types=set(['N']), count_args='cmin=33',
            fs_rf_args='e=1;nt=2000;mf=1')

        keys = value.keys()
        nkeys = len(keys)
        # Every blueprint key is a number
        for key in keys:
            nkey = int(key)
            self.assertGreater(nkey, 0)
            self.assertLessEqual(nkey, nkeys)

    def test_with_flipper_has_details(self):
        if not FLIPPERS.graybox_enabled:
            return True

        value = pre.DRT_NI_x_OE_COUNT_x_TF(
            model='RFC', available_types=set(['N']), count_args='cmin=33',
            fs_rf_args='e=1;nt=2000;mf=1')

        self.assertIn('blueprint', value.keys())
        self.assertIn('features', value.keys())
        self.assertIn('diagram', value.keys())

        dia = json.loads(value['diagram'])
        # The diagram is stored as json to circumvent any complaints
        # mongo may have about key names

    @pytest.mark.skip('We decided to go very blackboxy')
    def test_diagram_is_censored_but_not_blackbox(self):
        if not FLIPPERS.graybox_enabled:
            return True

        value = pre.DRT_NI_x_OE_COUNT_x_TF(
            model='RFC', available_types=set(['N', 'C']), count_args='cmin=33',
            fs_rf_args='e=1;nt=2000;mf=1', selection_type=None)

        self.assertIn('diagram', value.keys())
        dia = json.loads(value['diagram'])
        self.assertIn('NIA', dia['taskMap'])
        self.assertNotIn('CCAT', dia['taskMap'])

    @pytest.mark.skip('We decided to go very blackboxy')
    def test_diagram_is_censored_but_viable(self):
        if not FLIPPERS.graybox_enabled:
            return True

        value = pre.DRT_NI_x_OE_COUNT_x_TF(
            model='GBR logy;lr=0.05;n=1000;mf=0.5;md=[1, 3, 5];t_m=RMSE',
            available_types=set(['N', 'C', 'T']), count_args='cmin=33',
            fs_rf_args='e=1;nt=2000;mf=1', mi_type='median', selection_type=None)

        self.assertIn('diagram', value.keys())
        dia = json.loads(value['diagram'])

        # If this failed, then something else failed and the blackbox was
        # called - blackbox doesn't even have TXT
        self.assertIn('TXT', dia['taskMap'])

    def test_any_variable_selection_is_blackbox(self):
        if not FLIPPERS.graybox_enabled:
            return True

        value = pre.DRT_NI_x_OE_COUNT_x_TF(
                model='GBR', available_types=set(['N', 'C', 'T']))

        dia = json.loads(value['diagram'])
        self.assertNotIn('RFR', dia['taskMap'].keys())

    def test_any_variable_selection_is_blackbox_alt_case(self):
        if not FLIPPERS.graybox_enabled:
            return True

        value = pre.DRN_NI_x_HOT_x_RTM(
                model='GBR', available_types=set(['N', 'C', 'T']),
                selection_type='RF')

        dia = json.loads(value['diagram'])
        self.assertNotIn('RFR', dia['taskMap'].keys())

