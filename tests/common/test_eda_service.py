import os
import json
import unittest
import pytest
from bson.objectid import ObjectId
from mock import patch, Mock, DEFAULT

import numpy as np
import pandas as pd

from config.engine import EngConfig
from config.test_config import db_config as config
from common.wrappers import database

from common.services.project import ProjectServiceBase as ProjectService
from common.services.autopilot import AutopilotService, DONE, MANUAL, SEMI, AUTO
import common.services.eda as eda_service_mod
from common.engine.progress import ProgressSink
import common
from common import load_class

from MMApp.entities.user import UserService
from MMApp.entities.roles import RoleProvider
from MMApp.entities.jobqueue import QueueService
from MMApp.entities.permissions import Roles, Permissions

from common.services.eda import EdaService

from ModelingMachine.engine.eda_multi import eda_stream

from base_test_services import TestServiceBase

class TestProjectClass(TestServiceBase):

    def test_get_eda(self):
        username, uid, pid, dataset_id = self.create_project()

        eda = EdaService(pid,uid)

        out = eda.get()

        self.assertEqual(len(out), 2)
        for item in out:
            self.assertGreater(set(item.keys()), {'id','profile','summary'})

    def test_column_importances(self):
        username, uid, pid, dataset_id = self.create_project()

        eda = EdaService(pid,uid)

        out = eda.column_importances()

        self.assertEqual([i for i,k in out], ['b','a'])
        self.assertEqual([k for i,k in out], [0.2,0.1])

    @patch.object(common.services.eda.EdaService,'assert_has_permission')
    def test_wide_data(self,*args):
        uid = ObjectId()
        pid = ObjectId()

        eda = EdaService(pid,uid)
        eda_doc = {}
        eda_check = []
        max_batch_size = EngConfig['MAX_EDA_BATCH']
        test_blocks = 3
        for i in range(test_blocks * max_batch_size):
            eda_doc['col%07d' % i] = { 'key': 'value', 'metric_options': {'all':['x']} }
            eda_check.append({'id': 'col%07d' % i, 'key': 'value', 'metric_options': {'all':['x']} })
        eda.update(eda_doc)
        # make sure the correct number of blocks are created
        self.assertEqual(len(eda.eda_map['block_contents'].keys()),test_blocks)
        # self.assertEqual takes forever to compare long lists, so ...
        eda_list = sorted(eda.get())
        for i,j in enumerate(eda_check):
            self.assertEqual(eda_list[i],j)
        # test get_all_metrics()
        out = eda.get_all_metrics('col0000010')
        self.assertEqual(out,['x'])

    def test_get_target_metrics_list_raises_exception_when_no_eda_was_found(self):
        uid = ObjectId()
        pid = ObjectId()
        eda_service = EdaService(pid,uid, verify_permissions = False)

        with patch.multiple(eda_service, persistent = DEFAULT) as mocks:
            with self.assertRaises(ValueError):
                eda_service.get_target_metrics_list('does-not-matter')

class EdaTestMixin(object):

    def assert_recursive_dict_equal(self, source, cloned, keyname='root'):
        print('Checking {}'.format(keyname))
        print('Source: {}'.format(source))
        print('Cloned: {}'.format(cloned))
        self.assertEqual(set(source.keys()), set(cloned.keys()))
        for key in source.keys():
            print('\tchecking {}'.format(key))
            source_attr = source[key]
            cloned_attr = cloned[key]
            if isinstance(source_attr, dict):
                self.assert_recursive_dict_equal(source_attr, cloned_attr, keyname=keyname+'.'+key)
            else:
                self.assertEqual(source_attr, cloned_attr)

@pytest.mark.integration
class TestCloneEdaIntegration(unittest.TestCase, EdaTestMixin):

    @classmethod
    def setUpClass(cls):
        cls.persistent = database.new('persistent')

    def setUp(self):
        self.persistent.destroy(table='eda')
        self.persistent.destroy(table='eda_map')
        self.persistent.destroy(table='project')
        self.rng = np.random.RandomState(123)
        self.uid = ObjectId()
        self.pid = ProjectService.create_project(self.uid)

    @classmethod
    def tearDownClass(cls):
        cls.persistent.destroy(table='eda')
        cls.persistent.destroy(table='eda_map')
        cls.persistent.destroy(table='project')



    def test_eda1_plots_should_get_cloned(self):
        eda_service = EdaService(self.pid, self.uid, 'universe')
        nsamples = 50
        df = pd.DataFrame({
            u'targ': np.linspace(0, 1000, nsamples),
            u'many_values': [unicode(i)+u'x' for i in range(100, 100 + nsamples)],
            u'cat': self.rng.choice([u'a', u'b', u'c', u'd'], nsamples)})

        eda_doc, feature_list = eda_stream(
            df, progress=Mock(), eda_service=eda_service)

        fake_metadata = {'pid': ObjectId(), 'columns': [[idx, name, 0]
                                     for idx, name in enumerate(df.columns)]}

        stored_eda = self.persistent.read(table='eda',
                                          condition={'pid':self.pid},
                                          result={})['eda']

        self.maxDiff = None
        eda_clone = eda_service_mod.clone_eda_doc(stored_eda, fake_metadata)

        self.assert_recursive_dict_equal(stored_eda, eda_clone)

class TestCloneEda(unittest.TestCase, EdaTestMixin):

    TEST_DATA_DIR = os.path.join(os.path.split(__file__)[0], '..', 'testdata')

    def test_clone_copies_plots_at_eda1(self):
        '''Uses the fastiron-eda1.json test fixture'''
        ref_eda1_json_path = os.path.join(self.TEST_DATA_DIR, 'fixtures',
                                          'fastiron-eda1.json')
        with open(ref_eda1_json_path) as in_fp:
            eda1_data = json.load(in_fp)

        fake_metadata = {'pid': ObjectId(), 'columns': [[idx, col, 0] for idx, col
                                     in enumerate(eda1_data.keys())]}

        eda_clone = eda_service_mod.clone_eda_doc(eda1_data, fake_metadata)
        self.assert_recursive_dict_equal(eda1_data, eda_clone)

    def test_clone_ignores_eda2_stats(self):
        '''Uses the fastiron-eda1.json test fixture'''
        ref_eda1_json_path = os.path.join(self.TEST_DATA_DIR, 'fixtures',
                                          'fastiron-eda1.json')
        ref_eda2_json_path = os.path.join(self.TEST_DATA_DIR, 'fixtures',
                                          'fastiron-eda2.json')

        with open(ref_eda1_json_path) as in_fp:
            eda1_data = json.load(in_fp)
        with open(ref_eda2_json_path) as in_fp:
            eda2_data = json.load(in_fp)

        fake_metadata = {'pid': ObjectId(), 'columns': [[idx, col, 0] for idx, col
                                     in enumerate(eda2_data.keys())]}

        eda_clone = eda_service_mod.clone_eda_doc(eda2_data, fake_metadata)

        #Cloning at EDA2 should be the same as cloning at eda1
        self.assert_recursive_dict_equal(eda1_data, eda_clone)







if __name__ == '__main__':
    unittest.main()

