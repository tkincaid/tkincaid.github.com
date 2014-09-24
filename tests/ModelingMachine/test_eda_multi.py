import unittest
from mock import Mock, patch
import sys
import re
import os
import pandas, numpy
import random
import time
import json
import scipy.stats as stats
from pyace import ace

import pytest

mm = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(mm,'../..'))

import ModelingMachine.engine.eda_multi
import ModelingMachine.engine.eda_multi as eda_multi
from common.services.flippers import FLIPPERS

from common.services.eda import EdaService
from common.services.project import ProjectServiceBase
from bson import ObjectId

from config.test_config import db_config
from common.wrappers import database


@pytest.mark.db
class TestEdaStreamReturns(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.persistent = database.new('persistent')

    def setUp(self):
        self.persistent.destroy(table='eda')
        self.persistent.destroy(table='eda_map')
        self.persistent.destroy(table='project')
        self.rng = numpy.random.RandomState(123)
        self.uid = ObjectId()
        self.pid = ProjectServiceBase.create_project(self.uid)

    @classmethod
    def tearDownClass(cls):
        cls.persistent.destroy(table='eda')
        cls.persistent.destroy(table='eda_map')
        cls.persistent.destroy(table='project')

    def assert_eda_equal(self, reference, computed):
        '''TODO: This was copy-pasted from test_worker'''
        self.assertEqual(set(reference.keys()), set(computed.keys()))
        for key in reference.keys():
            self.assertEqual(set(reference[key].keys()),
                             set(computed[key].keys()))
            for subkey in reference[key].keys():
                if subkey == 'profile':
                    self.assert_eda_profile_equal(reference[key][subkey],
                                                  computed[key][subkey])
                elif subkey == 'metric_options':
                    # non deterministic ordering due to dict iteration
                    a = sorted(reference[key][subkey])
                    b = sorted(computed[key][subkey])
                    msg_if_fail = ('{}.{} differs:\n\tReference: {}'
                        '\n\tComputed: {}').format(
                            key, subkey, a, b)
                    self.assertEqual(a, b, msg_if_fail)
                else:
                    msg_if_fail = ('{}.{} differs:\n\tReference: {}'
                        '\n\tComputed: {}').format(
                            key, subkey, reference[key][subkey],
                            computed[key][subkey])
                    self.assertEqual(reference[key][subkey],
                                     computed[key][subkey], msg_if_fail)

    def assert_eda_profile_equal(self, reference, computed):
        '''TODO: This was copy-pasted from test_worker'''
        numerical_close_keys = ['info', 'raw_info']

        self.assertEqual(set(reference.keys()), set(computed.keys()))
        for key in reference.keys():
            if key in numerical_close_keys:
                np.testing.assert_almost_equal(reference[key], computed[key], 5)
            elif key in ('plot', 'plot2'):
                # FIXME dunno why plot doesnt match teh fixture
                continue
            else:
                self.assertEqual(reference[key], computed[key], key)

    def test_return_of_eda1_is_same_as_stored_in_db(self):
        eda_service = EdaService(self.pid, self.uid, 'universe')
        nsamples = 500
        df = pandas.DataFrame({
            'targ': numpy.linspace(0, 1000, nsamples),
            'many_values': [str(i)+'x' for i in range(10000, 10000 + nsamples)],
            'cat': self.rng.choice(['a', 'b', 'c', 'd'], nsamples)})

        eda_doc, feature_list = eda_multi.eda_stream(
            df, progress=Mock(), eda_service=eda_service)

        # Don't use eda_service.get because it does some mumbo jumbo on top
        stored_eda = self.persistent.read(table='eda',
                                          condition={'pid':self.pid},
                                          result={})

        self.maxDiff = None
        self.assert_eda_equal(stored_eda['eda'], eda_doc)

    def test_eda_multi_can_work_with_only_one_column(self):
        eda_service = EdaService(self.pid, self.uid, 'universe')

        # Arrange: Create dummy EDA record
        eda_id = eda_service._get_column_location('targ')
        existing_column_eda = {'value' : 'unmodified-value'}

        self.persistent.update(table='eda', condition={'_id': ObjectId(eda_id)},
            values = {'eda' : {'cat' : existing_column_eda}})

        stored_eda = eda_service.get()
        existing_eda_columns = dict([(i['id'],i) for i in stored_eda])

        # Act: Execute 1-column EDA
        nsamples = 10
        df = pandas.DataFrame({
            'targ': numpy.linspace(0, 1000, nsamples),
            'many_values': [str(i)+'x' for i in range(10000, 10000 + nsamples)],
            'cat': self.rng.choice(['a', 'b', 'c', 'd'], nsamples)})

        metadata = {
            'columns': [
                [ 0, 'targ', 0 ],
                [ 1, 'many_values', 0 ],
                [ 2, 'cat', 0 ]
            ]
        }

        result, _ = eda_multi.eda_stream(df, eda_service=eda_service,
            progress=Mock(), column_names=['targ'], metadata = metadata,
            eda_doc = existing_eda_columns)


        # Assert: We got new EDA for 1 column and the existing EDA was not modified
        stored_eda = self.persistent.read(table='eda',
                                          condition={'pid':self.pid},
                                          result=[])

        self.assertEqual(len(stored_eda), 1)
        eda = stored_eda[0]
        self.assertEqual(str(eda['_id']), eda_id)
        for k in ['profile', 'summary']:
            self.assertTrue(eda['eda']['targ'].get(k), '{} not found in EDA record'.format(k))
        self.assertEqual(eda['eda']['cat'], existing_column_eda)


class TestEdaMulti(unittest.TestCase):
    @patch("ModelingMachine.engine.eda_multi.varProfile")
    @patch("ModelingMachine.engine.eda_multi.time.time")
    def test_eda(self, timeMock, varProfileMock):
        conn = Mock()
        pipe = Mock()
        arr = numpy.ndarray((100, 5))
        arr.fill(1) # Fill a 100x5 multidim array with 1's
        dsn = pandas.DataFrame(arr, columns=["A", "B", "C", "D", "E"])
        col_list = [(0, "A"), (2, "C"), (4, "E")]
        col_names = ["A", "C", "E"]
        pid = "528d0390a69cec1ea16ff65b"
        dataset_id = "some_dataset_id"
        freq = 1
        def eda_conn_create(keyname, index, values):
            self.assertEqual(index, pid)
            self.assertLessEqual(len(values), len(col_list))
            self.assertGreaterEqual(len(values), 1)
            col_dict = dict(col_list)
            for colid in values:
                self.assertIn(int(colid), col_dict)
        def eda_varProfile(col):
            return {"name": col.name}
        conn.create.side_effect = eda_conn_create
        varProfileMock.side_effect = eda_varProfile
        ModelingMachine.engine.eda_multi.eda(pipe, dsn,
            target_name=None, target_type=None, col_list=col_names, eda_doc={})

    @patch("ModelingMachine.engine.eda_multi.time.time")
    @patch("ModelingMachine.engine.eda_multi.Workspace.get_metadata")
    def test_eda(self, getMetadataMock, timeMock):
        conn = Mock()
        pipe = Mock()
        arr = numpy.ndarray((100, 5))
        dsn = pandas.DataFrame({
            'y': ['$0','$1','2'],
            'x1': [1,2,3],
            'x2': [1,2,3]
        })
        col_list = [(0, "y"), (2, "x1"), (4, "x2")]
        col_names = ["y", "x1", "x2"]
        pid = "528d0390a69cec1ea16ff65b"
        dataset_id = "some_dataset_id"
        freq = 1
        def eda_conn_create(keyname, index, values):
            self.assertEqual(index, pid)
            self.assertLessEqual(len(values), len(col_list))
            self.assertGreaterEqual(len(values), 1)
            col_dict = dict(col_list)
            for colid in values:
                self.assertIn(int(colid), col_dict)
        def eda_varProfile(col):
            return {"name": col.name}
        conn.create.side_effect = eda_conn_create
        eda_doc = {'y':
            {'types':{'date':False,'nastring':False,'currency':False,'percentage':False,'length':False},
             'low_info':{},
            }

        }
        ModelingMachine.engine.eda_multi.eda(pipe, dsn,
            target_name='y', target_type='Regression', col_list=col_names, eda_doc=eda_doc)

    @pytest.mark.unit
    def test_eda_maps_binary_target(self):
        df = pandas.DataFrame({'Feat1': [5,2,3,4,5], 'Targ': ['a','b','a','b','a']})
        pipe = Mock()
        eda_doc = {'Targ': {'summary':[2], 'types':{'date':False,'nastring':False,'currency':False,'percentage':False,'length':False}}}
        with patch('ModelingMachine.engine.eda_multi._send_report') as srmock:
            result = eda_multi.eda(pipe, df, 'Targ', 'Classification', ['Feat1'], eda_doc,
                          target_metric='LogLoss', part_info={'holdout_pct': 0})
            print srmock.call_args
            eda_result = srmock.call_args[0][1]
        # look for histogram & ACE
        self.assertIn('plot',eda_result['profile'])
        self.assertIn('info',eda_result['profile'])

    def test_date_formatting(self):
        pipe = Mock()
        dsn = pandas.DataFrame({'a':[1,2,3,4,5],'b':['2009-10-11','0001-10-10','2010-06-03','2011-11-09','2010-04-11']})
        with patch('ModelingMachine.engine.eda_multi._send_report') as srmock:
            eda_multi.eda(pipe,dsn,'a','Regression',['b'],{'a':{'summary':[12], 'types':{'date':'','nastring':'','percentage':'','length':'','currency':''}}})
            eda_result = srmock.call_args[0][1]
            # check histogram contains dates
            self.assertTrue(all([re.match('[0-9]{4}-[0-9]{2}-[0-9]{2}',i[0]) for i in eda_result['profile']['plot']]))
            # check frequent values has one bad date
            self.assertIn('???',[i[0] for i in eda_result['profile']['plot2']])
            # check summary
            self.assertEqual(eda_result['summary'],[5, 0, '???', 293480.9085883441,'???', '2010-04-11', '2011-11-09'])

    def test_get_column_by_process(self):
        with patch.object(ModelingMachine.engine.eda_multi.multiprocessing, 'cpu_count', return_value = 100) as MockMultiprocessing:
            columns = ['x', 'y', 'z']
            pairs = ModelingMachine.engine.eda_multi.get_column_by_process(columns)
            self.assertItemsEqual(pairs, [[(0,'x')], [(1,'y')], [(2,'z')]])

        with patch.object(ModelingMachine.engine.eda_multi.multiprocessing, 'cpu_count', return_value = 4) as MockMultiprocessing:
            columns = ['x', 'y', 'z']
            pairs = ModelingMachine.engine.eda_multi.get_column_by_process(columns)
            self.assertItemsEqual(pairs, [[(0,'x')], [(1,'y')], [(2,'z')]])

        with patch.object(ModelingMachine.engine.eda_multi.multiprocessing, 'cpu_count', return_value = 3) as MockMultiprocessing:
            columns = ['x', 'y', 'z']
            pairs = ModelingMachine.engine.eda_multi.get_column_by_process(columns)
            self.assertItemsEqual(pairs, [[(0,'x')], [(1,'y')], [(2,'z')]])

        with patch.object(ModelingMachine.engine.eda_multi.multiprocessing, 'cpu_count', return_value = 2) as MockMultiprocessing:
            columns = ['x', 'y', 'z']
            pairs = ModelingMachine.engine.eda_multi.get_column_by_process(columns)
            self.assertItemsEqual(pairs, [[(0, 'x'), (2, 'z')], [(1, 'y')]])

        with patch.object(ModelingMachine.engine.eda_multi.multiprocessing, 'cpu_count', return_value = 1) as MockMultiprocessing:
            columns = ['x', 'y', 'z']
            pairs = ModelingMachine.engine.eda_multi.get_column_by_process(columns)
            self.assertItemsEqual(pairs, [[(0, 'x'), (1, 'y'), (2, 'z')]])

    def x_test_remove_duplicates(self):
        self.init_workspace('dup_test.csv')
        X = self.workspace.get_dataframe()
        self.assertTrue(np.all(X.columns==['a','b','c']))
        self.assertTrue(np.all(X.dtypes==['int64','int64','int64']))

    def x_test_remove_low_info(self):
        self.init_workspace('low_info.csv')
        X = self.workspace.get_dataframe()
        self.assertTrue(np.all(X.columns==['a','b']))
        self.assertTrue(np.all(X.dtypes==['int64','int64']))


class TestChooseAppropriateMetrics(unittest.TestCase):

    @pytest.mark.unit
    def test_eda_with_infs_doesnot_die_at_ace(self):
        s1 = pandas.Series(numpy.arange(10))
        s2 = pandas.Series(numpy.arange(10) - 5)
        df = pandas.DataFrame({'Feat1': s1/s2, 'Targ': numpy.arange(10)})
        pipe = Mock()
        eda_multi.eda(pipe, df, 'Targ', 'Regression', ['Feat1'],
                      {'Targ':
                        {'summary':[10],
                        'types':{'date':False,'nastring':False,'currency':False,'percentage':False,'length':False}}},
                      target_metric='LogLoss', part_info={'holdout_pct':0})
        # Nothing to assert? Maybe we should factor out that ACE bit

    @pytest.mark.unit
    def test_choose_appropriate_metrics_smoke(self):
        '''Every call should return a dict with 'all', 'recommended'

        Additionally, we want each metric to just have short_name
        '''
        X = pandas.Series(numpy.random.randn(50))
        fake_eda = {'summary':generate_summary(X),
                    'profile': {'type':'N'}}
        metrics = eda_multi.choose_appropriate_metrics(X, fake_eda)
        self.assertIn('all', metrics)
        self.assertIn('recommended', metrics)

        for m in metrics['all']:
            keys = m.keys()
            self.assertEqual(set(keys), set(['short_name']))

    @pytest.mark.unit
    def test_normal_dist_best_acc_is_rmse(self):
        X = pandas.Series(numpy.random.randn(50))
        fake_eda = {'summary':generate_summary(X),
                    'profile': {'type':'N'}}
        col_metrics = eda_multi.choose_appropriate_metrics(X, fake_eda)
        acc = col_metrics['recommended']['default']
        self.assertEqual(acc['short_name'], eda_multi.metrics.RMSE)


    @pytest.mark.unit
    def test_binary_best_acc_is_logloss(self):
        X = pandas.Series((numpy.random.rand(50) > 0.5).astype(numpy.float))
        fake_eda = {'summary':generate_summary(X),
                    'profile': {'type':'N'}}

        col_metrics = eda_multi.choose_appropriate_metrics(X, fake_eda)
        acc = col_metrics['recommended']['default']
        self.assertEqual(acc['short_name'], eda_multi.metrics.LOGLOSS)

    @pytest.mark.unit
    def test_positive_gaussian_ish_best_acc_is_rmse(self):
        data = numpy.random.randn(50)
        data = data - numpy.min(data) + 1
        X = pandas.Series(data)
        fake_eda = {'summary':generate_summary(X),
                    'profile': {'type':'N'}}

        col_metrics = eda_multi.choose_appropriate_metrics(X, fake_eda)
        acc = col_metrics['recommended']['default']
        self.assertEqual(acc['short_name'], eda_multi.metrics.RMSE)

    @pytest.mark.unit
    def test_gamma_available(self):
        data = numpy.random.randn(50)
        data = data - numpy.min(data) + 1
        X = pandas.Series(data)
        fake_eda = {'summary':[50, None, None, None, 1, 3, 6],
                    'profile': {'type':'N'}}

        col_metrics = eda_multi.choose_appropriate_metrics(X, fake_eda)
        self.assertIn(eda_multi.metrics.GAMMA_DEVIANCE,
                      [i['short_name'] for i in col_metrics['all']])

    @pytest.mark.unit
    def test_categorical_best_acc_is_logloss_best_rank_is_auc(self):
        '''For if we allow the user to specify a one-against-all classification
        '''
        choices = ['I', 'do', 'not', 'like', 'green', 'eggs', 'and', 'ham']
        data = [random.choice(choices) for i in xrange(50)]
        X = pandas.Series(data)

        fake_eda = {'summary':generate_summary(X),
                    'profile':{'type':'C'}}

        col_metrics = eda_multi.choose_appropriate_metrics(X, fake_eda)
        acc = col_metrics['recommended']['default']
        rec = col_metrics['recommended']['recommender']
        self.assertEqual(acc['short_name'], eda_multi.metrics.LOGLOSS)
        self.assertEqual(rec['short_name'], eda_multi.metrics.RMSE)
        self.assertIn(eda_multi.metrics.LOGLOSS, [i['short_name'] for i in col_metrics['all']])
        self.assertIn(eda_multi.metrics.AUC, [i['short_name'] for i in col_metrics['all']])

    @pytest.mark.unit
    def test_poisson_case(self):
        numpy.random.seed(12345)
        data = numpy.random.poisson(lam=0.2, size=1000)
        X = pandas.Series(data)
        fake_eda = {'summary': generate_summary(X),
                    'profile': {'type':'N'}}

        col_metrics = eda_multi.choose_appropriate_metrics(X, fake_eda)
        acc = col_metrics['recommended']['default']
        self.assertEqual(acc['short_name'], eda_multi.metrics.POISSON_DEVIANCE)

    @pytest.mark.unit
    def test_lognormal_case(self):
        numpy.random.seed(12345)
        data = numpy.exp(numpy.random.randn(500) * 100)
        X = pandas.Series(data)
        fake_eda = {'summary': generate_summary(X),
                    'profile': {'type':'N'}}

        col_metrics = eda_multi.choose_appropriate_metrics(X, fake_eda)
        acc = col_metrics['recommended']['default']
        self.assertEqual(acc['short_name'], eda_multi.metrics.GAMMA_DEVIANCE)

    @pytest.mark.unit
    def test_non_0_1_is_still_binary(self):
        '''Instead of requiring that the inputs be 0/1, allow any two class
        column to be recommded as binary.  This used to be the default
        '''
        numpy.random.seed(12345)
        levels = [1,2]
        data = numpy.random.choice(levels, 100)
        X = pandas.Series(data)
        fake_eda = {'summary': generate_summary(X),
                    'profile': {'type':'N'}}

        col_metrics = eda_multi.choose_appropriate_metrics(X, fake_eda)
        acc = col_metrics['recommended']['default']
        rec = col_metrics['recommended']['recommender']
        self.assertEqual(acc['short_name'], eda_multi.metrics.LOGLOSS)
        self.assertEqual(rec['short_name'], eda_multi.metrics.RMSE)

    @pytest.mark.unit
    def test_cat_twoclass_recommends_binary_metrics(self):
        numpy.random.seed(12345)
        levels = ['cat','dog']
        data = numpy.random.choice(levels, 100)
        X = pandas.Series(data)
        fake_eda = {'summary': generate_summary(X),
                    'profile': {'type':'C'}}

        col_metrics = eda_multi.choose_appropriate_metrics(X, fake_eda)
        acc = col_metrics['recommended']['default']
        rec = col_metrics['recommended']['recommender']
        self.assertEqual(acc['short_name'], eda_multi.metrics.LOGLOSS)
        self.assertEqual(rec['short_name'], eda_multi.metrics.RMSE)


    @pytest.mark.unit
    def test_metrics_recommend_uses_profiled_vartype(self):
        numpy.random.seed(12345)
        data = list(numpy.random.rand(100))
        data[35] = '.'
        X = pandas.Series(data)
        summary, l_i, Xstar = eda_multi._get_summary_plus(X, {'types': {'nastring': True}})
        fake_eda = {'summary': summary,
                    'profile': {'type':'N'}}

        col_metrics = eda_multi.choose_appropriate_metrics(Xstar, fake_eda)
        acc = col_metrics['recommended']['default']
        rank = col_metrics['recommended']['recommender']
        self.assertNotEqual(acc['short_name'], eda_multi.metrics.LOGLOSS)
        self.assertNotEqual(rank['short_name'], eda_multi.metrics.AUC)


    @pytest.mark.unit
    def test_get_summary_high_card(self):
        X = pandas.Series([str(i) + 'A' for i in range(1000)])
        summary, low_infos, col = eda_multi._get_summary_plus(X, {'types':{}})
        self.assertIn('high_cardinality', low_infos)
        self.assertTrue(low_infos['high_cardinality'])

    @pytest.mark.unit
    def test_metrics_numeric_mets_is_mad(self):
        '''We were recommending RMSLE for New_York_Mets.csv - MAD better
        '''
        bank = [20]*46 + [30]*2 + [35]*1 + [40]*1 + [45]*1 + [50]*1
        X = pandas.Series(bank)
        summary, l_i, Xstar = eda_multi._get_summary_plus(X, {'types':{}})
        fake_eda = {'summary': summary,
                    'profile': {'type':'N'}}

        col_metrics = eda_multi.choose_appropriate_metrics(Xstar, fake_eda)
        acc = col_metrics['recommended']['default']
        rec = col_metrics['recommended']['recommender']
        self.assertEqual(acc['short_name'], eda_multi.metrics.MAD)
        self.assertEqual(rec['short_name'], eda_multi.metrics.MAD)

    @pytest.mark.unit
    def test_regression_metrics_with_negative_values_doesnot_approve_gamma(self):
        X = pandas.Series(numpy.linspace(-1, 10, 100))
        summary, l__i, Xstar = eda_multi._get_summary_plus(X, {'types':{}})
        fake_eda = {'summary': summary,
                    'profile': {'type': 'N'}}
        col_metrics = eda_multi.choose_appropriate_metrics(Xstar, fake_eda)
        chosen_metrics = col_metrics['all']

        self.assertNotIn(eda_multi.metrics.GAMMA_DEVIANCE,
                         [i['short_name'] for i in chosen_metrics])

    @pytest.mark.unit
    def test_min_inflated_is_MAD_and_GINI_NORM(self):
        rng = numpy.random.RandomState(1)
        first_half = numpy.zeros(50)
        sec_half = rng.rand(50) * numpy.linspace(1.0, 0.0, 50)
        data = numpy.append(first_half, sec_half)
        X = pandas.Series(data)
        summary, l__i, Xstar = eda_multi._get_summary_plus(X, {'types':{}})
        fake_eda = {'summary': summary,
                    'profile': {'type': 'N'}}
        col_metrics = eda_multi.choose_appropriate_metrics(Xstar, fake_eda)
        acc = col_metrics['recommended']['default']
        rec = col_metrics['recommended']['recommender']
        self.assertEqual(acc['short_name'], eda_multi.metrics.MAD)
        self.assertEqual(rec['short_name'], eda_multi.metrics.MAD)

    @pytest.mark.unit
    def test_rmsle_not_allowed_for_negative_response(self):
        rng = numpy.random.RandomState(1)
        data = numpy.exp(10 *rng.rand(100))
        data[0] = -0.3
        X = pandas.Series(data)
        summary, l_i, Xstar = eda_multi._get_summary_plus(X, {'types':{}})
        fake_eda = {'summary': summary,
                    'profile': {'type': 'N'}}
        col_metrics = eda_multi.choose_appropriate_metrics(Xstar, fake_eda)
        chosen_metrics = col_metrics['all']
        self.assertNotIn(eda_multi.metrics.RMSLE,
                         [i['short_name'] for i in chosen_metrics])

    @pytest.mark.unit
    def test_dates_not_seen_as_categorical(self):
        '''Issue #3072'''
        eda_report = {
                u'id': 9,
                u'low_info': {
                    u'duplicate': False,
                    u'empty': False,
                    u'few_values': False,
                    u'high_cardinality': False,
                    u'ref_id': False},
                u'name': u'saledate',
                u'profile': {
                    u'miss_count': 0.0,
                    u'miss_ymean': None,
                    u'name': u'saledate',
                    u'plot': u'Not worth faking for this test',
                    u'plot2': u'Not worth faking for this test',
                    u'type': u'D',
                    u'y': None},
                u'raw_variable_index': 9,
                u'summary': [148,
                    0,
                    u'11/03/2007 23:41',
                    64688951.24722292,
                    u'01/20/2004 00:00',
                    u'08/02/2007 00:00',
                    u'08/25/2011 00:00'],
                u'transform_args': [],
                u'transform_id': 0,
                u'types': {
                    u'category': False,
                    u'conversion': u'%m/%d/%Y %H:%M',
                    u'currency': False,
                    u'date': True,
                    u'length': False,
                    u'nastring': False,
                    u'numeric': True,
                    u'percentage': False,
                    u'text': False}}

        # By this point in the EDA function, the date strings have been
        # converted to timestamps
        first_date = 1074660240.0
        last_date = 1314330240.0
        dates = numpy.linspace(first_date, last_date, 100)
        col_metrics = eda_multi.choose_appropriate_metrics(dates, eda_report)
        selected = col_metrics['recommended']['default']['short_name']
        self.assertNotEqual(eda_multi.metrics.LOGLOSS, selected)  # Original
        self.assertEqual(eda_multi.metrics.RMSE, selected)  # Much better


    @pytest.mark.unit
    def test_lengths_not_seen_as_categorical(self):
        '''Issue #3072 (well, a similar case)'''
        eda_report = {
                u'id': 44,
                u'low_info': {
                    u'duplicate': False,
                    u'empty': False,
                    u'few_values': False,
                    u'high_cardinality': False,
                    u'ref_id': False},
                u'name': u'Stick_Length',
                u'profile': {
                    u'miss_count': 385.0,
                    u'miss_ymean': None,
                    u'name': u'Stick_Length',
                    u'plot': 'Not worth faking',
                    u'plot2': 'Not worth faking',
                    u'type': u'L',
                    u'y': None},
                u'raw_variable_index': 44,
                u'summary': [9,
                    284,
                    128.66666666666666,
                    17.37303146322547,
                    114.0,
                    126.0,
                    189.0],
                u'transform_args': [],
                u'transform_id': 0,
                u'types': {u'category': False,
                    u'conversion': u'L',
                    u'currency': False,
                    u'date': False,
                    u'length': True,
                    u'nastring': False,
                    u'numeric': True,
                    u'percentage': False,
                    u'text': False}}


        # By this point in the EDA function, the date length strings have been
        # converted to floats
        first_length = 24
        last_length = 120
        lengths = numpy.linspace(first_length, last_length, 100)
        col_metrics = eda_multi.choose_appropriate_metrics(lengths, eda_report)
        self.assertNotEqual(eda_multi.metrics.LOGLOSS,
                            col_metrics['recommended']['default']['short_name'])

    @pytest.mark.unit
    def test_percents_not_seen_as_categorical(self):
        eda_report = {
            u'id': 2,
            u'low_info': {u'duplicate': False,
                u'empty': False,
                u'few_values': False,
                u'high_cardinality': False,
                u'ref_id': False},
            u'name': u'Percents',
            u'profile': {u'miss_count': 0.0,
                u'miss_ymean': None,
                u'name': u'Percents',
                u'plot': 'Not worth faking',
                u'plot2': 'Not worth faking',
                u'type': u'P',
                u'y': None},
            u'raw_variable_index': 2,
            u'summary': [12, 0, 6.5, 3.452052529534663, 1.0, 6.5, 12.0],
            u'transform_args': [],
            u'transform_id': 0,
            u'types': {u'category': False,
                u'conversion': u'P',
                u'currency': False,
                u'date': False,
                u'length': False,
                u'nastring': False,
                u'numeric': True,
                u'percentage': True,
                u'text': False}}
        # Column will have been converted to floats by this point
        pcts = pandas.Series(numpy.linspace(1, 1200, 100))
        col_metrics = eda_multi.choose_appropriate_metrics(pcts, eda_report)
        self.assertNotEqual(eda_multi.metrics.LOGLOSS,
                            col_metrics['recommended']['default']['short_name'])

    @pytest.mark.unit
    @patch('ModelingMachine.engine.metrics.FLIPPERS', autospec=True)
    def test_gini_not_in_metric_list_feature_flipper_false(self, mock_flippers):
        mock_flippers.allow_gini = False
        X = pandas.Series(numpy.linspace(-1, 10, 100))
        summary, l__i, Xstar = eda_multi._get_summary_plus(X, {'types':{}})
        fake_eda = {'summary': summary,
                    'profile': {'type': 'N'}}
        col_metrics = eda_multi.choose_appropriate_metrics(Xstar, fake_eda)
        chosen_metrics = col_metrics['all']

        self.assertNotIn(eda_multi.metrics.GINI,
                         [i['short_name'] for i in chosen_metrics])

    @pytest.mark.unit
    @patch('ModelingMachine.engine.metrics.FLIPPERS', autospec=True)
    def test_gini_not_in_metric_list_feature_flipper_true(self, mock_flippers):
        mock_flippers.allow_gini = True
        X = pandas.Series(numpy.linspace(-1, 10, 100))
        summary, l__i, Xstar = eda_multi._get_summary_plus(X, {'types':{}})
        fake_eda = {'summary': summary,
                    'profile': {'type': 'N'}}
        col_metrics = eda_multi.choose_appropriate_metrics(Xstar, fake_eda)
        chosen_metrics = col_metrics['all']

        self.assertIn(eda_multi.metrics.GINI,
                         [i['short_name'] for i in chosen_metrics])

class TestACE(unittest.TestCase):
    def test_perfect_score(self):
        ''' check that predictors that are correlated with target get a really high score '''
        df = pandas.DataFrame({'x1': [1,2,3,4,5,6,7,8,9,10],'x2':[2,3.1,1,1.3,2,0,3,0,2,3],'y':[2,4,6,8,10,12,14,16,18,20]})
        info = ace(df, 'y', [], cv=False)
        self.assertEqual(info[0][0],1)
        info = ace(df, 'y', [], cv=True)
        self.assertAlmostEqual(info[0][0],0.9757575757575758)

    def test_data_with_na(self):
        ''' make sure N/As in the data don't cause problems '''
        df = pandas.DataFrame({'x1': [5,2,3,4,float('nan'),6,2,8,9,10],'x2':[2,3.1,1,1.3,2,0,3,0,2,3],'y':[2,4,6,8,10,12,14,16,float('nan'),20]})
        info = ace(df, 'y', [], cv=False)
        self.assertAlmostEqual(info[0][0],0.8811970965696457)
        info = ace(df, 'y', [], cv=True)
        self.assertAlmostEqual(info[0][0],0.0084934208055749139)

    def test_random_predictor(self):
        ''' random predictors and constants should get a score close to zero '''
        # check random predictor
        df = pandas.DataFrame({'x1': numpy.random.random(5000), 'y':numpy.arange(5000)})
        info = ace(df, 'y', [], cv=True)
        self.assertAlmostEqual(info[0][0],0.0,places=2)
        # check constants
        df = pandas.DataFrame({'x1': numpy.ones(5000), 'y':numpy.arange(5000)})
        info = ace(df, 'y', [], cv=True)
        self.assertAlmostEqual(info[0][0],0.0,places=2)

    def test_credibility(self):
        ''' test that credibility gets better scores for uncorrelated cat vars '''
        df = pandas.DataFrame({'x1': [1,2,3,4,5,5,4,3,2,1],'x2':[2,3.1,1,1.3,2,0,3,0,2,3],'y':[2,4,6,8,10,12,14,16,18,20]})
        info1 = ace(df, 'y', ['x1'], cv=True, K=0)
        info2 = ace(df, 'y', ['x1'], cv=True, K=10)
        self.assertGreater(info2[0][0],info1[0][0])

    def test_data_with_cat_predictor(self):
        ''' check that cat vars are handled properly
            (ie. high cardinality vars don't get a really
            high score '''
        df = pandas.DataFrame({'x1': [1,2,3,4,5,6,7,8,9,10],'x2':[2,3.1,1,1.3,2,0,3,0,2,3],'y':[2,4,6,8,10,12,14,16,18,20]})
        info = ace(df, 'y', ['x1'], cv=False)
        self.assertAlmostEqual(info[0][0],0.0)
        info = ace(df, 'y', ['x1'], cv=True)
        self.assertAlmostEqual(info[0][0],-0.015900501614787066)

    def test_data_with_cat_response(self):
        ''' check that ace can handle binary classification '''
        df = pandas.DataFrame({'x1': [1,2,3,4,5,6,7,8,9,10],'x2':[2,3.1,1,1.3,2,0,3,0,2,3],'y':[1,0,1,1,0,0,0,1,0,0]})
        info = ace(df, 'y', ['y'], cv=False)
        self.assertAlmostEqual(info[0][0],0.1864956845854453)
        info = ace(df, 'y', ['y'], cv=True)
        self.assertAlmostEqual(info[0][0],-0.19553434939544601)

    def test_weights(self):
        ''' check weighted ACE '''
        # create a data frame where half is correlated and half is not
        df = pandas.DataFrame({'x1': [1,2,3,4,5,1,1,1,1,1],'x2':[2,3.1,1,1.3,2,0,3,0,2,3],'y':[2,4,6,8,10,12,14,16,18,20]})
        # check that giving more weight to the correlated half of the
        # array gives a higher score
        weights = numpy.array([1,1,1,1,1,2,2,2,2,2])
        df['w'] = weights
        info1 = ace(df, 'y', [], cv=False, weight='w')
        weights = numpy.array([2,2,2,2,2,1,1,1,1,1])
        df['w'] = weights
        info2 = ace(df, 'y', [], cv=False, weight='w')
        self.assertGreater(info2[0][0],info1[0][0])
        # check that higher weights give higher and lower scores
        df = pandas.DataFrame({'x1':numpy.concatenate((numpy.arange(50),numpy.ones(50))),'y':numpy.arange(100)*2})
        # get scores with 1 & 2 weights
        weights = numpy.array(numpy.concatenate((numpy.ones(50),numpy.ones(50)*2)))
        df['w'] = weights
        info3 = ace(df, 'y', [], cv=True, weight='w')
        weights = numpy.array(numpy.concatenate((numpy.ones(50)*2,numpy.ones(50))))
        df['w'] = weights
        info4 = ace(df, 'y', [], cv=True, weight='w')
        self.assertGreater(info4[0][0],info3[0][0])
        # get scores with 1 & 3 weights
        weights = numpy.array(numpy.concatenate((numpy.ones(50),numpy.ones(50)*3)))
        df['w'] = weights
        info5 = ace(df, 'y', [], cv=True, weight='w')
        weights = numpy.array(numpy.concatenate((numpy.ones(50)*3,numpy.ones(50))))
        df['w'] = weights
        info6 = ace(df, 'y', [], cv=True, weight='w')
        # uncorrelated half of data should decrease when weights are increased
        self.assertLess(info5[0][0],info3[0][0])
        # correlated half of data should increase when weights are increased
        self.assertGreater(info6[0][0],info4[0][0])


def generate_summary(column):
    '''Just a wrapper for the code in eda_multi

    Parameters
    ----------
    column : pandas.Series
        The column to analyze

    Returns
    -------
    summary : list
        list of values.  See eda_multi for the explanation of the positions

    '''
    summary, l_i, conv = eda_multi._get_summary_plus(column, {'types':{}})
    return summary

if __name__ == '__main__':
    unittest.main()
