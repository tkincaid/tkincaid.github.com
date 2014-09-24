from bson import ObjectId
import datetime
import pandas as pd
import numpy as np

import copy
import unittest
import pytest
import math
from mock import patch

import common.io as io
import common.io.dataset_reader as dataset_reader
import common.io.query as pq
from common.io.dataset_reader import DatasetReader
from common.io.csv_reader import CSVReader
from ModelingMachine.engine.dstransform import TransformationEnum


class BaseReaderTestSetup(unittest.TestCase):
    nrows_fixture = 100
    metadata_fixture = {
        u'_id': ObjectId('5373c9358bd88f6a920902bc'),
        u'columns': [[0, u'x1', 0], [1, u'x2', 0], [2, u'x3', 0],
                     [3, u'y', 0]],
        u'controls': {
            u'de47acd5-4593-4d25-a750-e6b3966f4fa4': {u'encoding': u'ASCII',
                                                      u'type': u'csv'},
            u'dirty': False},
        u'created': datetime.datetime(2014, 5, 14, 19, 51, 17, 609000),
        u'files': [u'de47acd5-4593-4d25-a750-e6b3966f4fa4'],
        u'name': u'Raw Features',
        u'originalName': u'Raw Features',
        u'pid': ObjectId('5373c9338bd88f655d884b03'),
        u'shape': [nrows_fixture, 4],
        u'typeConvert': {u'x1': True, u'x2': u'', u'x3': True, u'y': True},
        u'varTypeString': u'NCNN'}

    project_fixture = {
        u'_id': ObjectId('5373c9338bd88f655d884b03'),
        u'active': 1,
        u'created': 1402165675.588697,
        u'default_dataset_id': u'5373c9358bd88f6a920902bc',
        u'holdout_pct': 20,
        u'holdout_unlocked': False,
        u'metric': u'RMSE',
        u'mode': 1,
        u'originalName': u'pretend_dataset.csv',
        u'partition': {u'folds': 5,
                       u'holdout_pct': 20,
                       u'reps': 5,
                       u'seed': 0,
                       u'total_size': 200},
        u'roles': {u'5359d6cb8bd88f5cddefd3a8': [u'OWNER']},
        u'stage': u'modeling:',
        u'target': {u'name': u'y', u'size': 0.8*nrows_fixture,
                    u'type': u'Regression'},
        u'target_options': {u'missing_maps_to': None,
                            u'name': u'y',
                            u'positive_class': None},
        u'tokens': {u'5359d6cb8bd88f5cddefd3a8': u'GGjZ_NRRMePiyQ=='},
        u'uid': u'5359d6cb8bd88f5cddefd3a8',
        u'version': 1
    }

    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(0)
        cls._raw_dataframe = pd.DataFrame({
            'x1': rng.randn(cls.nrows_fixture),
            'x2': rng.choice(['a', 'b', 'c'], cls.nrows_fixture),
            'x3': rng.randn(cls.nrows_fixture),
            'y': rng.rand(cls.nrows_fixture)})


class TestDatasetReaderInterfaceMethods(BaseReaderTestSetup):
    '''These are the calls that most parts of the app actually need to use.

    The most common will probably prove to be

    get_data - without any arguments, returns the entire dataset
               by specifying the partition in {'all', 'holdout', 'training'}
               will return the proper subset of rows.  `holdout_pct` must also
               be passed in the case that `holdout` or `training` is selected,
               but after we transition to storing the partition rows that
               will not be necessary

    get_predictors - gets the dataset sans target column.  All known
                     conversions stored within the `metadata` collection
                     will also be applied. Partition can also be specified.

    get_target_series - returns the single column, with any known expected
                        transforms.  Partition can also be specified
    '''
    def make_patched_reader(self, metadata):
        disk_reader = CSVReader.from_dict(metadata)
        disk_reader._raw_dataframe = self._raw_dataframe
        return DatasetReader(disk_reader)

    @pytest.mark.unit
    def test_default_call_gets_all_data(self):
        reader = self.make_patched_reader(self.metadata_fixture)
        data = reader.get_data()
        self.assertEqual(data.shape, (100, 4))

    @pytest.mark.unit
    def test_gets_correct_holdout_initial_implementation(self):
        '''Originally, the holdout_pct was stored in the project
        instead of the dataset, so we need to support taking it
        as an argument
        '''
        reader = self.make_patched_reader(self.metadata_fixture)
        data = reader.get_data(partition='holdout', part_info={'holdout_pct':20})

        ref = [-0.977278, 0.443863, 0.378163, 1.950775,
               -0.895467, -0.510805, -0.028182,  0.302472,
               -1.726283, 0.177426, -0.401781, 0.051945,
               0.128983, 1.139401, -1.234826,  -0.684810,
               1.054452, -0.403177, 1.222445,  0.356366]

        np.testing.assert_almost_equal(ref, data['x1'].values, 6)

    @pytest.mark.unit
    def test_gets_correct_training_initial_implementation(self):
        '''Originally, the holdout_pct was stored in the project
        instead of the dataset, so we need to support taking it
        as an argument
        '''
        reader = self.make_patched_reader(self.metadata_fixture)
        data = reader.get_data(partition='training', part_info={'holdout_pct':20})

        ref = [1.76405235, 0.40015721, 0.97873798, 2.2408932, 1.86755799,
               0.95008842, -0.15135721, -0.10321885, 0.4105985, 0.14404357,
               1.45427351, 0.76103773, 0.12167502, 0.33367433, 1.49407907,
               -0.20515826, 0.3130677, -0.85409574, -2.55298982, 0.6536186]

        np.testing.assert_almost_equal(ref, data['x1'].values[:20], 6)

    @pytest.mark.unit
    def test_get_targeted_data_applies_transforms(self):
        '''The target name and options are stored with the project.  We
        use the project information to format the dataframe when requested
        by the worker for use in
        '''
        reader = self.make_patched_reader(self.metadata_fixture)
        data = reader.get_targeted_data('training',
                                        self.project_fixture)

        ref = [1.76405235, 0.40015721, 0.97873798, 2.2408932, 1.86755799,
               0.95008842, -0.15135721, -0.10321885, 0.4105985, 0.14404357,
               1.45427351, 0.76103773, 0.12167502, 0.33367433, 1.49407907,
               -0.20515826, 0.3130677, -0.85409574, -2.55298982, 0.6536186]

        np.testing.assert_almost_equal(ref, data['x1'].values[:20], 6)

    @pytest.mark.unit
    def test_get_predictors_gets_reproducible_holdout(self):
        '''The target name and options are stored with the project.  We
        use the project information to format the dataframe when requested
        by the worker for use in
        '''
        reader = self.make_patched_reader(self.metadata_fixture)
        data = reader.get_predictors('holdout',
                                     self.project_fixture)

        ref = [-0.97727788, 0.44386323, 0.37816252, 1.9507754, -0.89546656,
               -0.51080514, -0.02818223, 0.3024719, -1.7262826, 0.17742614,
               -0.40178094, 0.0519454, 0.12898291, 1.13940068, -1.23482582,
               -0.68481009, 1.05445173, -0.40317695, 1.22244507, 0.3563664]

        np.testing.assert_almost_equal(ref, data['x1'].values[:20], 6)

    @pytest.mark.unit
    def test_get_predictors_gets_reproducible_training(self):
        '''The target name and options are stored with the project.  We
        use the project information to format the dataframe when requested
        by the worker for use in
        '''
        reader = self.make_patched_reader(self.metadata_fixture)
        data = reader.get_predictors('training',
                                     self.project_fixture)

        ref = [1.76405235, 0.40015721, 0.97873798, 2.2408932, 1.86755799,
               0.95008842, -0.15135721, -0.10321885, 0.4105985, 0.14404357,
               1.45427351, 0.76103773, 0.12167502, 0.33367433, 1.49407907,
               -0.20515826, 0.3130677, -0.85409574, -2.55298982, 0.6536186]

        np.testing.assert_almost_equal(ref, data['x1'].values[:20], 6)

    @pytest.mark.unit
    def test_get_predictors_gets_reproducible_all(self):
        '''The target name and options are stored with the project.  We
        use the project information to format the dataframe when requested
        by the worker for use in
        '''
        reader = self.make_patched_reader(self.metadata_fixture)
        data = reader.get_predictors('all',
                                     self.project_fixture)

        ref = [1.76405235, 0.40015721, 0.97873798, 2.2408932, 1.86755799,
               -0.97727788, 0.95008842, -0.15135721, -0.10321885, 0.4105985,
               0.14404357, 1.45427351, 0.76103773, 0.12167502, 0.44386323,
               0.33367433, 1.49407907, -0.20515826, 0.3130677, -0.85409574]
        np.testing.assert_almost_equal(ref, data['x1'].values[:20], 6)

    @pytest.mark.unit
    def test_unknown_storage_type_raises_error(self):
        meta = copy.deepcopy(self.metadata_fixture)
        meta['controls']['de47acd5-4593-4d25-a750-e6b3966f4fa4']['type'] = \
            'foobar'
        with self.assertRaises(ValueError):
            reader = DatasetReader.from_record(meta)


class TestSubsetFunction(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(12)
        self.data = pd.DataFrame(
            rng.randn(200, 12),
            columns=[chr(ord('a') + i) for i in range(12)])

    def test_subset(self):
        dsT = pq.subset_data(self.data,
                             {'holdout_pct':20},
                             partition='training')
        self.assertEqual(dsT.shape, (160, 12))

    def test_subset_holdout(self):
        dsH = pq.subset_data(self.data,
                             {'holdout_pct':20},
                             partition='holdout')
        self.assertEqual(dsH.shape, (40, 12))

    def test_subset_holdout_and_training_disjoint(self):
        dsH = pq.subset_data(self.data,
                             {'holdout_pct':20},
                             partition='holdout')
        dsT = pq.subset_data(self.data,
                             {'holdout_pct':20},
                             partition='training')
        self.assertEqual(set(dsT.index) & set(dsH.index), set([]))

    def test_subset_holdout_and_training_cover_full_index(self):
        dsH = pq.subset_data(self.data,
                             {'holdout_pct':20},
                             partition='holdout')
        dsT = pq.subset_data(self.data,
                             {'holdout_pct':20},
                             partition='training')
        self.assertEqual(set(dsT.index) | set(dsH.index),
                         set(self.data.index))


class TestTypeConversion(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(12)
        n_samples = 100
        pcts = ['{}%'.format(i) for i in 100 * rng.rand(n_samples)]

        datefmt = '%Y-%m-%d'
        dates = [(datetime.date(2014, 5, 20) -
                  datetime.timedelta(i)).strftime(datefmt)
                 for i in range(n_samples)]

        feet = rng.randint(7, size=n_samples)
        inches = rng.randint(12, size=n_samples)
        lengths = ['{}\'{}"'.format(foot, inch)
                   for (foot, inch) in zip(feet, inches)]

        currencies = ['${}'.format(i)
                      for i in np.round(100*rng.rand(n_samples), 2)]

        self.dataframe = pd.DataFrame({
            'date': dates,
            'percent': pcts,
            'length': lengths,
            'currency': currencies
        })

    def test_convert_many_at_once(self):
        '''I am assuming (maybe a bad move) that there are unit tests
        in place for the individual type converters.  Here we'll just
        make sure that we applying type conversion to several columns
        at once behaves as we expect
        '''
        typeConvert = {'date': '%',
                       'percent': 'P',
                       'length': 'L',
                       'currency': '$',
                       'na': 'NA'}

        data = pq.convert_column_types(self.dataframe,
                                       typeConvert)
        self.assertEqual(data['date'].dtype, 'int64')
        self.assertEqual(data['percent'].dtype, 'float64')
        self.assertEqual(data['length'].dtype, 'float64')
        self.assertEqual(data['currency'].dtype, 'float64')


class TestRemapColumns(unittest.TestCase):

    def setUp(self):
        self.universe_column_spec = [
            [0, 'target', 0],
            [1, 'age', 0],
            [2, 'irrelephant', 0]]

    def test_map_columns_when_layout_is_same(self):
        featurelist_column_spec = [[0, 'target', 0], [1, 'age', 0]]

        dataframe = pd.DataFrame([[0, 0]], columns=['x1', 'age'])
        columns = pq.remap_columns(
            dataframe, featurelist_column_spec, self.universe_column_spec,
            target_name='target')

        self.assertEqual(columns, [[1, 'age', 0]])

    def test_map_columns_when_layout_is_different(self):
        featurelist_column_spec = [[0, 'target', 0], [1, 'age', 0]]

        dataframe = pd.DataFrame([[0, 0]], columns=['age', 'x1'])
        columns = pq.remap_columns(
            dataframe, featurelist_column_spec, self.universe_column_spec,
            target_name='target')

        self.assertEqual(columns, [[0, 'age', 0]])

    def test_map_columns_when_target_is_present(self):
        featurelist_column_spec = [[0, 'target', 0], [1, 'age', 0]]

        dataframe = pd.DataFrame([[0, 0, 0]], columns=['x1', 'target', 'age'])
        columns = pq.remap_columns(
            dataframe, featurelist_column_spec, self.universe_column_spec,
            target_name='target')
        self.assertEqual(columns, [[2, 'age', 0]])

    def test_doesnot_affect_inputs(self):
        featurelist_column_spec = [[0, 'target', 0], [1, 'age', 0]]

        dataframe = pd.DataFrame([[0, 0, 0]], columns=['x1', 'target', 'age'])
        columns = pq.remap_columns(
            dataframe, featurelist_column_spec, self.universe_column_spec,
            target_name='target')

        self.assertEqual(featurelist_column_spec,
                         [[0, 'target', 0], [1, 'age', 0]])
        self.assertEqual(self.universe_column_spec, [
            [0, 'target', 0],
            [1, 'age', 0],
            [2, 'irrelephant', 0]])


class TestSanitizeFeatures(unittest.TestCase):

    def test_hyphen_sanitized(self):
        name = 'haber-er'
        sanitized = io.sanitized_feature_names([name])
        self.assertNotIn('-', sanitized[0])

    def test_period_sanitized(self):
        name = 'bomb.com'
        sanitized = io.sanitized_feature_names([name])
        self.assertNotIn('.', sanitized[0])

    def test_dollar_sanitized(self):
        name = 'cash$money'
        sanitized = io.sanitized_feature_names([name])
        self.assertNotIn('$', sanitized[0])

    def test_extra_space_removed(self):
        name = ' spacious '
        sanitized = io.sanitized_feature_names([name])
        self.assertNotIn(' ', sanitized[0])

    def test_internal_space_okay(self):
        name = ' space cowboy '
        sanitized = io.sanitized_feature_names([name])
        self.assertEqual(sanitized[0], 'space cowboy')

    def test_name_strip_collisions_avoided(self):
        names = ['space ', 'space  ']
        sanitized = io.sanitized_feature_names(names)
        self.assertEqual(len(set(sanitized)), 2)
        self.assertEqual(sanitized[0], 'space')
        self.assertEqual(sanitized[1], 'space_1')

    def test_rename_collisions_avoided(self):
        names = ['name$', 'name_', 'name.']
        sanitized = io.sanitized_feature_names(names)
        self.assertEqual(len(set(sanitized)), 3)
        self.assertEqual(sanitized[0], 'name_1')
        self.assertEqual(sanitized[1], 'name_')
        self.assertEqual(sanitized[2], 'name_2')

    def test_empty_string_avoided(self):
        names = ['  ']
        sanitized = io.sanitized_feature_names(names)
        self.assertEqual(sanitized[0], '_blank')

    def test_multiple_empty_strings_not_ugly(self):
        names = ['  ', '   ', '    ']
        sanitized = io.sanitized_feature_names(names)
        self.assertEqual(len(set(sanitized)), 3)
        self.assertEqual(sanitized[0], '_blank')
        self.assertEqual(sanitized[1], '_blank_1')
        self.assertEqual(sanitized[2], '_blank_2')

    def test_sanitize_name_is_idempotent(self):
        n = ' $grand.status_ '
        sanitized_many_times = io.sanitize_name(io.sanitize_name(io.sanitize_name(n)))
        sanitized_once = io.sanitize_name(n)
        self.assertEqual(sanitized_once, sanitized_many_times)

class TestApplyColumns(unittest.TestCase):
    def test_ignores_weights(self):
        rng = np.random.RandomState(1)
        df = pd.DataFrame(rng.randn(10, 2), columns=['a', 'b'])

        bad_columns = [[0, 'a', 0],
                       [1, 'b', 1234],
                       [2, 'w', 0]]

        conv = pq.apply_columns(df, bad_columns)
        self.assertTrue(np.all(conv.b.isnull()))

    def test_uncaught_exception(self):
        rng = np.random.RandomState(1)
        df = pd.DataFrame(rng.randn(10, 2), columns=['a', 'b'])

        bad_columns = [[0, 'a', 0],
                       [1, 'b', 1234]]

        conv = pq.apply_columns(df, bad_columns)
        self.assertTrue(np.all(conv.b.isnull()))

    def test_transform_exception_on_dtype_obj(self):
        rng = np.random.RandomState(1)
        s1 = pd.Series(['cat', 'dog'])
        df = pd.DataFrame([s1], columns=['a'])

        bad_columns = [[0, 'a', 10]]

        conv = pq.apply_columns(df, bad_columns)
        self.assertTrue(np.all(conv.a.isnull()))

    def test_transform_exception_on_negative_val(self):
        rng = np.random.RandomState(1)
        df = pd.DataFrame(rng.randn(10, 2), columns=['a', 'b'])

        s1 = pd.Series([-1, 2])
        df = pd.DataFrame([s1], columns=['a'])

        bad_columns = [[0, 'a', 10]]

        conv = pq.apply_columns(df, bad_columns)
        self.assertTrue(np.all(conv.a.isnull()))
