from bson import ObjectId
import datetime
import pandas as pd
import numpy as np

import copy
import unittest
import pytest
import math
from mock import patch

from tests.IntegrationTests.storage_test_base import StorageTestBase

from tests.common.test_data_reader import BaseReaderTestSetup

import common.io.dataset_reader as dataset_reader
from common.io.dataset_reader import DatasetReader
from common.io.csv_reader import CSVReader
from ModelingMachine.engine.dstransform import TransformationEnum


class TestCSVReaderConstructors(unittest.TestCase):
    '''Tests the way that CSV reader builds dataframes from the information
    stored in the database
    '''
    sample_record = {
        'files': ['a_filename_somewhere'],
        'name': 'Special Features',
        'typeConvert': {
            'feat1': True,
            'feat2': ''},
        'created': datetime.datetime(2014, 5, 10, 1, 2, 3),
        'pid': ObjectId('5223feedabadbeadbeef1234'),
        'controls': {
            'a_filename_somewhere': {'encoding': 'ASCII',
                                     'type': 'csv'},
            'dirty': False},
        'varTypeString': 'NC',
        'shape': [2, 200],
        'originalName': 'The Real OG',
        '_id': ObjectId('5223deadbeefbeadbade1234'),
        'columns': [
            [0, 'feat1', 0],
            [1, 'feat2', 0]],
    }

    def test_can_construct_from_valid_record(self):
        reader = CSVReader.from_dict(self.sample_record)
        self.assertEqual(reader.name, 'Special Features')

    def test_constructor_with_missing_data_should_error(self):
        with self.assertRaises(KeyError):
            bad_dict = copy.deepcopy(self.sample_record)
            del bad_dict['files']
            reader = CSVReader.from_dict(bad_dict)

KICKCARS_METADATA_FIXTURE = {
    u'_id': ObjectId('5373b2ac8bd88f66af0902bc'),
    u'columns': [[0, u'RefId', 0],
                 [1, u'IsBadBuy', 0],
                 [2, u'PurchDate', 0],
                 [3, u'Auction', 0],
                 [4, u'VehYear', 0],
                 [5, u'VehicleAge', 0],
                 [6, u'Make', 0],
                 [7, u'Model', 0],
                 [8, u'Trim', 0],
                 [9, u'SubModel', 0],
                 [10, u'Color', 0],
                 [11, u'Transmission', 0],
                 [12, u'WheelTypeID', 0],
                 [13, u'WheelType', 0],
                 [14, u'VehOdo', 0],
                 [15, u'Nationality', 0],
                 [16, u'Size', 0],
                 [17, u'TopThreeAmericanName', 0],
                 [18, u'MMRAcquisitionAuctionAveragePrice', 0],
                 [19, u'MMRAcquisitionAuctionCleanPrice', 0],
                 [20, u'MMRAcquisitionRetailAveragePrice', 0],
                 [21, u'MMRAcquisitonRetailCleanPrice', 0],
                 [22, u'MMRCurrentAuctionAveragePrice', 0],
                 [23, u'MMRCurrentAuctionCleanPrice', 0],
                 [24, u'MMRCurrentRetailAveragePrice', 0],
                 [25, u'MMRCurrentRetailCleanPrice', 0],
                 [26, u'PRIMEUNIT', 0],
                 [27, u'AUCGUART', 0],
                 [28, u'BYRNO', 0],
                 [29, u'VNZIP1', 0],
                 [30, u'VNST', 0],
                 [31, u'VehBCost', 0],
                 [32, u'IsOnlineSale', 0],
                 [33, u'WarrantyCost', 0]],
    u'controls': {
        u'projects/5373b2a98bd88f655d884aee/raw/kickcars-sample-200.csv': {
            u'encoding': u'ASCII',
            u'type': u'csv'},
        u'dirty': False},
    u'created': datetime.datetime(2014, 5, 14, 18, 15, 8, 161000),
    u'files': [u'projects/5373b2a98bd88f655d884aee/raw/kickcars-sample-200.csv'],
    u'name': u'Raw Features',
    u'originalName': u'Raw Features',
    u'pid': ObjectId('5373b2a98bd88f655d884aee'),
    u'shape': [200, 34],
    u'typeConvert': {u'AUCGUART': u'',
                     u'Auction': u'',
                     u'BYRNO': True,
                     u'Color': u'',
                     u'IsBadBuy': True,
                     u'IsOnlineSale': True,
                     u'MMRAcquisitionAuctionAveragePrice': True,
                     u'MMRAcquisitionAuctionCleanPrice': True,
                     u'MMRAcquisitionRetailAveragePrice': True,
                     u'MMRAcquisitonRetailCleanPrice': True,
                     u'MMRCurrentAuctionAveragePrice': True,
                     u'MMRCurrentAuctionCleanPrice': True,
                     u'MMRCurrentRetailAveragePrice': True,
                     u'MMRCurrentRetailCleanPrice': True,
                     u'Make': u'',
                     u'Model': u'',
                     u'Nationality': u'',
                     u'PRIMEUNIT': u'',
                     u'PurchDate': u'%m/%d/%Y',
                     u'RefId': True,
                     u'Size': u'',
                     u'SubModel': u'',
                     u'TopThreeAmericanName': u'',
                     u'Transmission': u'',
                     u'Trim': u'',
                     u'VNST': u'',
                     u'VNZIP1': True,
                     u'VehBCost': True,
                     u'VehOdo': True,
                     u'VehYear': True,
                     u'VehicleAge': True,
                     u'WarrantyCost': True,
                     u'WheelType': u'',
                     u'WheelTypeID': True},
    u'varTypeString': u'NNNCNNCCCCCCNCNCCCNNNNNNNNCCNNCNNN'}


@pytest.mark.integration
class TestCSVReaderDiskAccessMethods(StorageTestBase):
    '''These tests are understandably slow because they hit the disk.
    '''
    def setUp(self):
        self.metadata = copy.deepcopy(KICKCARS_METADATA_FIXTURE)

    def test_load_csv_dataframe_smoketest(self):
        # files are put into storage with a prefix of the project that
        # created them
        # storage test base creates files with a specific project ID
        metadata = copy.deepcopy(self.metadata)

        self.create_test_files(['kickcars-sample-200.csv', ])
        reader = CSVReader.from_dict(metadata)
        dataframe = reader.get_raw_dataframe()

        for column in metadata['columns']:
            column_name = column[1]  # TODO - Get a model around this
            self.assertIn(column_name, dataframe.columns)

    def test_load_csv_dataframe_via_datareader(self):
        metadata = copy.deepcopy(self.metadata)

        self.create_test_files(['kickcars-sample-200.csv', ])
        reader = DatasetReader.from_record(metadata)
        dataframe = reader.get_raw_dataframe()
        for column in metadata['columns']:
            column_name = column[1]  # TODO - Get a model around this
            self.assertIn(column_name, dataframe.columns)

    @pytest.mark.skip
    def test_getting_data_uses_safe_columns_names(self):
        pass

    @pytest.mark.skip
    def test_can_load_dataset_based_on_more_than_one_file(self):
        pass


class TestCSVReaderInterfaceMethods(BaseReaderTestSetup):
    '''These tests are for what happens in between the raw dataset read
    and the returned values.  Our strategy for CSV files is to just
    start with the whole raw dataset and then pare down, so we can set up
    this test by just dropping in a fake dataframe
    '''
    def make_patched_reader(self, metadata):
        disk_reader = CSVReader.from_dict(metadata)
        disk_reader._raw_dataframe = self._raw_dataframe
        return disk_reader

    def test_can_provide_data_with_subsetting_columns_by_column_name(self):
        '''This is how the DataProcessor will get its predictors - it knows
        which is the target column, so it can ask for all of them _except_
        that column
        '''
        reader = self.make_patched_reader(self.metadata_fixture)
        sought_columns = ['x1', 'x2', 'x3']
        dataframe = reader.get_raw_dataframe(columns=sought_columns)
        for column in sought_columns:
            self.assertIn(column, dataframe.columns)
        not_included = ['y']
        for column in not_included:
            self.assertNotIn(column, dataframe.columns)

    def test_automatically_applies_stored_column_operations(self):
        metadata = copy.deepcopy(self.metadata_fixture)
        metadata['columns'][0][2] = TransformationEnum.square
        reader = self.make_patched_reader(metadata)
        dataframe = reader.get_formatted_dataframe()
        ref = self._raw_dataframe['x1']
        check = dataframe['x1']
        np.testing.assert_almost_equal(ref ** 2, check)


class TestCSVReaderTypeConversion(BaseReaderTestSetup):

    def setUp(self):
        rng = np.random.RandomState(123)
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

        self._raw_dataframe = pd.DataFrame({
            'x1': dates,
            'x2': pcts,
            'x3': lengths,
            'y': currencies
        })

    def make_patched_reader(self, metadata):
        disk_reader = CSVReader.from_dict(metadata)
        disk_reader._raw_dataframe = self._raw_dataframe
        return disk_reader

    def test_automatically_applies_typeconvert(self):
        metadata = copy.deepcopy(self.metadata_fixture)
        typeConvert = {'x1': '%',
                       'x2': 'P',
                       'x3': 'L',
                       'y': '$',
                       'na': 'NA'}  # An extra typeConvert is no problem
        metadata['typeConvert'] = typeConvert
        reader = self.make_patched_reader(metadata)

        #Act
        dataframe = reader.get_formatted_dataframe()

        #Assert
        self.assertEqual(dataframe['y'][0], 17.23)
        self.assertEqual(dataframe['x1'][0], 735373)
        self.assertAlmostEqual(dataframe['x2'][0], 69.6469185, 5)
        self.assertEqual(dataframe['x3'][0], 70)


class TestCSVReaderGetTargetSeries(BaseReaderTestSetup):
    '''These tests ensure that the target series is convert according to its
    typeConvert spec (stored in the metadata), and also lets us make
    assertions on how we expect some data formats to be handled
    '''
    def make_patched_reader(self, metadata):
        disk_reader = CSVReader.from_dict(metadata)
        disk_reader._raw_dataframe = self._raw_dataframe
        return DatasetReader(disk_reader)

    def test_target_needs_no_conversion(self):
        metadata = copy.deepcopy(self.metadata_fixture)
        typeConvert = {'x1': '%',
                       'x2': 'P',
                       'x3': 'L',
                       'y': True,
                       'na': 'NA'}  # An extra typeConvert is no problem
        metadata['typeConvert'] = typeConvert
        reader = self.make_patched_reader(metadata)
        rng = np.random.RandomState(0)
        yseries = pd.Series(0.01 * rng.randint(low=500, high=10000, size=100))
        reader.reader._raw_dataframe['y'] = yseries

        #Act
        dataframe = reader.get_target_series('all', self.project_fixture)

        #Assert
        self.assertTrue(np.issubdtype(dataframe.dtype, np.number))

    def test_target_has_typeconversion_applied(self):
        metadata = copy.deepcopy(self.metadata_fixture)
        typeConvert = {'x1': '%',
                       'x2': 'P',
                       'x3': 'L',
                       'y': '$',
                       'na': 'NA'}  # An extra typeConvert is no problem
        metadata['typeConvert'] = typeConvert
        reader = self.make_patched_reader(metadata)
        rng = np.random.RandomState(0)
        yseries = pd.Series(0.01 * rng.randint(low=500, high=10000, size=100))
        reader.reader._raw_dataframe['y'] = yseries.map(
            lambda x: '${}'.format(x))

        #Act
        dataframe = reader.get_target_series('all', self.project_fixture)

        #Assert
        self.assertTrue(np.issubdtype(dataframe.dtype, np.number))

    def test_target_with_dollars_and_commas(self):
        metadata = copy.deepcopy(self.metadata_fixture)
        typeConvert = {'x1': '%',
                       'x2': 'P',
                       'x3': 'L',
                       'y': '$',
                       'na': 'NA'}  # An extra typeConvert is no problem
        metadata['typeConvert'] = typeConvert
        reader = self.make_patched_reader(metadata)
        rng = np.random.RandomState(0)
        yseries = pd.Series(0.01 * rng.randint(low=500, high=1e8, size=100))
        reader.reader._raw_dataframe['y'] = yseries.map(
            lambda x: '${:20,.2f}'.format(x))

        #Act
        dataframe = reader.get_target_series('all', self.project_fixture)

        #Assert
        self.assertTrue(np.issubdtype(dataframe.dtype, np.number))

    def test_target_with_dollars_and_commas_and_negatives(self):
        import locale
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        metadata = copy.deepcopy(self.metadata_fixture)
        typeConvert = {'x1': '%',
                       'x2': 'P',
                       'x3': 'L',
                       'y': '$',
                       'na': 'NA'}  # An extra typeConvert is no problem
        metadata['typeConvert'] = typeConvert
        reader = self.make_patched_reader(metadata)
        rng = np.random.RandomState(0)
        yseries = pd.Series(0.01 * rng.randint(low=-1e8, high=1e8, size=100))
        formatted_yseries = yseries.map(
            lambda x: locale.currency(x, grouping=True))
        reader.reader._raw_dataframe['y'] = formatted_yseries

        #Act
        dataframe = reader.get_target_series('all', self.project_fixture)

        #Assert
        self.assertTrue(np.issubdtype(dataframe.dtype, np.number))
