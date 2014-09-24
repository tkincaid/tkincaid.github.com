from test_base import TestBase
import unittest
from MMApp.entities.dataset import DatasetService, UploadedFile, RequestError
from common.entities.dataset import METADATA_TABLE_NAME
from MMApp.entities.featurelist import FeaturelistService
from mock_classes import MockFileStorage
from config.app_config import config
from pymongo.errors import InvalidId
from bson.objectid import ObjectId
import common.services.eda
import common.wrappers.dbs.mongo_db
import common.exceptions as exceptions
import redis
import pymongo
import json
import urllib2
import os
from mock import Mock, patch, DEFAULT, MagicMock, mock_open
import pytest

@pytest.mark.unit
class TestDatasetService(TestBase):

    EDA_MAP = {
        "_id" : ObjectId("533cc9182b05aba2e73ee2c0"),
        "dataset_id" : "universe",
        "pid" : ObjectId("533cc918637aba59b8dd453a"),
        "block_contents": {
            "533cc9182b05aba2e73ee2c0": [
                "WarrantyCost",
                "PRIMEUNIT",
            ]
        }
    }
    EDA = {
        "_id" : ObjectId("533cc9182b05aba2e73ee2c0"),
        "dataset_id" : "universe",
        "eda" : {
            "WarrantyCost" : {
                "profile" : {
                    "info" : 0.11250000000000004,
                    "plot" : [
                        [
                            462,
                            3,
                            0
                        ],
                        [
                            496.3251724137931,
                            3,
                            0
                        ]
                    ],
                    "name" : "WarrantyCost",
                    "miss_count" : 0,
                    "raw_info" : 0.55625,
                    "y" : "IsBadBuy",
                    "plot2" : [
                        [
                            "1003.0",
                            2,
                            1
                        ],
                        [
                            "1020.0",
                            7,
                            2
                        ]
                    ],
                    "type" : "N",
                    "miss_ymean" : 0
                },
                "name" : "WarrantyCost",
                "transform_args" : [ ],
                "low_info" : {
                    "high_cardinality" : False,
                    "duplicate" : False,
                    "empty" : False,
                    "few_values" : False
                },
                "metric_options" : {
                    "all" : [
                        {
                            "short_name" : "Gini Norm"
                        },
                        {
                            "short_name" : "Gamma Deviance"
                        }
                    ],
                    "recommended" : {
                        "default" : { "short_name" : "RMSLE" },
                        "recommender" : { "short_name" : "Gini Norm" },
                        "weighted" : { "short_name" : "Weighted RMSLE" },
                        "weight+rec" : { "short_name" : "Weighted Gini Norm" },
                    }
                },
                "summary" : [
                    88,
                    0,
                    1263.53,
                    694.0322636701231,
                    462,
                    1131,
                    6519
                ],
                "raw_variable_index" : 33,
                "transform_id" : 0,
                "id" : 33,
                "types" : {
                    "text" : False,
                    "currency" : False,
                    "length" : False,
                    "date" : False,
                    "percentage" : False,
                    "nastring" : True
                }
            },
            "PRIMEUNIT" : {
                "profile" : {
                    "info" : 0.02632575757575739,
                    "plot" : [
                        [
                            "==Missing==",
                            184,
                            60
                        ],
                        [
                            "NO",
                            16,
                            1
                        ]
                    ],
                    "name" : "PRIMEUNIT",
                    "raw_info" : 0.5131628787878787,
                    "y" : "IsBadBuy",
                    "type" : "C"
                },
                "name" : "PRIMEUNIT",
                "transform_args" : [ ],
                "low_info" : {
                    "high_cardinality" : False,
                    "duplicate" : False,
                    "empty" : False,
                    "few_values" : False
                },
                "metric_options" : {
                    "all" : [
                        { "short_name" : "AUC" },
                        { "short_name" : "LogLoss" }
                    ],
                    "recommended" : {
                        "default" : { "short_name" : "LogLoss" },
                        "recommender" : { "short_name" : "AUC" },
                        "weighted" : { "short_name" : "Weighted LogLoss" },
                        "weight+rec" : { "short_name" : "Weighted AUC" },
                    }
                },
                "summary" : [
                    1,
                    184
                ],
                "raw_variable_index" : 26,
                "transform_id" : 0,
                "id" : 26,
                "types" : {
                    "text" : False,
                    "currency" : False,
                    "length" : False,
                    "date" : False,
                    "percentage" : False,
                    "nastring" : False
                }
            }
        },
        "pid" : ObjectId("533cc918637aba59b8dd453a")
    }

    METADATA =  [{
            "_id" : ObjectId("533cc8c5637aba538add4539"),
            "columns" : [
                [0,"Unnamed: 0",0],
                [1,"SeriousDlqin2yrs",0],
                [2,"RevolvingUtilizationOfUnsecuredLines",0],
                [3,"age",0],
                [4,"NumberOfTime30_59DaysPastDueNotWorse",0],
                [5,"DebtRatio",0],
                [6,"MonthlyIncome",0],
                [7,"NumberOfOpenCreditLinesAndLoans",0],
                [8,"NumberOfTimes90DaysLate",0],
                [9,"NumberRealEstateLoansOrLines",0],
                [10,"NumberOfTime60_89DaysPastDueNotWorse",0],
                [11,"NumberOfDependents",0]
            ],
            "controls" : {
                "2314c052-b6eb-482f-b9f6-49fa2b379094" : {
                    "type" : "csv",
                    "encoding" : "ASCII"
                },
                "dirty" : False
            },
            "created" : "2014-04-03T02:34:45.699Z",
            "files" : [
                "2314c052-b6eb-482f-b9f6-49fa2b379094"
            ],
            "name" : "universe",
            "originalName" : "universe",
            "pid" : ObjectId("533cc8c5637aba538add453a"),
            "shape" : [
                200,
                12
            ],
            "typeConvert" : {
                "NumberOfTime30_59DaysPastDueNotWorse" : False,
                "NumberOfOpenCreditLinesAndLoans" : False,
                "Unnamed: 0" : False,
                "age" : False,
                "DebtRatio" : False,
                "NumberOfDependents" : False,
                "MonthlyIncome" : False,
                "SeriousDlqin2yrs" : False,
                "RevolvingUtilizationOfUnsecuredLines" : False,
                "NumberRealEstateLoansOrLines" : False,
                "NumberOfTime60_89DaysPastDueNotWorse" : False,
                "NumberOfTimes90DaysLate" : False
            },
            "varTypeString" : "NNNNNNNNNNNN"
        }]


    def setUp(self):
        self.patchers = []
        self.addCleanup(self.stopPatching)
        self.service = DatasetService(tempstore=Mock(), persistent=Mock())
        self.fl_service = FeaturelistService(persistent=Mock())

        mock_requests = patch('MMApp.entities.dataset.requests')
        self.MockRequests = mock_requests.start()
        self.patchers.append(mock_requests)

    def stopPatching(self):
        super(TestDatasetService, self).tearDown()
        for patcher in self.patchers:
            if patcher:
                patcher.stop()

    def tearDown(self):
        self.redis_conn.flushall()

    def test_validate_url(self):
        #No url no file
        self.assertRaises(TypeError,DatasetService.validate_URL, None)

        valid_urls = [
            #TODO: localhosst Should be invalid in production
            'http://localhost/hello.txt',
            'http://146.115.8.91/hello.txt',
            'https://146.115.8.91/path/to/file/hello.csv',
            'https://s3.amazonaws.com/some-folder/58439/400216/yweafrvqsuep685/fome-file.csv',
            'https://docs.google.com/file/d/0B15dR6gFNczBampRelc5NU92MG8/edit?usp=sharing',
            'https://usr:psw@s3.amazonaws.com/testlio/project/260/test-data/normal-size.csv',
            'https://usr@s3.amazonaws.com:8080'
        ]

        invalid_urls = [
            'does-not-exist',
            'http://domain',
            'ftp://something.com/yes/files',
            'http://bad-domain',
            'http:bad-domain',
            'https://usr:@s3.amazonaws.com',
            'https://:pws@s3.amazonaws.com'
            'https://:@s3.amazonaws.com'
        ]

        for url in valid_urls:
            self.assertTrue(DatasetService.validate_URL(url))

        for url in invalid_urls:
            self.assertRaisesRegexp(Exception, 'not a valid url', DatasetService.validate_URL, url)

    def test_process_url_upload(self):
        pid = ObjectId()
        uid = ObjectId()
        service = DatasetService(pid, uid, tempstore=Mock(), persistent=Mock())

        self.MockRequests.get.return_value.status_code = 200

        url = 'https://www.census.gov/econ/cbp/download/noise_layout/County_Layout.txt'
        uploaded_file = service.process_url_upload(url)

        #TODO: Url, no file file_size
        self.assertIsNotNone(uploaded_file.original_filename)
        self.assertIsNotNone(uploaded_file.local_path)
        self.assertIsNotNone(uploaded_file.dataset_id)
        self.assertIsNotNone(uploaded_file.unique_id)

    def test_process_url_upload_with_args_and_no_extension(self):
        pid = ObjectId()
        uid = ObjectId()
        service = DatasetService(pid, uid, tempstore=Mock(), persistent=Mock())

        self.MockRequests.get.return_value.status_code = 200

        url = 'https://does-not-exist.com/my-file.csv?arg1=123&arg2=abc'
        result = service.process_url_upload(url)
        self.assertIsNotNone(result)

        url = 'https://does-not-exist.com/no-extension/'
        result = service.process_url_upload(url)
        self.assertIsNotNone(result)

        url = 'https://does-not-exist.com/no-extension/?args=123'
        result = service.process_url_upload(url)
        self.assertIsNotNone(result)

    def test_fetch_url_with_huge_file(self):
        def infinite(chunk_size):
            while True:
                yield 'hello world'

        url = 'https://www.fake.com/does-not-exist.txt'
        fake_open = mock_open()
        with patch('MMApp.entities.dataset.open', fake_open, create=True):
            mock_request = self.MockRequests.get.return_value
            mock_request.iter_content.side_effect = infinite

            with patch.object(self.service, 'validate_file_size') as mock_validate_file_size:
                mock_validate_file_size.side_effect = Exception('BOOM!')
                self.assertRaises(Exception, self.service.fetch_url, url, 'file-name')

    def test_fetch_url_with_no_size(self):
        pid = ObjectId()
        uid = ObjectId()
        service = DatasetService(pid, uid, tempstore=Mock(), persistent=Mock())

        url = 'https://www.fake.com/does-not-exist.txt'

        mock_url_info = MagicMock()
        self.MockRequests.get.return_value.headers = mock_url_info
        self.MockRequests.get.return_value.status_code = 200
        mock_url_info.__getitem__.return_value =  config['MAX_CONTENT_LENGTH'] + 1
        mock_url_info.keys.return_value = ['content-length']
        self.assertRaisesRegexp(Exception, 'is greater than the max allowed', service.fetch_url, url, 'file-name')

    def test_fetch_url_with_not_ok_status(self):
        pid = ObjectId()
        uid = ObjectId()
        service = DatasetService(pid, uid, tempstore=Mock(), persistent=Mock())

        url = 'https://www.fake.com/does-not-exist.txt'

        self.MockRequests.get.return_value.status_code = 400
        self.assertRaisesRegexp(Exception, 'Request was unsuccessful. HTTP status code: 400', service.fetch_url, url, 'file-name')

    def test_process_file_upload(self):
        url = None
        new_filename = 'does-not-exist.txt'
        up_file = MockFileStorage(new_filename)
        pid = ObjectId()
        uid = ObjectId()
        service = DatasetService(pid, uid, tempstore=Mock(), persistent=Mock())
        uploaded_file = service.process_file_upload(up_file)
        self.assertIsNotNone(uploaded_file.original_filename)
        self.assertIsNotNone(uploaded_file.local_path)
        self.assertIsNotNone(uploaded_file.dataset_id)
        self.assertIsNotNone(uploaded_file.unique_id)

        # Exception, file not uploaded
        self.assertRaises(TypeError, self.service.process_file_upload, None)

    def test_validate_file_size(self):
        #valid file
        filename = 'x'

        with patch('MMApp.entities.dataset.os') as mock_os:
            #Mocks st_size and st_mtime of os.stat return value
            mock_stat = Mock()
            mock_stat.st_size = 1
            mock_stat.st_mtime = 0
            mock_os.stat.return_value = mock_stat

            #Act
            self.service.validate_file_size(filename)
            # Mainly testing if this errors or not

            #Assert
            #Don't try to remove a valid file
            self.assertFalse(mock_os.remove.called)

    @patch.dict('MMApp.entities.dataset.app_config', {'MAX_CONTENT_LENGTH' : 100*1024*1024}, clear=True)
    def test_validate_file_size_large(self):
        #valid file
        filename = 'x'

        with patch('MMApp.entities.dataset.os') as mock_os:
            mock_stat = Mock()
            mock_stat.st_size = 100*1024*1024
            mock_stat.st_mtime = 0
            mock_os.stat.return_value = mock_stat

            #Act
            self.service.validate_file_size(filename)
            # Mainly testing if this errors or not

            #Assert
            #Don't try to remove a valid file
            self.assertFalse(mock_os.remove.called)

    def test_validate_file_size_too_large(self):
        #valid file
        filename = 'x'

        with patch('MMApp.entities.dataset.os') as mock_os:
            mock_stat = Mock()
            mock_stat.st_size = config['MAX_CONTENT_LENGTH'] + 1
            mock_stat.st_mtime = 0
            mock_os.stat.return_value = mock_stat

            self.assertRaisesRegexp(Exception, 'is greater than the max allowed',  self.service.validate_file_size, filename)
            self.assertTrue(mock_os.remove.called)

    def test_create_server_filename(self):
        pid = ObjectId()
        uid = ObjectId()
        service = DatasetService(pid, uid, tempstore=Mock(), persistent=Mock())
        uploaded_file = service.create_server_filename()
        #Make sure it creates files on the right path
        self.assertTrue(uploaded_file.local_path.startswith(config['UPLOAD_FOLDER']))
        self.assertEqual(len(os.path.basename(uploaded_file.local_path)), 36)

        self.assertIsNotNone(uploaded_file.dataset_id)
        self.assertIsNotNone(uploaded_file.unique_id)

    def test_validate_filename(self):
        #Valid file name
        filename = os.path.join('tests', 'testdata', 'good_generated.txt')
        self.service.validate_filename(filename)
        #Invalid extension

    def test_validate_filename_catches_garbage_extension(self):
        filename = os.path.join('tests', 'testdata', 'good_generated.txt.xyz')
        with self.assertRaises(exceptions.InvalidFilenameError) as e:
            self.service.validate_filename(filename)

    def test_empty_filenames_are_not_cool(self):
        #Empty file name
        filename = None
        self.assertRaises(TypeError, self.service.validate_filename, filename)

    def test_weird_directory_walks_are_okay(self):
        filename = '../here/../there/../anywhere/file.csv'
        self.service.validate_filename(filename)

    def test_urls_are_not_a_problem_with_extra_dots(self):
        filename = 'http://fileserver.place.com/file.csv'
        self.service.validate_filename(filename)

    def test_save_metadata(self):
        #Invalid Id
        pid = '123'
        uploaded_file = UploadedFile(dataset_id = 'x', pid=pid, original_filename = 'x')
        self.assertRaises(ValueError, self.service.save_metadata, pid, uploaded_file)
        #Good Id

        pid = '313233343536373839303930'
        self.service.persistent.read.return_value = 'AnOID'
        self.service.persistent.count.return_value = 0
        uploaded_file = UploadedFile(dataset_id = 'unit_test_dataset_id', pid=pid, original_filename = 'unit_test_file_name')

        dataset_id = uploaded_file.dataset_id
        expected_values = {'pid':ObjectId(pid),
                           'dataset_id':dataset_id,
                           'files':[dataset_id],
                           'originalName':uploaded_file.original_filename,
                           'name': dataset_id,
                           'created':1,
                           'newdata':True}

        with patch('MMApp.entities.dataset.datetime.datetime') as mock_time:
            mock_time.utcnow.return_value = 1

            #Act
            self.service.save_metadata(pid, uploaded_file)

            #Assert
            self.service.persistent.update.assert_called_with(table=METADATA_TABLE_NAME, condition={'pid':pid, 'dataset_id':dataset_id}, values=expected_values)
            oid =  self.service.save_metadata(pid, uploaded_file)
            self.assertIsNotNone(oid)

    def test_column_packing(self):
        columns = [[0,"0",0],[1,"1",0],[2,"2",1,1],[3,"3",1,2]]
        unpacked = self.fl_service.unpack_columns(columns)
        packed = self.fl_service.pack_columns(unpacked)
        self.assertEqual(columns, packed)

    def test_get_universe_metadata_no_auth_no_return(self):
        with self.assertRaises(ValueError):
            # FIXME I think that by design this is supposed to return a
            # RequestError, but if you submit None for both `uid` and
            # `pid` it raises a ValueError
            meta = self.service.get_universe_metadata()

    def test_get_universe_metadata_authed_gets_returned(self):
        with patch.multiple(self.service, persistent = DEFAULT, assert_can_view = DEFAULT) as mocks:
            mocks['persistent'].read.return_value = self.METADATA
            metadata = self.service.get_universe_metadata()
            self.assertEqual(metadata['originalName'], 'universe')

    def test_get_universe_metadata_not_found_returns_none(self):
        with patch.multiple(self.service, persistent = DEFAULT, assert_can_view = DEFAULT) as mocks:
            mocks['persistent'].read.return_value = []
            metadata = self.service.get_universe_metadata()
            self.assertIsNone(metadata)

    def test_get_universe_feature_names(self):
        with patch.multiple(self.fl_service, persistent = DEFAULT, assert_can_view = DEFAULT) as mocks:
            mocks['persistent'].read.return_value = self.METADATA

            eda = self.fl_service.get_universe_feature_names()

            self.assertEqual(len(eda), 12)

            feature_names = [feature['name'] for feature in eda]

            self.assertIn('SeriousDlqin2yrs', feature_names)
            self.assertIn('NumberOfDependents', feature_names)

    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    @patch.object(DatasetService, 'assert_can_view')
    def test_get_universe_profile(self,amock,vmock):
        with patch.multiple(self.service, persistent = DEFAULT, assert_can_view = DEFAULT) as mocks:
            mocks['persistent'].read.return_value = self.EDA

            with patch.object(common.services.eda.EdaService, 'eda_map', self.EDA_MAP):
                with patch.object(common.wrappers.dbs.mongo_db.MongoDB,'read') as prmock:
                    prmock.return_value = self.EDA
                    eda = self.service.get_universe_profile()

            # We have names
            self.assertTrue(eda)

            feature = eda.pop()
            name = feature['name']

            # We have profile and profile info
            self.assertIn('profile', feature)
            self.assertIn('type_label', feature['profile'])

            # But... no graph data (plots)
            self.assertNotIn('plot', feature['profile'])
            self.assertNotIn('plot2', feature['profile'])

    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    def test_get_universe_graphs(self,*args):
        with patch.multiple(self.service, persistent = DEFAULT, assert_can_view = DEFAULT) as mocks:
            mocks['persistent'].read.return_value = self.EDA

            with patch.object(common.services.eda.EdaService, 'eda_map', self.EDA_MAP):
                with patch.object(common.wrappers.dbs.mongo_db.MongoDB,'read') as prmock:
                    prmock.return_value = self.EDA
                    eda = self.service.get_universe_graphs()

            # We have names
            self.assertTrue(eda)

            feature = eda.pop(0)
            name = feature['name']

            # We have profile but no other stuff
            self.assertIn('profile', feature)
            self.assertNotIn('metric_options', feature)

            # We do have graph data (plots)
            self.assertIn('plot', feature['profile'])
            self.assertIn('plot2', feature['profile'], feature)

    def test_get_all_datasets(self):
        with patch.multiple(self.fl_service, persistent = DEFAULT, assert_can_view = DEFAULT) as mocks:
            pid = ObjectId()
            dataset_list = [{
                u'files': [u'projects/53923eccf9ac63682d1784d6/raw/1c2e2ed7-55e7-4af3-a6c9-9ecadc319ea1'],
                u'name': u'universe',
                u'typeConvert': {u'NumberOfTime30_59DaysPastDueNotWorse': True, u'NumberOfOpenCreditLinesAndLoans': True, u'Unnamed: 0': True, u'age': True, u'DebtRatio': True, u'NumberOfDependents': True, u'MonthlyIncome': True, u'SeriousDlqin2yrs': True, u'RevolvingUtilizationOfUnsecuredLines': True, u'NumberRealEstateLoansOrLines': True, u'NumberOfTime60_89DaysPastDueNotWorse': True, u'NumberOfTimes90DaysLate': True},
                u'created': '2000/1/1',
                u'pid': ObjectId('53923eccf9ac63682d1784d6'),
                u'controls': {u'dirty': False, u'projects/53923eccf9ac63682d1784d6/raw/1c2e2ed7-55e7-4af3-a6c9-9ecadc319ea1': {u'type': u'csv', u'encoding': u'ASCII'}},
                u'varTypeString': u'NNNNNNNNNNNN',
                u'shape': [1000, 12],
                u'originalName': u'universe',
                u'_id': ObjectId('53923eccf9ac636954753b8f'),
                u'columns': [[0, u'Unnamed: 0', 0], [1, u'SeriousDlqin2yrs', 0], [2, u'RevolvingUtilizationOfUnsecuredLines', 0], [3, u'age', 0], [4, u'NumberOfTime30_59DaysPastDueNotWorse', 0], [5, u'DebtRatio', 0], [6, u'MonthlyIncome', 0], [7, u'NumberOfOpenCreditLinesAndLoans', 0], [8, u'NumberOfTimes90DaysLate', 0], [9, u'NumberRealEstateLoansOrLines', 0], [10, u'NumberOfTime60_89DaysPastDueNotWorse', 0], [11, u'NumberOfDependents', 0]]
            }]
            mocks['persistent'].read.return_value = dataset_list
            result = self.fl_service.get_all_datasets(pid)
            self.assertEqual(result,dataset_list)

    def test_create_dataset(self):
        with patch.multiple(self.fl_service, persistent = DEFAULT, assert_can_view = DEFAULT, assert_can_edit = DEFAULT) as mocks:
            pid = ObjectId()
            dataset_list1 = {
                u'files': [u'projects/53923eccf9ac63682d1784d6/raw/1c2e2ed7-55e7-4af3-a6c9-9ecadc319ea1'],
                u'name': u'universe',
                u'typeConvert': {u'NumberOfTime30_59DaysPastDueNotWorse': True, u'NumberOfOpenCreditLinesAndLoans': True, u'Unnamed: 0': True, u'age': True, u'DebtRatio': True, u'NumberOfDependents': True, u'MonthlyIncome': True, u'SeriousDlqin2yrs': True, u'RevolvingUtilizationOfUnsecuredLines': True, u'NumberRealEstateLoansOrLines': True, u'NumberOfTime60_89DaysPastDueNotWorse': True, u'NumberOfTimes90DaysLate': True},
                u'created': '2000/1/1',
                u'pid': ObjectId('53923eccf9ac63682d1784d6'),
                u'controls': {u'dirty': False, u'projects/53923eccf9ac63682d1784d6/raw/1c2e2ed7-55e7-4af3-a6c9-9ecadc319ea1': {u'type': u'csv', u'encoding': u'ASCII'}},
                u'varTypeString': u'NNNNNNNNNNNN',
                u'shape': [1000, 12],
                u'originalName': u'universe',
                u'_id': ObjectId('53923eccf9ac636954753b8f'),
                u'columns': [[0, u'Unnamed: 0', 0], [1, u'SeriousDlqin2yrs', 0], [2, u'RevolvingUtilizationOfUnsecuredLines', 0], [3, u'age', 0], [4, u'NumberOfTime30_59DaysPastDueNotWorse', 0], [5, u'DebtRatio', 0], [6, u'MonthlyIncome', 0], [7, u'NumberOfOpenCreditLinesAndLoans', 0], [8, u'NumberOfTimes90DaysLate', 0], [9, u'NumberRealEstateLoansOrLines', 0], [10, u'NumberOfTime60_89DaysPastDueNotWorse', 0], [11, u'NumberOfDependents', 0]]
            }
            dataset_list2 = {
                u'files': [u'projects/53923eccf9ac63682d1784d6/raw/1c2e2ed7-55e7-4af3-a6c9-9ecadc319ea1'],
                u'name': u'featurelist',
                u'typeConvert': {u'NumberOfTime30_59DaysPastDueNotWorse': True, u'NumberOfOpenCreditLinesAndLoans': True, u'Unnamed: 0': True, u'age': True, u'DebtRatio': True, u'NumberOfDependents': True, u'MonthlyIncome': True, u'SeriousDlqin2yrs': True, u'RevolvingUtilizationOfUnsecuredLines': True, u'NumberRealEstateLoansOrLines': True, u'NumberOfTime60_89DaysPastDueNotWorse': True, u'NumberOfTimes90DaysLate': True},
                u'created': '2000/1/1',
                u'pid': ObjectId('53923eccf9ac63682d1784d6'),
                u'controls': {u'dirty': False, u'projects/53923eccf9ac63682d1784d6/raw/1c2e2ed7-55e7-4af3-a6c9-9ecadc319ea1': {u'type': u'csv', u'encoding': u'ASCII'}},
                u'varTypeString': u'NNNNNNNNNNNN',
                u'shape': [1000, 12],
                u'originalName': u'universe',
                u'_id': ObjectId('53923eccf9ac636954753b8f'),
                u'columns': [[0, u'Unnamed: 0', 0], [1, u'SeriousDlqin2yrs', 0], [2, u'RevolvingUtilizationOfUnsecuredLines', 0], [3, u'age', 0], [4, u'NumberOfTime30_59DaysPastDueNotWorse', 0], [5, u'DebtRatio', 0], [6, u'MonthlyIncome', 0], [7, u'NumberOfOpenCreditLinesAndLoans', 0], [8, u'NumberOfTimes90DaysLate', 0], [9, u'NumberRealEstateLoansOrLines', 0], [10, u'NumberOfTime60_89DaysPastDueNotWorse', 0], [11, u'NumberOfDependents', 0]]
            }
            expected_list = {
                u'files': [u'projects/53923eccf9ac63682d1784d6/raw/1c2e2ed7-55e7-4af3-a6c9-9ecadc319ea1'],
                u'pid': ObjectId('53923eccf9ac63682d1784d6'),
                u'shape': [1000, 0],
                u'name': 'featurelist',
                u'typeConvert': {u'NumberOfTime30_59DaysPastDueNotWorse': True, u'NumberOfOpenCreditLinesAndLoans': True, u'NumberRealEstateLoansOrLines': True, u'DebtRatio': True, u'NumberOfDependents': True, u'NumberOfTime60_89DaysPastDueNotWorse': True, u'NumberOfTimes90DaysLate': True, u'Unnamed: 0': True, u'age': True, u'MonthlyIncome': True, u'SeriousDlqin2yrs': True, u'RevolvingUtilizationOfUnsecuredLines': True},
                u'controls': {u'dirty': False, u'projects/53923eccf9ac63682d1784d6/raw/1c2e2ed7-55e7-4af3-a6c9-9ecadc319ea1': {u'type': u'csv', u'encoding': u'ASCII'}},
                u'varTypeString': '',
                u'originalName':
                'featurelist', u'columns': []
            }
            mocks['persistent'].read.side_effect = [[dataset_list1],[dataset_list2]]
            # this should error since the featurelist already exists in our mock
            with self.assertRaises(ValueError):
                self.fl_service.create_dataset('featurelist',[])

    def test_get_dataset_id(self):
        with patch.multiple(self.fl_service, persistent = DEFAULT, assert_can_view = DEFAULT, assert_can_edit = DEFAULT) as mocks:
            fake_meta = {
                'name': 'featurelist',
                '_id': ObjectId('53923eccf9ac636954753b8f')
            }
            mocks['persistent'].read.return_value = [fake_meta]
            dataset_id = self.fl_service.get_dataset_id('featurelist')
            self.assertEqual(dataset_id,fake_meta['_id'])

    def test_get_feature_names(self):
        with patch.multiple(self.fl_service, persistent = DEFAULT, assert_can_view = DEFAULT, assert_can_edit = DEFAULT) as mocks:
            fake_meta = {
                'name': 'featurelist',
                'columns': [(0,'col1',0),(1,'col2',0)]
            }
            mocks['persistent'].read.return_value = [fake_meta]
            cols = self.fl_service.get_feature_names('featurelist')
            self.assertEqual(cols,['col1','col2'])

if __name__ == '__main__':
    unittest.main()
