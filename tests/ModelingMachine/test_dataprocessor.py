#!/usr/bin/python
# -*- coding: utf-8 -*-
############################################################################
#
#       unit test for DataProcessor
#
#       Author: Tom DeGodoy
#
#       Copyright DataRobot, Inc. 2013
#
###########################################################################
import pandas
import numpy as np
from mock import Mock, patch
import config.test_config

import pandas as pd
import numpy as np
import sys
import unittest
import os
import shutil
import itertools
import json
import datetime
from bson import ObjectId
from mock import patch

from cStringIO import StringIO

mm = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(mm,'../..'))

from config.engine import EngConfig
from ModelingMachine.engine.data_processor import DataProcessor, record_failed_prediction_request
from ModelingMachine.engine.worker_request import WorkerRequest
import common.storage as storage
import common.io.query as pq
import common.exceptions as ex
from common.io import dataset_reader
from tests.IntegrationTests.storage_test_base import StorageTestBase

class MockDataProcessor(DataProcessor):
    """Mock DP to test the methods that don't rely on request. """

    def __init__(self):
        pass

class TestDataProcessor(StorageTestBase):

    @classmethod
    def setUpClass(cls):
        super(TestDataProcessor, cls).setUpClass()

        cls.testdatafile = 'credit-sample-200.csv'
        cls.create_test_files([cls.testdatafile])

        cls.project = {'target': {'name': 'SeriousDlqin2yrs',
                                  'type': 'Binary',
                                  'size': 200},
                       'partition': {
                                  'holdout_pct': 20,
                                  'reps': 5,
                                  'version': 1.1},
                       'target_options': {'missing_maps_to': 0,
                                          'positive_class': 1,
                                          'name': 'SeriousDlqin2yrs'}}

        cls.api_patch = patch('ModelingMachine.engine.data_processor.APIClient')
        cls.api_mock = cls.api_patch.start()

    @classmethod
    def tearDownClass(cls):
        super(TestDataProcessor, cls).tearDownClass()
        cls.api_patch.stop()

    def setUp(self):
        self.api_mock.return_value.get_project.return_value = self.project
        request = WorkerRequest({'pid': '1', 'uid': '1', 'dataset_id': '1', 'command': 'fit', 'max_reps': 0, 'samplepct': 100})
        self.dataprocessor = DataProcessor(request)

    def tearDown(self):
        pass

    def test_data_manipulation(self):
        """
        create training datasets
        create target vector
        create prediction datasets
        """
        target_name = self.project['target']['name']
        self.api_mock.return_value.get_metadata.return_value = [
            {'_id': '0',
             'pid': '1',
             'created': datetime.datetime.now(),
             'name':'universe',
             'originalName': 'credit-sample-200.csv',
             'varTypeString': 'NN',
             'shape': [2, 100],
             'controls':{},
             'columns': [[1,target_name,0],[3,"age",0]],
             'files': ['projects/' + str(self.pid) + '/raw/' + self.testdatafile],
             'typeConvert': {}},
            {'_id': '1',
             'pid': '1',
             'name':'test',
             'originalName': 'credit-sample-200.csv',
             'created': datetime.datetime.now(),
             'varTypeString': 'NN',
             'shape': [2, 100],
             'controls':{},
             'columns': [[1,target_name,0],[3,"age",0]],
             'files': ['projects/' + str(self.pid) + '/raw/' + self.testdatafile],
             'typeConvert': {}},
            {'_id': '2',
             'pid': '1',
             'name':'new',
             'created': datetime.datetime.now(),
             'originalName': 'credit-sample-200.csv',
             'newdata':True,
             'controls':{},
             'shape': [2, 100],
             'varTypeString': 'NN',
             'columns': [[1,target_name,0],[3,"age",0]],
             'files': ['projects/' + str(self.pid) + '/raw/' + self.testdatafile],
             'typeConvert': {}}]
        request = WorkerRequest({'pid': '1', 'uid': '1', 'dataset_id': '1',
                                 'command': 'fit', 'max_reps': 0,
                                 'samplepct': 100})

        #target
        #this will map the target values to (0,1) because target type is Binary
        target_vector = self.dataprocessor.target_vector()
        target_series = target_vector['main']
        self.assertItemsEqual(np.unique(target_series), [0,1])

        #this will be none because 'holdout_pct' isn't set in the project data
        self.assertIsNone(target_vector['holdout'])

        #prediction dataset
        predictors = self.dataprocessor.predictors()
        pred_dataframe = predictors['1']['main']
        self.assertItemsEqual(list(pred_dataframe.columns), ["age"])
        self.assertEqual(self.dataprocessor.get_vartypestring_without_target('1'), "N")

        request = WorkerRequest({'pid': '1', 'uid': '1', 'dataset_id': '1', 'scoring_dataset_id': '2', 'command': 'predict', 'max_reps': 0, 'samplepct':100})
        dp2 = DataProcessor(request)
        data = dp2.request_datasets()
        self.assertEqual(data.keys(), ['1'])
        self.assertEqual(data['1'].keys(), ['scoring', 'vartypes'])
        scoring_data = data['1']['scoring']
        vartypes = data['1']['vartypes']
        self.assertEqual(list(scoring_data.columns), ["age"])
        self.assertEqual(vartypes, "N")

    def test_get_vartypestring_without_target(self):
        target_name = 'target_name'
        self.api_mock.return_value.get_project.return_value = {'target': {'name': target_name}}
        #{'dataset_id': {}}
        self.api_mock.return_value.get_metadata.return_value = [
            {'_id': '1',
             'varTypeString': 'TNN',
             'columns': [[0,target_name,0], [1,'a',0], [2,'b',0]],
             'typeConvert': {}},
            {'_id': '2',
             'varTypeString': 'NTN',
             'columns': [[1,'a',0], [0,target_name,0], [2,'b',0]], 'typeConvert': {}},
            {'_id': '3',
             'varTypeString': 'NNT',
             'columns': [[1,'a',0], [2,'b',0], [0,target_name,0]], 'typeConvert': {}}]

        expected = 'NN'

        self.assertEqual(expected, self.dataprocessor.get_vartypestring_without_target('1'))
        self.assertEqual(expected, self.dataprocessor.get_vartypestring_without_target('2'))
        self.assertEqual(expected, self.dataprocessor.get_vartypestring_without_target('3'))

    def test_get_dataset_ids(self):
        request = {
            'uid': str(ObjectId()),
            'pid': str(ObjectId()),
            'dataset_id': '1',
            'blueprint': {
                '1': [['324234234'],['STK'],'T'],
                '2': [['1'],['GAM'],'P']
            },
            'blender': {
                'inputs': [
                    {
                     'dataset_id': '2',
                     'blueprint': {
                        '1': [['NUM'],['NI'],'T'],
                        '2': [['1'],['USERTASK id=123abc'],'P']
                     }
                    },{
                     'blender': {
                        'inputs': [
                            { 'dataset_id': '3',
                              'blueprint': {
                                '1': [['NUM'],['NI'],'T'],
                                '2': [['1'],['USERTASK id=456def'],'P']
                              }
                            },{ 'dataset_id': '4',
                                'blueprint': {
                                '1': [['NUM'],['NI'],'T'],
                                '2': [['1'],['USERTASK id=789ghi'],'P']
                              }
                            }
                        ],
                      },
                      'dataset_id': '5',
                      'blueprint': {
                          '1': [['234232434'],['STK'],'T'],
                          '2': [['1'],['GLM'],'P']
                      },
                   }
                ]
            },
            'partitions': [[0,-1]],
            'command': 'fit',
            'samplepct': 100
        }
        request = WorkerRequest(request)
        dp = DataProcessor(request)
        self.assertItemsEqual(set(['1', '2', '3', '4', '5']), dp.dataset_ids)

        request = {
            'uid': str(ObjectId()),
            'pid': str(ObjectId()),
            'dataset_id': '1',
            'blueprint': {
                '1': [['NUM'],['NI'],'T'],
                '2': [['1'],['USERTASK id=123abc'],'T'],
                '3': [['2'],['USERTASK id=456def'],'P']
            },
            'partitions': [[0,-1]],
            'samplepct': 100,
            'command': 'fit'
        }

        request = WorkerRequest(request)
        dp = DataProcessor(request)
        #TODO: use bi test helper to create a VALID request
        self.assertItemsEqual(set(['1']), dp.dataset_ids)
        self.assertIsInstance(dp.dataset_ids, set)

    def test_get_task_ids(self):
        request = {
            'uid': str(ObjectId()),
            'pid': str(ObjectId()),
            'blueprint': {
                '1': [['324234234'],['STK'],'T'],
                '2': [['1'],['GAM'],'P']
            },
            'blender': {
                'inputs': [
                    {
                     'blueprint': {
                        '1': [['NUM'],['NI'],'T'],
                        '2': [['1'],['USERTASK id=123abc'],'P']
                     }
                    },{
                     'blender': {
                        'inputs': [
                            { 'blueprint': {
                                '1': [['NUM'],['NI'],'T'],
                                '2': [['1'],['USERTASK id=456def'],'P']
                              }
                            },{ 'blueprint': {
                                '1': [['NUM'],['NI'],'T'],
                                '2': [['1'],['USERTASK id=789ghi'],'P']
                              }
                            }
                        ],
                      },
                      'blueprint': {
                          '1': [['234232434'],['STK'],'T'],
                          '2': [['1'],['GLM'],'P']
                      },
                   }
                ]
            },
            'partitions': [[0,-1]],
            'command': 'fit',
            'samplepct': 100
        }
        request = WorkerRequest(request)
        self.assertItemsEqual(set(['123abc', '456def', '789ghi']),self.dataprocessor.get_user_task_ids(request))
        request = {
            'uid': str(ObjectId()),
            'pid': str(ObjectId()),
            'blueprint': {
                '1': [['NUM'],['NI'],'T'],
                '2': [['1'],['USERTASK id=123abc'],'T'],
                '3': [['2'],['USERTASK id=456def'],'P']
            },
            'partitions': [[0,-1]],
            'samplepct': 100,
            'command': 'fit'
        }
        request = WorkerRequest(request)
        #TODO: use bi test helper to create a VALID request
        self.assertItemsEqual(set(['123abc', '456def']),self.dataprocessor.get_user_task_ids(request))
        self.assertIsInstance(self.dataprocessor.get_user_task_ids(request), set)

    def test_user_tasks(self):
        self.api_mock.return_value.get_task_code_from_id_list.return_value = {'123abc': {'modelfit': '',
                                                                                   'modelpredict': '',
                                                                                   'modelsource': '',
                                                                                   'modeltype': ''
                                                                                   },
                                                                              '456def': {'modelfit': '',
                                                                                   'modelpredict': '',
                                                                                   'modelsource': '',
                                                                                   'modeltype': ''
                                                                                   },
                                                                              '789ghi': {'modelfit': '',
                                                                                   'modelpredict': '',
                                                                                   'modelsource': '',
                                                                                   'modeltype': ''
                                                                                   }
                                                                              }
        request = {
            'pid': '1',
            'uid': '1',
            'samplepct': 100,
            'dataset_id': '1',
            'blueprint': {
                '1': [['324234234'],['STK'],'T'],
                '2': [['1'],['GAM'],'P']
            },
            'blender': {
                'inputs': [
                    {
                     'dataset_id': '2',
                     'blueprint': {
                        '1': [['NUM'],['NI'],'T'],
                        '2': [['1'],['USERTASK id=123abc'],'P']
                     }
                    },{
                     'blender': {
                        'inputs': [
                            { 'dataset_id': '3',
                              'blueprint': {
                                '1': [['NUM'],['NI'],'T'],
                                '2': [['1'],['USERTASK id=456def'],'P']
                              }
                            },{ 'dataset_id': '4',
                                'blueprint': {
                                '1': [['NUM'],['NI'],'T'],
                                '2': [['1'],['USERTASK id=789ghi'],'P']
                              }
                            }
                        ],
                      },
                      'dataset_id': '5',
                      'blueprint': {
                          '1': [['234232434'],['STK'],'T'],
                          '2': [['1'],['GLM'],'P']
                      },
                   }
                ]
            },
            'partitions': [[0,-1]],
            'samplepct': 100,
            'command': 'fit'
        }
        request = WorkerRequest(request)
        dp = DataProcessor(request)
        self.assertItemsEqual(set(['123abc', '456def', '789ghi']), dp.get_user_task_ids(request))

        self.assertItemsEqual(set(['123abc', '456def', '789ghi']), dp.user_tasks.keys())
        for key in dp.user_tasks:
            self.assertItemsEqual(['modelfit', 'modelpredict', 'modelsource', 'modeltype'], dp.user_tasks[key].keys())

    def test_predict_stream(self):
        target_name = self.project['target']['name']
        fake_project_metadata = [
            {'_id': '0',
             'name':'universe',
             'created': datetime.datetime.now(),
             'originalName': 'originalName.csv',
             'pid': '1',
             'controls': {},
             'shape': [2, 100],
             'varTypeString': 'NN',
             'columns': [[1, target_name, 0],[3, "age", 0]],
             'files': ['projects/' + str(self.pid) + '/raw/' + self.testdatafile],
             'typeConvert': {}},
            {'_id': '1',
             'name':'test',
             'created': datetime.datetime.now(),
             'originalName': 'originalName.csv',
             'pid': '1',
             'controls': {},
             'shape': [2, 100],
             'varTypeString': 'NN',
             'columns': [[1, target_name, 0],[3, "age", 0]],
             'files': ['projects/' + str(self.pid) + '/raw/' + self.testdatafile],
             'typeConvert': {}}]
        self.api_mock.return_value.get_metadata.return_value = fake_project_metadata
        metadata = self.dataprocessor.metadata['1']
        reader = dataset_reader.DatasetReader.from_record(metadata)
        dataframe = reader.get_targeted_data('all', self.dataprocessor.project)

        request = {'pid': '1', 'uid': '1', 'dataset_id': '1',
                   'command': 'predict', 'max_reps': 0, 'samplepct': 100}
        request['scoring_dataset_id'] = ''
        request['scoring_data'] = {'mimetype': 'application/json',
                        'stream': dataframe.to_json(orient='records')}

        request = WorkerRequest(request)
        dp = DataProcessor(request)
        data = dp.request_datasets()

        self.assertEqual(data.keys(), ['1'])
        self.assertEqual(data['1'].keys(), ['scoring', 'vartypes'])
        scoring_data = data['1']['scoring']
        columns_without_target = list(dataframe.columns)
        columns_without_target.remove(target_name)
        self.assertItemsEqual(list(scoring_data.columns), columns_without_target)

    def test_empty_stream_detected(self):
        target_name = self.project['target']['name']
        fake_project_metadata = [
            {'_id': '0',
             'name':'universe',
             'created': datetime.datetime.now(),
             'originalName': 'originalName.csv',
             'pid': '1',
             'controls': {},
             'shape': [2, 100],
             'varTypeString': 'NN',
             'columns': [[1, target_name, 0],[3, "age", 0]],
             'files': ['projects/' + str(self.pid) + '/raw/' + self.testdatafile],
             'typeConvert': {}},
            {'_id': '1',
             'name':'test',
             'created': datetime.datetime.now(),
             'originalName': 'originalName.csv',
             'pid': '1',
             'controls': {},
             'shape': [2, 100],
             'varTypeString': 'NN',
             'columns': [[1, target_name, 0],[3, "age", 0]],
             'files': ['projects/' + str(self.pid) + '/raw/' + self.testdatafile],
             'typeConvert': {}}]
        self.api_mock.return_value.get_metadata.return_value = fake_project_metadata

        request = {'pid': '1', 'uid': '1', 'dataset_id': '1',
                   'command': 'predict', 'max_reps': 0, 'samplepct': 100}
        request['scoring_dataset_id'] = ''
        request['scoring_data'] = {'mimetype': 'application/json',
                        'stream': ''}

        request = WorkerRequest(request)
        dp = DataProcessor(request)
        with self.assertRaises(ex.EmptyUploadStreamError):
            data = dp.request_datasets()

    def test_scoring_data_to_dataframe(self):
        df = pd.DataFrame(data=[[1,2,'foo'], [2,1,'bar']], columns=['foo', 'bar', 'name'])
        for fmt in ['json', 'csv']:
            data = StringIO()
            if fmt == 'csv':
                mimetype = 'text/plain'
                df.to_csv(data, index=False)
            elif fmt == 'json':
                df.to_json(data, orient='records')
                mimetype = 'application/json'
            else:
                raise ValueError()
            data = data.getvalue()
            scoring_data = {'mimetype': mimetype,
                            'stream': data}

            dp = MockDataProcessor()
            out = dp.scoring_data_to_dataframe(scoring_data)
            self.assertEqual(sorted(out.columns.tolist()), sorted(df.columns.tolist()))
            np.testing.assert_array_equal(out[df.columns].values,
                                          df[df.columns].values)

    def test_scoring_data_to_dataframe_utf8(self):
        df = pd.DataFrame(data=[[1,2,u'föö'], [2,1,u'bär']], columns=['foo', 'bar', 'name'])
        for fmt in ['csv', 'json']:
            data = StringIO()
            if fmt == 'csv':
                mimetype = 'text/plain'
                df.to_csv(data, index=False, encoding='utf-8')
            elif fmt == 'json':
                df.to_json(data, orient='records')
                mimetype = 'application/json'
            else:
                raise ValueError()
            data = data.getvalue().decode('utf-8')
            scoring_data = {'mimetype': mimetype,
                            'stream': data}

            dp = MockDataProcessor()
            out = dp.scoring_data_to_dataframe(scoring_data)
            self.assertEqual(sorted(out.columns.tolist()), sorted(df.columns.tolist()))
            np.testing.assert_array_equal(out[df.columns].values,
                                          df[df.columns].values)

    @patch.object(DataProcessor, 'project')
    @patch.object(DataProcessor, 'metadata')
    def test_weights(self,*args,**kwargs):
        with patch('ModelingMachine.engine.data_processor.dataset_reader.DatasetReader') as mock_reader:
            request = {'pid': '1', 'uid': '1', 'dataset_id': '1',
                    'command': 'fit', 'partitions':[[-1,-1]], 'samplepct': 100}
            dp = DataProcessor(WorkerRequest(request))
            dp.project.get.return_value = {'weight':['col2'], 'offset':['col3', 'col4']}
            dp.get_dataset_id_by_name = Mock()
            dp.get_dataset_id_by_name.return_value = 'testdatasetid'
            ds = pandas.DataFrame({'col1':np.random.rand(10), 'col2':np.random.rand(10),
                                   'col3':np.random.rand(10), 'col4':np.random.rand(10)})
            mock_reader.from_record.return_value.get_predictors.return_value = ds

            out = dp.weights

            self.assertEqual(mock_reader.from_record.return_value.get_predictors.call_count, 2)

            expected = ds['col2'].values
            self.assertTrue(np.all(out['weight']['main']==expected))
            self.assertTrue(np.all(out['weight']['holdout']==expected))

            expected = ds['col3'].values + ds['col4'].values
            self.assertTrue(np.all(out['offset']['main']==expected))
            self.assertTrue(np.all(out['offset']['holdout']==expected))

    def test_record_failed_prediction_request(self):
        request = {'pid': '1', 'uid': '1', 'dataset_id': '1',
                   'command': 'predict', 'max_reps': 0, 'samplepct': 100}

        ## Test if latin chars if present in prediction set can be logged
        request['scoring_data'] = {'mimetype': 'application/json',
                                   'stream': 'guaranáe=mc²MüellerMaßstab\xa0nobreakᏖෟႳႿ'.decode('utf-16')}
        record_failed_prediction_request(request)

        ## Test if unicode chars if present in prediction set can be logged
        request['scoring_data'] = {'mimetype': 'application/json',
                        'stream': 'Gl\xfcck'.decode('windows-1252')}
        record_failed_prediction_request(request)

        ## Test if ascii chars if present in prediction set can be logged
        request['scoring_data'] = {'mimetype': 'application/json',
                        'stream': 'Gladf, adad'}
        record_failed_prediction_request(request)


if __name__ == '__main__':
    unittest.main()
