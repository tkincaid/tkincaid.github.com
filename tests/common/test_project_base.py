import unittest
import pytest
from bson.objectid import ObjectId
from mock import Mock, patch

from config.test_config import db_config as config
from common.wrappers import database

from common.services.project import ProjectServiceBase as ProjectService


class TestProjectClass(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.tempstore = database.new("tempstore", host=config['tempstore']['host'],
                port=config['tempstore']['port'])
        self.persistent = database.new("persistent", host=config['persistent']['host'],
                port=config['persistent']['port'], dbname=config['persistent']['dbname'])

    @classmethod
    def tearDownClass(self):
        self.persistent.destroy(table='project')
        self.persistent.destroy(table='leaderboard')
        self.persistent.destroy(table='metadata')
        self.persistent.destroy(table='prediction_tabulation')

    def test_prediction_tabulation(self):
        lid = str(ObjectId())
        pid = str(ObjectId())
        dataset_id = '1234'
        doc = {'testkey':'asdf'}

        project = ProjectService(pid)
        project.write_prediction_tabulation(lid, dataset_id, doc)
        out = project.read_prediction_tabulation(lid)

        self.assertGreaterEqual(set(out.keys()), set(['lid','pid','dataset_id','plotdata']))

    def test_delete_leaderboard_item(self):
        pid = str(ObjectId())

        project = ProjectService(pid)
        project.assert_has_permission = Mock()

        lid = self.persistent.create({'pid':project.pid}, table='leaderboard')

        check = project.read_leaderboard_item({'_id':lid})
        self.assertEqual(check, {'pid':project.pid, '_id':lid})

        project.delete_leaderboard_item(str(lid))
        self.assertEqual(project.assert_has_permission.call_count, 1)

        check = project.read_leaderboard_item({'_id':lid})
        self.assertEqual(check, None)

    def test_set_recommender(self):
        pid = str(ObjectId())
        dataset_id = str(ObjectId())

        project = ProjectService(pid)
        project.assert_has_permission = Mock()
        project.get_metadata = Mock()
        project.get_metadata.return_value = {'columns':[(0,'uid',0),(1,'iid',1)],'varTypeString':'CC'}
        req = {'recommender_user_id':'uid','recommender_item_id':'iid'}

        out = project.set_recommender(req, dataset_id)

        p = self.persistent.read(table='project', condition={'_id':ObjectId(pid)}, result={})
        self.assertEqual(p['recommender'], {'user_id':'uid', 'item_id':'iid'})

    def test_init_project_default_values(self):
        fake_pid = ObjectId('53b21c358bd88f3102780a73')
        fake_uid = ObjectId('5359d6cb8bd88f5cddefd3a8')
        fake_server_file_identifier = '5dccf898-0aff-4e63-b5b4-93e9b2367ed6'
        fake_filename = u'projects/{}/raw/{}'.format(
            fake_pid, fake_server_file_identifier)
        fake_dataset_id = ObjectId('53b21c468bd88f31fa96c2ca')

        project = ProjectService(fake_pid, fake_uid)

        project.init_project(filename=fake_filename,
                             originalName='MyUploadedFile.csv',
                             default_dataset_id=fake_dataset_id)

        p = self.persistent.read(table='project', condition={'_id':fake_pid},
                                 result={})

        project_initial_keys = ['originalName', '_id', 'default_dataset_id',
                                'created', 'active', 'holdout_unlocked',
                                'stage', 'uid']
        self.assertEqual(set(project_initial_keys), set(p.keys()))
        self.assertTrue(p['active'])
        self.assertFalse(p['holdout_unlocked'])
        self.assertEqual(p['stage'], 'eda:')

    def test_set_weights(self):
        pid = str(ObjectId())
        dataset_id = str(ObjectId())

        project = ProjectService(pid)
        project.assert_has_permission = Mock()
        project.get_metadata = Mock()

        values = {'weight':'weight_column', 'offset':'offset_column'}

        #case 1: valid request
        project.get_metadata.return_value = {'columns':[(0,'weight_column',0),(1,'offset_column',1)],'varTypeString':'NN'}

        # action
        project.set_weights(values, dataset_id)

        # assertions
        project.get_metadata.assert_called_once_with(dataset_id)

        meta = self.persistent.read(table='metadata', condition={'_id':ObjectId(dataset_id)}, result={})
        self.assertEqual(meta['varTypeString'],'WO')

        p = self.persistent.read(table='project', condition={'_id':ObjectId(pid)}, result={})
        self.assertEqual(p['weights'], values)

    def test_set_weights_list(self):
        pid = str(ObjectId())
        dataset_id = str(ObjectId())

        project = ProjectService(pid)
        project.assert_has_permission = Mock()
        project.get_metadata = Mock()

        values = {'weight':['weight_column'], 'offset':['offset_col1', 'offset_col2']}

        #case 1: valid request
        project.get_metadata.return_value = {'columns':[(0,'weight_column',0),(1,'offset_col1',0),(2,'offset_col2',0)],'varTypeString':'NNN'}

        # action
        project.set_weights(values, dataset_id)

        # assertions
        project.get_metadata.assert_called_once_with(dataset_id)

        meta = self.persistent.read(table='metadata', condition={'_id':ObjectId(dataset_id)}, result={})
        self.assertEqual(meta['varTypeString'],'WOO')

        p = self.persistent.read(table='project', condition={'_id':ObjectId(pid)}, result={})
        self.assertEqual(p['weights'], values)


    def test_set_weight_failure_column_not_in_dataset(self):
        pid = str(ObjectId())
        dataset_id = str(ObjectId())

        project = ProjectService(pid)
        project.assert_has_permission = Mock()
        project.get_metadata = Mock()

        # failure 1: column not in dataset
        values = {'weight':'weight_column', 'offset':'offset_column'}

        project.get_metadata.return_value = {'columns':[(0,'weight_column',0),(1,'some_column',1)],'varTypeString':'NN'}

        # action
        with self.assertRaises(ValueError) as e:
            project.set_weights(values, dataset_id)

        # assertions
        project.get_metadata.assert_called_once_with(dataset_id)

    def test_set_weight_failure_invalid_request_key(self):
        pid = str(ObjectId())
        dataset_id = str(ObjectId())

        project = ProjectService(pid)
        project.assert_has_permission = Mock()
        project.get_metadata = Mock()

        # failure 2: invalid key in request
        values = {'weight':'weight_column', 'invalid_key':'offset_column'}

        project.get_metadata.return_value = {'columns':[(0,'weight_column',0),(1,'offset_column',1)],'varTypeString':'NN'}

        # action
        with self.assertRaises(ValueError) as e:
            project.set_weights(values, dataset_id)

        # assertions
        self.assertEqual(project.get_metadata.call_count, 0)

    def test_set_weight_failure_variable_ot_numeric(self):
        pid = str(ObjectId())
        dataset_id = str(ObjectId())

        project = ProjectService(pid)
        project.assert_has_permission = Mock()
        project.get_metadata = Mock()

        # failure 3: offset column is not numeric
        values = {'weight':'weight_column', 'offset':'offset_column'}

        project.get_metadata.return_value = {'columns':[(0,'weight_column',0),(1,'offset_column',1)],'varTypeString':'NC'}

        # action
        with self.assertRaises(ValueError) as e:
            project.set_weights(values, dataset_id)

        # assertions
        project.get_metadata.assert_called_once_with(dataset_id)

    def test_create_metrics_list_specified_eda(self):
        pid = str(ObjectId())
        dataset_id = str(ObjectId())

        project = ProjectService(pid)
        project.assert_has_permission = Mock()
        project.get_metadata = Mock()

        # Valid request: no weights and not recommender
        self.persistent.create(values={'_id': str(ObjectId()), 'pid': project.pid, 'eda': {'col1' : { 'metric_options': {'all': [{'short_name': 'metric1'}, {'short_name': 'metric2'}]}}}}, table='eda')
        eda = self.persistent.read(table='eda', condition={'pid': project.pid}, result={})

        metric_map = {'metric1': {'direction': -1}, 'metric2': {'direction': 1}}

        project.create_metrics_list(metric_map, False, False, dataset_id, 'col1', 'Regression', ['metric1', 'metric2'], 'metric2')


        project_db = self.persistent.read(condition={'_id': project.pid}, table='project', result={})
        self.assertIn('metric_detail', project_db)
        metric = project_db['metric_detail']

        self.assertEqual(len(metric), 2)

        for met in metric:
            self.assertIn('ascending', met)
            self.assertIsInstance(met['ascending'], bool)
            self.assertIn('name', met)

        self.assertIn({'name': 'metric1', 'ascending' : True}, metric)
        self.assertIn({'name': 'metric2', 'ascending' : False}, metric)

    def test_create_metrics_list_regression_adds_weighted(self):
        pid = str(ObjectId())
        dataset_id = str(ObjectId())

        project = ProjectService(pid)
        project.assert_has_permission = Mock()
        project.get_metadata = Mock()

        # Valid request: no weights and not recommender
        self.persistent.create(values={'_id': str(ObjectId()), 'pid': project.pid, 'eda': {'col1' : { 'metric_options': {'all': [{'short_name': 'Gini'}, {'short_name': 'RMSE'}]}}}}, table='eda')
        eda = self.persistent.read(table='eda', condition={'pid': project.pid}, result={})
        metric_map = {'RMSE': {'direction': -1}, 'Gini': {'direction': -1}, 'Weighted Gini': {'direction': -1}}

        project.create_metrics_list(metric_map, True, False, dataset_id, 'col1', 'Regression', ['Gini', 'RMSE'], 'RMSE')


        project_db = self.persistent.read(condition={'_id': project.pid}, table='project', result={})
        self.assertIn('metric_detail', project_db)
        metric = project_db['metric_detail']

        self.assertEqual(len(metric), 4)

        for met in metric:
            self.assertIn('ascending', met)
            self.assertIsInstance(met['ascending'], bool)
            self.assertIn('name', met)

        self.assertIn({'name': 'Gini', 'ascending' : True}, metric)
        self.assertIn({'name': 'Weighted Gini', 'ascending' : True}, metric)

    def test_create_metrics_list_classification_adds_recommender(self):
        pid = str(ObjectId())
        dataset_id = str(ObjectId())

        project = ProjectService(pid)
        project.assert_has_permission = Mock()
        project.get_metadata = Mock()

        # Valid request: no weights and not recommender
        self.persistent.create(values={'_id': str(ObjectId()), 'pid': project.pid, 'eda': {'col1' : { 'metric_options': {'all': [{'short_name': 'AUC'}, {'short_name': 'LogLoss'}]}}}}, table='eda')
        eda = self.persistent.read(table='eda', condition={'pid': project.pid}, result={})

        metric_map = {'RMSE': {'direction': -1}, 'AUC': {'direction': -1}, 'Weighted AUC': {'direction': 1}}

        project.create_metrics_list(metric_map, False, True, dataset_id, 'col1', 'Binary', ['AUC', 'LogLoss'], 'AUC')


        project_db = self.persistent.read(condition={'_id': project.pid}, table='project', result={})
        self.assertIn('metric_detail', project_db)
        metric = project_db['metric_detail']

        self.assertGreater(len(metric), 2)

        for met in metric:
            self.assertIn('ascending', met)
            self.assertIsInstance(met['ascending'], bool)
            self.assertIn('name', met)

        self.assertIn({'name': 'LogLoss', 'ascending' : False}, metric)
        self.assertIn({'name': 'Rate@Top10%', 'ascending' : False}, metric)

    def test_create_metrics_list_classification_keeps_weighted_project_metrics_if_selecte_as_project_metrics(self):
        pid = str(ObjectId())
        dataset_id = str(ObjectId())

        project = ProjectService(pid)
        project.assert_has_permission = Mock()
        project.get_metadata = Mock()

        # Valid request: no weights and not recommender
        self.persistent.create(values={'_id': str(ObjectId()), 'pid': project.pid, 'eda': {'col1' : { 'metric_options': {'all': [{'short_name': 'AUC'}, {'short_name': 'LogLoss'}]}}}}, table='eda')
        self.persistent.update(table='project', condition={'_id': project.pid}, values={'metric': 'Weighted RMSE'})
        eda = self.persistent.read(table='eda', condition={'pid': project.pid}, result={})

        metric_map = {'RMSE': {'direction': -1}, 'AUC': {'direction': -1}, 'Weighted AUC': {'direction': 1}}

        project.create_metrics_list(metric_map, False, False, dataset_id, 'col1', 'Binary', ['AUC', 'LogLoss'], target_metric=None)


        project_db = self.persistent.read(condition={'_id': project.pid}, table='project', result={})
        self.assertIn('metric_detail', project_db)
        metric = project_db['metric_detail']
        print metric
        self.assertGreater(len(metric), 2)

        for met in metric:
            self.assertIn('ascending', met)
            self.assertIsInstance(met['ascending'], bool)
            self.assertIn('name', met)

        self.assertIn({'name': 'Weighted RMSE', 'ascending' : False}, metric)
if __name__ == '__main__':
    unittest.main()
