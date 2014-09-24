import unittest
from MMApp.entities.dataset import DatasetService, UploadedFile
from MMApp.entities.featurelist import FeaturelistService
import config.test_config as config
from config.engine import EngConfig
from bson.objectid import ObjectId
import os
from mock import patch, Mock
import tempfile
import json

from common.wrappers import database
from common.storage import FileObject
from MMApp.entities.db_conn import DBConnections

PROJECT_TABLE = 'project'

class TestDatasetService(unittest.TestCase):

    test_pid = str(ObjectId())
    test_qid = '1'
    test_uid = str(ObjectId())

    @classmethod
    def setUpClass(cls):
        dbs = DBConnections()
        cls.redis_conn = dbs.get_redis_connection()
        cls.get_collection = dbs.get_collection

    @classmethod
    def tearDownClass(cls):
        DBConnections().destroy_database()
        cls.clear_tempstore_except_workers()

    @classmethod
    def clear_tempstore_except_workers(self):
        workers = set(self.redis_conn.smembers('workers'))
        secure_workers = set(self.redis_conn.smembers('secure-workers'))
        ide_workers = set(self.redis_conn.smembers('ide-workers'))
        self.redis_conn.flushdb()
        if workers:
            self.redis_conn.sadd('workers', *workers)
        if secure_workers:
            self.redis_conn.sadd('secure-workers', *secure_workers)
        if ide_workers:
            self.redis_conn.sadd('ide-workers', *ide_workers)

    def setUp(self):
        self.service = DatasetService()
        self.fl_service = FeaturelistService()

    def tearDown(self):
        self.clear_tempstore_except_workers()

    def test_validate_filesize(self):
        #Valid file
        mmapp = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filename = os.path.join(mmapp, 'testdata', 'good_generated.csv')

        self.service.validate_file_size(filename)

    def test_save_metadata(self):
        #Invalid Id
        pid = '123'
        uploaded_file = UploadedFile(dataset_id = 'x', pid=pid, original_filename = 'x')
        self.assertRaises(ValueError, self.service.save_metadata, pid, uploaded_file)
        #Good Id

        pid = '313233343536373839303930'
        uploaded_file = UploadedFile(dataset_id = 'unit_test_dataset_id', pid=pid, original_filename = 'unit_test_file_name')
        oid =  self.service.save_metadata(pid, uploaded_file)
        self.assertIsNotNone(oid)
        #Make sure it was saved
        doc = self.get_collection('metadata').find_one({'dataset_id': uploaded_file.dataset_id, 'pid': ObjectId(pid)})
        self.assertIsNotNone(doc)
        self.assertEqual(doc['dataset_id'], uploaded_file.dataset_id)

    @patch('MMApp.entities.dataset.ProjectService', autospec=True)
    def test_delete_dataset_only_removes_new_data(self, MockProjectService):
        persistent = database.new('persistent')
        uid = ObjectId()
        roles = { str(uid) : [  'OWNER' ] }
        pid = persistent.create(table=PROJECT_TABLE, values = {'uid' : uid, 'roles': roles})

        # Non-prediction datasets are OK
        dataset_id = persistent.create(table='metadata', values = {'pid' : pid})

        service = DatasetService(pid, uid, dataset_id)
        with patch.object(service, 'assert_can_edit', return_value = True):
            service.delete_dataset()

        dataset = persistent.read(table='metadata', condition = {'_id' : dataset_id}, result = [])

        self.assertTrue(dataset)

        # New data on the other hand...
        persistent.update(table='metadata', values = {'newdata': True},
            condition = {'_id' : dataset_id })

        service = DatasetService(pid, uid, dataset_id)
        with patch.object(service, 'assert_can_edit', return_value = True):
            service.delete_dataset()

        dataset = persistent.read(table='metadata', condition = {'_id' : dataset_id}, result = [])

        self.assertFalse(dataset)

    def test_set_prediction_as_complete(self):
        # Deletes lid from computing and removes the deleted prediction flag
        pid = ObjectId()
        self.service.pid = pid

        lid = '538e104fa6844e702ac3def9'
        lid2 = '538e104fa6844e702ac3def8'
        dataset_id = self.service.persistent.create(table='metadata', values = {
            'pid' : pid, 'deleted': [ObjectId(lid), ObjectId(lid2) ],
            'computing': [lid, lid2], 'name': 'prediction-dataset'})

        self.service.set_prediction_as_complete(lid, dataset_id)

        metadata = self.service.persistent.read(table = 'metadata',
            condition = {'_id':dataset_id}, result = {})

        deleted = metadata['deleted']
        self.assertEqual(len(deleted), 1)
        self.assertEqual(deleted[0], ObjectId(lid2))

        computing = metadata['computing']
        self.assertEqual(len(computing), 1)
        self.assertEqual(computing[0], lid2)

    def test_delete_lid_from_computing(self):
        persistent = database.new('persistent')
        uid = ObjectId()
        roles = { str(uid) : [  'OWNER' ] }
        pid = persistent.create(table=PROJECT_TABLE, values = {'uid' : uid, 'roles': roles})
        dataset_id = persistent.create(table='metadata',
            values = {'pid' : pid,'newdata': True,'created':'6/3/2014 6:28:50 PM', 'originalName': 'test.csv'})

        self.service.set_as_computing(pid, dataset_id, '538e104fa6844e702ac3def9')
        self.service.set_as_computing(pid, dataset_id, '538e1021a6844e70e415f981')

        cancel_prediction_for_lid = '538e104fa6844e702ac3def9'
        self.service.remove_lid_from_computing(cancel_prediction_for_lid,dataset_id)
        metadata = self.fl_service.get_all_datasets(pid)
        self.assertTrue(metadata)
        has_lid = cancel_prediction_for_lid in metadata[0]['computing']
        self.assertFalse(has_lid)

    def test_remove_deleted_prediction_flag(self):
        persistent = database.new('persistent')
        pid = ObjectId()
        lid_1 = '538e104fa6844e702ac3def9'
        lid_2 = '538e1021a6844e70e415f981'
        dataset_id = persistent.create(table='metadata', values = {'pid' : pid, 'deleted':[
            ObjectId(lid_1),
            ObjectId(lid_2),
        ]})

        self.service.remove_deleted_prediction_flag(lid_1, dataset_id)
        metadata = self.fl_service.get_all_datasets(pid)
        self.assertEqual(len(metadata[0]['deleted']), 1)

        self.service.remove_deleted_prediction_flag(lid_2, dataset_id)
        metadata = self.fl_service.get_all_datasets(pid)
        self.assertEqual(len(metadata[0]['deleted']), 0)

    def test_delete_dataset_predictions_and_files(self):
        FILE_STORAGE_PREFIX = str(ObjectId)
        LOCAL_FILE_STORAGE_DIR = '/tmp'
        persistent = database.new('persistent')
        uid = ObjectId()
        roles = { str(uid) : [  'OWNER' ] }
        pid = persistent.create(table=PROJECT_TABLE, values = {'uid' : uid, 'roles': roles})

        def create_temp_file():
            new_file_name = str(ObjectId())
            tf = tempfile.NamedTemporaryFile(mode='w',delete=False)
            tf.close()
            with patch.dict(EngConfig, {'FILE_STORAGE_PREFIX': FILE_STORAGE_PREFIX,
                'LOCAL_FILE_STORAGE_DIR': LOCAL_FILE_STORAGE_DIR}):
                fo = FileObject(new_file_name)
                fo.move(tf.name)
            return new_file_name

        f1 = create_temp_file()
        f2 = create_temp_file()
        f3 = create_temp_file()

        dr_metadata_1 = persistent.create(table='metadata')
        dr_metadata_1 = str(dr_metadata_1)
        user_dataset_1 = persistent.create(table='metadata', values = {
            'newdata': True, 'files' : [f1] ,'pid': pid })
        user_dataset_1 = str(user_dataset_1)
        user_target_dataset = persistent.create(table='metadata', values = {
            'newdata': True, 'files' : [f2, f3], 'pid': pid})
        user_target_dataset = str(user_target_dataset)

        lid_1 = persistent.create(table='leaderboard', values = {'pid': pid})
        lid_2 = persistent.create(table='leaderboard', values = {'pid': pid})

        prediction_1 = persistent.create(table='predictions', values = {'dataset_id': dr_metadata_1, 'pid': pid, 'lid': lid_1})
        prediction_2 = persistent.create(table='predictions', values = {'dataset_id': user_dataset_1, 'pid': pid, 'lid': lid_2})
        prediction_3 = persistent.create(table='predictions', values = {'dataset_id': user_target_dataset, 'pid': pid, 'lid': lid_2})
        prediction_4 = persistent.create(table='predictions', values = {'dataset_id': user_target_dataset, 'pid': pid, 'lid': lid_2})

        dataset_service = DatasetService(pid, uid, user_target_dataset)
        with patch.object(dataset_service, 'assert_can_edit', return_value = True):
            dataset_service.delete_dataset()

        # Metadata
        metadata = persistent.read(table='metadata', criteria =
            {'pid' : pid}, fields = ['_id'], result =[])

        metadata = [str(dataset['_id']) for dataset in metadata]

        self.assertIn(dr_metadata_1, metadata)
        self.assertIn(user_dataset_1, metadata)
        self.assertNotIn(user_target_dataset, metadata)

        #Predictions
        predictions = persistent.read(table='predictions', criteria =
            {'pid' : pid}, fields = ['dataset_id'], result =[])

        predictions = set([prediction['dataset_id'] for prediction in predictions])

        self.assertIn(dr_metadata_1, predictions)
        self.assertIn(user_dataset_1, predictions)
        self.assertNotIn(user_target_dataset, predictions)

        # Files

        f1 = os.path.join(LOCAL_FILE_STORAGE_DIR, FILE_STORAGE_PREFIX, f1)
        user_dataset_file_exists = os.path.isfile(f1)
        self.assertTrue(user_dataset_file_exists)

        f2 = os.path.join(LOCAL_FILE_STORAGE_DIR, FILE_STORAGE_PREFIX, f2)
        target_dataset_file_exists = os.path.isfile(f2)
        self.assertTrue(target_dataset_file_exists)

        f3 = os.path.join(LOCAL_FILE_STORAGE_DIR, FILE_STORAGE_PREFIX, f3)
        target_dataset_file_exists = os.path.isfile(f3)
        self.assertTrue(target_dataset_file_exists)

    def test_get_project_data_sample(self):
        uid = ObjectId()
        roles = { str(uid) : [  'OWNER' ] }
        pid = self.service.persistent.create(table=PROJECT_TABLE, values = {'uid' : uid,
            'roles': roles,
        })
        self.service.pid = pid

        tests = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(tests, 'testdata', 'kickcars-sample-200.csv')

        mocked_dataframe = Mock()
        mocked_dataframe.shape = {}
        mocked_dataframe.columns = []

        dataset_id = self.service.create_metadata('universe', [file_path],
            controls = {}, dataframe =  mocked_dataframe)

        self.assertIsNotNone(dataset_id)
        self.assertIsInstance(dataset_id, ObjectId)

        self.service.persistent.update(table= PROJECT_TABLE, condition = {'_id': pid},
            values = {'default_dataset_id': dataset_id})

        data = self.service.get_project_data_sample(pid)
        self.assertTrue(data)
        records = json.loads(data)
        self.assertIn('RefId', records[0])
        self.assertIn('IsBadBuy', records[0])

    def test_dataset_is_removed_if_it_already_exists(self):
        mocked_dataframe = Mock()
        mocked_dataframe.shape = {}
        mocked_dataframe.columns = []

        pid = ObjectId()
        service = DatasetService(pid = pid)

        for i in range(2):
            service.create_metadata('universe', ['file_path'],
                controls = {}, dataframe =  mocked_dataframe)

        metadata_records = service.persistent.read(table='metadata',
            condition={'pid':pid, 'name':'universe'}, result = [])

        self.assertTrue(metadata_records)
        self.assertEqual(len(metadata_records), 1)

if __name__ == '__main__':
    unittest.main()
