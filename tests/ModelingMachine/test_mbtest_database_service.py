import unittest
from bson import ObjectId

from config.test_config import db_config
from common.wrappers import database
import tests.ModelingMachine.mbtest_database_service as mbtest_database_service

class TestMBTestDatabaseService(unittest.TestCase):

    def setUp(self):
        self.dbname = 'test_mbtest_tmpdb'
        self.persistent = database.new('persistent', dbname=self.dbname)

    def test_update_instance_ids_new_ids(self):
        run_id = ObjectId()
        data = { 'instance_ids' : [ 'test_id_1', 'test_id_2', 'test_id_3' ]}
        self.persistent.conn['mbtest_run'].insert({'_id':run_id})

        # test creates new entry
        mbtest_database_service.MBTestDatabaseService(persistent=self.persistent
                ).update_instance_ids(run_id, data['instance_ids'])

        test_result = self.persistent.conn['mbtest_run'].find_one({'_id': run_id})
        self.assertEqual(test_result['instance_ids'], data['instance_ids'])

    def test_update_instance_ids_no_dupes(self):
        run_id = ObjectId()
        data = { 'instance_ids' : [ 'test_id_1', 'test_id_2', 'test_id_3' ]}
        self.persistent.conn['mbtest_run'].insert({'_id':run_id})

        mbtest_database_service.MBTestDatabaseService(persistent=self.persistent
                ).update_instance_ids(run_id, data['instance_ids'])

        # test updates entry without duplicates
        mbtest_database_service.MBTestDatabaseService(persistent=self.persistent
                ).update_instance_ids(run_id, ['test_id_3', 'test_id_4'])

        test_result = self.persistent.conn['mbtest_run'].find_one({'_id': run_id})
        self.assertEqual(test_result['instance_ids'],
                [ 'test_id_1', 'test_id_2', 'test_id_3', 'test_id_4' ])

    def test_update_instance_ids_empty_list(self):
        run_id = ObjectId()
        data = { 'instance_ids' : [ 'test_id_1', 'test_id_2', 'test_id_3' ]}
        self.persistent.conn['mbtest_run'].insert({'_id':run_id})

        mbtest_database_service.MBTestDatabaseService(persistent=self.persistent
                ).update_instance_ids(run_id, data['instance_ids'])

        # test works with empty list
        mbtest_database_service.MBTestDatabaseService(persistent=self.persistent
                ).update_instance_ids(run_id, [])

        test_result3 = self.persistent.conn['mbtest_run'].find_one({'_id': run_id})
        self.assertEqual(test_result3['instance_ids'],
                [ 'test_id_1', 'test_id_2', 'test_id_3' ])

    def tearDown(self):
        self.persistent.db_connection.drop_database(self.persistent.dbname)
