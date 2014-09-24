import unittest
from bson import ObjectId
import json
import os
import config.test_config as config

from common.services.eda import EdaService
from common.wrappers import database

class TestEdaService(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        self.persistent = database.new('persistent')
        self.persistent.db_connection.drop_database(self.persistent.dbname)

    def test_get_eda_metrics_list_works_across_multiple_records(self):
        BATCH_SIZE = 2
        pid = ObjectId()
        eda_service = EdaService(pid, uid=None, verify_permissions=False)
        eda_service.max_batch_size = BATCH_SIZE

        eda_file = os.path.join(os.path.dirname(__file__),'../testdata/fixtures/fastiron-eda1.json')

        with open(eda_file) as f:
            full_eda = json.loads(f.read())

        eda = {}
        for i in xrange(3):
            k = full_eda.keys()[i]
            eda[k] = full_eda.pop(k)

        eda_service.update(eda)

        eda_map = eda_service.eda_map

        features = set(eda_map['column_location'].keys())
        expected_blocks = len(features)/BATCH_SIZE + 1

        distinct_blocks = set(eda_map['block_contents'].keys())
        self.assertEqual(len(distinct_blocks), expected_blocks)

        for k in eda.keys():
            metrics = eda_service.get_target_metrics_list(k)
            self.assertTrue(metrics)
