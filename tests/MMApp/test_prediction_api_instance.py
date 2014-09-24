import unittest
from MMApp.entities.prediction_api_instance import ApiInstanceService
from MMApp.entities.instance import INSTANCE_TABLE
from common.wrappers import database
from bson import ObjectId
from config.test_config import db_config

class TestInstanceUnitTests(unittest.TestCase):
    pass

class TestInstanceDB(unittest.TestCase):

    def setUp(self):
        self.persistent = database.new('persistent')

    def tearDown(self):
        self.persistent.destroy(table=INSTANCE_TABLE)

    def test_activate_model_adds_lid(self):
        uid = ObjectId()
        instance_id = self.persistent.create(table=INSTANCE_TABLE, values={'uid': uid})
        instance_service = ApiInstanceService(uid)

        lid = ObjectId()
        instance_service.activate_model(instance_id, lid)
        lid2 = ObjectId()
        instance_service.activate_model(instance_id, lid2)

        instance = self.persistent.read(table=INSTANCE_TABLE, result = [],
            condition={'uid': uid, '_id': instance_id})

        self.assertTrue(instance)
        models = instance[0]['models']
        self.assertIn(lid, models)
        self.assertIn(lid2, models)

    def test_deactivate_model_removes_lid(self):
        uid = ObjectId()
        lid = ObjectId()
        lid2 = ObjectId()
        instance_id = self.persistent.create(table=INSTANCE_TABLE, values={
            'uid':uid,
            'models': [lid, lid2]
        })

        instance_service = ApiInstanceService(uid)

        instance_service.deactivate_model(instance_id, lid)

        instance = self.persistent.read(table=INSTANCE_TABLE, result = [],
            condition={'uid': uid, '_id': instance_id})

        self.assertTrue(instance)
        models = instance[0]['models']
        self.assertNotIn(lid, models)
        self.assertIn(lid2, models)


