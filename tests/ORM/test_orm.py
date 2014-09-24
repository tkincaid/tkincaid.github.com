import unittest
from mock import patch
from ORM.orm import OnDemandResourceManager
from MMApp.entities.instance import InstanceModel, InstanceRequestModel
from MMApp.entities.instance import InstanceService, INSTANCE_TABLE
from common.wrappers import database
from config.test_config import db_config

class TestORM(unittest.TestCase):

    def setUp(self):
        self.persistent = database.new('persistent')
        self.addCleanup(self.stopPatching)

        self.patchers = []
        orm_worker_mock = patch('ORM.orm.orm_worker')
        self.ORMWorkerMock = orm_worker_mock.start()
        self.patchers.append(orm_worker_mock)

        self.persistent.create
        self.uid = database.ObjectId(None)

    def stopPatching(self):
        for patcher in self.patchers:
            if patcher:
                patcher.stop()

    def tearDown(self):
        self.persistent.destroy(table=INSTANCE_TABLE)

    def test_setup_instance(self):
        class CeleryJob:
            def __init__(self):
                self.id = database.ObjectId(None)

        self.ORMWorkerMock.setup.delay = lambda x: CeleryJob()

        uid = database.ObjectId(None)

        instances = [
            InstanceModel(uid = uid),
            InstanceModel(uid = uid),
        ]

        instance_service = InstanceService(uid)
        request = instance_service.create_request(instances)
        instance_service.save_request(request)

        request_id = request._id
        resource_manager = OnDemandResourceManager()
        resource_manager.setup(request_id)

        db_request = instance_service.get_request(request_id)

        self.assertTrue(db_request.started_on)

