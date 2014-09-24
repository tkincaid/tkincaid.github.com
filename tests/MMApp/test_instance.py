import unittest
from mock import patch
from MMApp.entities.instance import InstanceService, InstanceModel
from MMApp.entities.instance import InstanceRequestModel, InstanceResource
from MMApp.entities.instance import SetupStage as stage, SetupStatus as ss
from MMApp.entities.instance import INSTANCE_TABLE, INSTANCE_REQUEST_TABLE
from common.wrappers import database
from config.test_config import db_config

class TestInstanceUnitTests(unittest.TestCase):
    pass

class TestInstanceDB(unittest.TestCase):

    def setUp(self):
        self.addCleanup(self.stopPatching)
        self.persistent = database.new('persistent')

        self.patchers = []

        mock_api_client = patch('MMApp.entities.instance.ORMAPIClient')
        self.MockAPIClient = mock_api_client.start()
        self.patchers.append(mock_api_client)

    def stopPatching(self):
        for patcher in self.patchers:
            if patcher:
                patcher.stop()

    def tearDown(self):
        self.persistent.destroy(table=INSTANCE_TABLE)

    def test_save(self):
        host_name = 'dr-component-2563.datarobot.com'
        version_deployed = '0a638e'
        uid = database.ObjectId(None)
        instance = InstanceModel(host_name = host_name, version_deployed = version_deployed)
        instance_service = InstanceService(uid = uid)
        instance_id = instance_service.save(instance)
        self.assertTrue(instance_id)

        result = self.persistent.read(table=INSTANCE_TABLE, condition={'uid': uid},
            result = [])
        self.assertTrue(result)
        self.assertEqual(len(result), 1)

    def test_get_by_id(self):
        uid = database.ObjectId(None)
        instance_id = self.persistent.create(table=INSTANCE_TABLE, values={'uid': uid})
        self.assertTrue(instance_id)

        instance_service = InstanceService(uid = uid)
        instance = instance_service.get(instance_id)
        self.assertIsInstance(instance, InstanceModel)
        instance._id = instance_id

    def test_get_list_includes_all(self):
        uid = database.ObjectId(None)
        self.persistent.create(table=INSTANCE_TABLE, values={'uid': uid, 'setup_stage' : stage.TERMINATE})
        self.persistent.create(table=INSTANCE_TABLE, values={'uid': uid, 'setup_stage' : stage.LAUNCH})
        self.persistent.create(table=INSTANCE_TABLE, values={'uid': uid, 'setup_stage' : stage.SUSPEND})

        instance_service = InstanceService(uid = uid)
        instance = instance_service.get(include_all = True)
        self.assertIsInstance(instance, list)
        self.assertEqual(len(instance), 3)

    def test_get_list_excludes_terminated_and_suspended(self):
        uid = database.ObjectId(None)
        self.persistent.create(table=INSTANCE_TABLE, values={'uid': uid, 'setup_stage' : stage.TERMINATE})
        self.persistent.create(table=INSTANCE_TABLE, values={'uid': uid, 'setup_stage' : stage.LAUNCH})
        self.persistent.create(table=INSTANCE_TABLE, values={'uid': uid, 'setup_stage' : stage.SUSPEND})

        instance_service = InstanceService(uid = uid)
        instance = instance_service.get(include_all = False)
        self.assertIsInstance(instance, list)
        self.assertEqual(len(instance), 1)

    def test_save_instances(self):
        uid = database.ObjectId(None)
        instances =  [
            InstanceModel(uid = uid),
            InstanceModel(uid = uid),
        ]

        instance_service = InstanceService(uid)
        instance_service.save_instances(instances)

        result = self.persistent.read(table=INSTANCE_TABLE, condition={'uid': uid},
            result = [])
        self.assertTrue(result)
        self.assertEqual(len(result), 2)

    def test_save_request(self):
        uid = database.ObjectId(None)
        request = InstanceRequestModel(uid = uid)

        instance_service = InstanceService(uid)
        instance_service.save_request(request)

        result = self.persistent.read(table=INSTANCE_REQUEST_TABLE, condition={'uid': uid},
            result = [])
        self.assertTrue(result)
        self.assertEqual(len(result), 1)

    def test_update_instance(self):
        uid = database.ObjectId(None)
        instance_id = self.persistent.create(table=INSTANCE_TABLE, values={'uid': uid})
        instance_service = InstanceService(uid)

        instance = InstanceModel(_id = instance_id, status = 99)
        instance_service.update(instance)

        result = self.persistent.read(table=INSTANCE_TABLE, condition={'_id': instance_id},
            result = {})

        db_instance = InstanceModel.from_dict(result)

        self.assertEqual(db_instance._id, instance._id )

    def test_update_request(self):
        uid = database.ObjectId(None)
        request_id = self.persistent.create(table=INSTANCE_REQUEST_TABLE, values={'uid': uid})
        instance_service = InstanceService(uid)

        request = InstanceRequestModel(_id = request_id, started_on = 616565)
        instance_service.update_request(request)

        result = self.persistent.read(table=INSTANCE_REQUEST_TABLE, condition={'_id': request_id},
            result = {})

        db_request = InstanceRequestModel.from_dict(result)

        self.assertEqual(db_request._id, request._id )
        self.assertEqual(db_request.started_on, request.started_on)

    def test_update_setup_status(self):
        uid = database.ObjectId(None)
        request_id = self.persistent.create(table=INSTANCE_TABLE, values={'uid': uid})
        instance_service = InstanceService(uid)

        instance = InstanceModel(_id = request_id, setup_status = ss.FAILED, setup_stage = stage.LAUNCH)

        instance_service.update(instance, True)

        result = self.persistent.read(table=INSTANCE_TABLE, condition={'_id': request_id},
            result = {})

        db_request = InstanceModel.from_dict(result)

        self.assertEqual(db_request.setup_status, ss.FAILED )
        self.assertGreater(db_request.setup_status_changed_on, 0)

    def test_stage_completed(self):
        instance = InstanceModel(setup_status = ss.FAILED)
        self.assertTrue(instance.setup_failed())

        instance = InstanceModel(setup_status = ss.COMPLETED)
        self.assertFalse(instance.setup_failed())

    def test_get_request(self):
        '''
            Integration test for create_request, save_request and get_request
        '''
        uid = database.ObjectId(None)
        instances = [
            InstanceModel(uid = uid),
            InstanceModel(uid = uid),
        ]

        instance_service = InstanceService(uid)
        request = instance_service.create_request(instances)
        instance_service.save_request(request)

        request_db = instance_service.get_request(str(request._id))

        self.assertEqual(request.uid, request_db.uid)

        for i in request.instances:
            db_i = next(db_i for db_i in request_db.instances if db_i._id == i._id)
            self.assertEqual(db_i.uid, db_i.uid)
            self.assertEqual(db_i.setup_status, db_i.setup_status)

    def test_get_failed_instances(self):
        uid = database.ObjectId(None)
        instances = [
            InstanceModel(uid = uid,
                setup_stage = stage.LAUNCH, setup_status = ss.COMPLETED),
            InstanceModel(uid = uid,
                setup_stage = stage.PROVISION, setup_status = ss.FAILED),
            InstanceModel(uid = uid,
                setup_stage = stage.LAUNCH, setup_status = ss.REQUESTED),
            InstanceModel(uid = uid,
                setup_stage = stage.DEPLOY, setup_status = ss.FAILED),
        ]

        instance_service = InstanceService(uid)
        for i in instances:
            i._id = self.persistent.create(table=INSTANCE_TABLE, values=i.to_dict())

        db_instances = instance_service.get_failed_instances()
        self.assertTrue(db_instances)
        self.assertEqual(len(db_instances), 2)

        # Verify records match by id and status
        for db_i in db_instances:
            i = next(i for i in instances if i._id == db_i._id)
            self.assertEqual(db_i.setup_status, i.setup_status)
            self.assertEqual(db_i.setup_stage, i.setup_stage)

    def test_get_in_progress_instances(self):
        instances = [
            InstanceModel(setup_stage = stage.LAUNCH, setup_status = ss.INPROCESS),
            InstanceModel(setup_stage = stage.PROVISION, setup_status = ss.COMPLETED),
            InstanceModel(setup_stage = stage.LAUNCH, setup_status = ss.INPROCESS),
            InstanceModel(setup_stage = stage.DEPLOY, setup_status = ss.COMPLETED),
            InstanceModel(setup_stage = stage.PING, setup_status = ss.INPROCESS),
            InstanceModel(setup_stage = stage.PING, setup_status = ss.FAILED),
        ]

        instance_service = InstanceService(None)
        for i in instances:
            i._id = self.persistent.create(table=INSTANCE_TABLE, values=i.to_dict())

        db_instances = instance_service.get_in_progress_instances()
        self.assertTrue(db_instances)
        self.assertEqual(len(db_instances), 3)

        # Verify records match by id and status
        for db_i in db_instances:
            i = next(i for i in instances if i._id == db_i._id)
            self.assertEqual(db_i.setup_status, i.setup_status)
            self.assertEqual(db_i.setup_stage, i.setup_stage)
            self.assertFalse(db_i.is_deployed())

    def test_get_deployed_instances(self):
        instances = [
            InstanceModel(setup_stage = stage.PROVISION, setup_status = ss.COMPLETED),
            InstanceModel(setup_stage = stage.DEPLOY, setup_status = ss.INPROCESS),
            InstanceModel(setup_stage = stage.DEPLOY, setup_status = ss.COMPLETED),
            InstanceModel(setup_stage = stage.PING, setup_status = ss.INPROCESS),
            InstanceModel(setup_stage = stage.PING, setup_status = ss.FAILED),
            InstanceModel(setup_stage = stage.PING, setup_status = ss.COMPLETED),
        ]

        instance_service = InstanceService(None)
        for i in instances:
            i._id = self.persistent.create(table=INSTANCE_TABLE, values=i.to_dict())

        db_instances = instance_service.get_deployed_instances()
        self.assertTrue(db_instances)
        self.assertEqual(len(db_instances), 4)

        # Verify records match by id and status
        for db_i in db_instances:
            i = next(i for i in instances if i._id == db_i._id)
            self.assertEqual(db_i.setup_status, i.setup_status)
            self.assertEqual(db_i.setup_stage, i.setup_stage)
            self.assertTrue(db_i.is_deployed())

    def test_mark_as_failed(self):
        uid = database.ObjectId(None)
        instance = InstanceModel(uid = uid, setup_stage = stage.LAUNCH, setup_status = ss.INPROCESS)

        instance._id = self.persistent.create(table=INSTANCE_TABLE, values=instance.to_dict())

        instance_service = InstanceService(uid)
        instance_service.mark_as_failed(instance)

        instance_db = self.persistent.read(table=INSTANCE_TABLE, condition={'_id': instance._id},
            result = {})

        instance_db = InstanceModel.from_dict(instance_db)

        self.assertTrue(instance_db.setup_status_changed_on)
        # Mark as failed
        self.assertEqual(instance_db.setup_status, ss.FAILED)
        # Mantain original action
        self.assertEqual(instance_db.setup_stage, stage.LAUNCH)

        # Make sure other states are not turned on
        self.assertNotEqual(instance_db.setup_status, ss.INPROCESS)
        self.assertNotEqual(instance_db.setup_status, ss.COMPLETED)
        self.assertNotEqual(instance_db.setup_status, ss.REQUESTED)

    @patch('MMApp.entities.instance.requests', autospec=True)
    def test_ping_instance_not_found(self, MockRequests):
        response = MockRequests.get.return_value
        response.status_code = 404
        instance_service = InstanceService(None)
        instance = InstanceModel(
            resource = InstanceResource.PREDICTION_API,
            host_name = 'ec2-54-85-38-114.compute-1.amazonaws.com')
        result = instance_service.ping_instance(instance)
        self.assertFalse(result)

    @patch('MMApp.entities.instance.requests', autospec=True)
    def test_ping_instance(self, MockRequests):
        response = MockRequests.get.return_value
        response.status_code = 200
        response.json.return_value = {'response': 'pong', 'token': '123'}

        instance_service = InstanceService(None)

        with patch.object(instance_service, 'get_random_token', return_value='123'):
            instance = InstanceModel(
                resource = InstanceResource.PREDICTION_API,
                host_name = 'ec2-54-85-38-114.compute-1.amazonaws.com'
            )
            result = instance_service.ping_instance(instance)
            self.assertTrue(result)

    def test_not_supported_ping_instance(self):
        with self.assertRaises(ValueError):
            instance = InstanceModel(resource = 'x')
            InstanceService(None).ping_instance(instance)

    def test_setup_complete_with_control_actions(self):
        instance = InstanceModel(setup_status = ss.COMPLETED, setup_stage = stage.SUSPEND)
        self.assertFalse(instance.setup_complete(stage.TERMINATE))

        instance = InstanceModel(setup_status = ss.COMPLETED, setup_stage = stage.TERMINATE)
        self.assertFalse(instance.setup_complete(stage.STOP))

        instance = InstanceModel(setup_status = ss.COMPLETED, setup_stage = stage.STOP)
        self.assertFalse(instance.setup_complete(stage.START))

    def test_setup_complete_no_argument(self):
        instance = InstanceModel(setup_status = ss.COMPLETED)
        self.assertTrue(instance.setup_complete())

        instance = InstanceModel(setup_status = ss.REQUESTED)
        self.assertFalse(instance.setup_complete())

    def test_setup_complete_with_argument(self):
        instance = InstanceModel(setup_status = ss.COMPLETED, setup_stage = stage.PROVISION)
        self.assertTrue(instance.setup_complete(stage.LAUNCH))

        instance = InstanceModel(setup_status = ss.COMPLETED, setup_stage = stage.DEPLOY)
        self.assertTrue(instance.setup_complete(stage.PROVISION))

        instance = InstanceModel(setup_status = ss.COMPLETED, setup_stage =stage.LAUNCH)
        self.assertTrue(instance.setup_complete(stage.LAUNCH))

        instance = InstanceModel(setup_status = ss.REQUESTED, setup_stage =stage.PROVISION)
        self.assertFalse(instance.setup_complete(stage.PROVISION))

        instance = InstanceModel(setup_status = ss.FAILED, setup_stage =stage.PROVISION)
        self.assertFalse(instance.setup_complete(stage.PROVISION))

        # Deploy failed but we passed this stage already
        instance = InstanceModel(setup_status = ss.FAILED, setup_stage =stage.DEPLOY)
        self.assertTrue(instance.setup_complete(stage.PROVISION))

    def test_start_updates_status_and_stage(self):
        uid = database.ObjectId(None)
        instance_id = self.persistent.create(table=INSTANCE_TABLE, values={'uid':uid})
        instance_id = str(instance_id)
        instance_service = InstanceService(uid)

        instance = instance_service.stop(instance_id)

        self.assertEqual(instance.setup_stage, stage.STOP)
        self.assertEqual(instance.setup_status, ss.REQUESTED)
        self.assertTrue(instance.setup_status_changed_on)

