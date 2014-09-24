import time
import unittest
from mock import patch, DEFAULT, call, Mock
from MMApp.entities.instance import InstanceService, InstanceModel, INSTANCE_TABLE
from MMApp.entities.instance import SetupStage as stage, SetupStatus as ss
from common.wrappers import database
from ORM import orm_worker
from config.test_config import db_config

USER_TABLE_NAME = 'users'

class TestORMServer(unittest.TestCase):

    def setUp(self):
        self.addCleanup(self.stopPatching)
        self.persistent = database.new('persistent')

        self.patchers = []

        ansible_mock = patch('ORM.orm_worker.ansible_api')
        self.AnsibleMock = ansible_mock.start()
        self.patchers.append(ansible_mock)

        revoke_mock = patch('ORM.orm_worker.revoke')
        self.RemoveMock = revoke_mock.start()
        self.patchers.append(revoke_mock)

        chain_mock = patch('ORM.orm_worker.chain')
        self.ChainMock = chain_mock.start()
        self.patchers.append(chain_mock)


    def stopPatching(self):
        for patcher in self.patchers:
            if patcher:
                patcher.stop()

    def tearDown(self):
        self.persistent.destroy(table=INSTANCE_TABLE)
        self.persistent.destroy(table=USER_TABLE_NAME)

    def mock_launch(self, instance):
        instance.instance_id = 'i-fd195aae'
        instance.private_ip = '10.50.234.33'
        instance.host_name = 'ec2-54-88-149-236.compute-1.amazonaws.com'

    @patch('ORM.orm_worker.InstanceService', autospec = True)
    def test_launch_exits_if_launch_is_complete(self, MockInstanceService):
        instance_service = MockInstanceService.return_value
        instance = InstanceModel(
            _id = database.ObjectId(None),
            setup_status = ss.COMPLETED,
            setup_stage = stage.LAUNCH
        )
        instance_service.get_single_instance.return_value = instance
        orm_worker.launch(instance._id)
        self.assertFalse(self.AnsibleMock.launch.called)

    @patch('ORM.orm_worker.InstanceService', autospec = True)
    def test_provision_exits_if_job_is_in_process_already(self, MockInstanceService):
        instance_service = MockInstanceService.return_value
        instance = InstanceModel(
            _id = database.ObjectId(None),
            setup_status = ss.INPROCESS,
            setup_stage = stage.PROVISION
        )
        instance_service.get_single_instance.return_value = instance
        orm_worker.provision(instance._id)
        self.assertFalse(self.AnsibleMock.provision.called)

    @patch('ORM.orm_worker.InstanceService', autospec = True)
    def test_provision_exits_if_provision_is_complete(self, MockInstanceService):
        instance_service = MockInstanceService.return_value

        instance = InstanceModel(
            _id = database.ObjectId(None),
            setup_status = ss.COMPLETED,
            setup_stage = stage.PROVISION
        )
        instance_service.get_single_instance.return_value = instance
        orm_worker.provision(instance._id)
        self.assertFalse(self.AnsibleMock.provision.called)

        instance = InstanceModel(
            _id = database.ObjectId(None),
            setup_status = ss.FAILED,
            setup_stage = stage.PROVISION
        )
        instance_service.get_single_instance.return_value = instance
        orm_worker.provision(instance._id)
        self.assertTrue(self.AnsibleMock.provision.called)

    def test_launch_raises_exception(self):
        uid = database.ObjectId(None)
        instance_service = InstanceService(uid)
        instance = InstanceModel()
        instance_service.save(instance)

        self.AnsibleMock.launch.side_effect = Exception('BOOM!')

        with self.assertRaises(Exception):
            orm_worker.launch(instance._id)

        instance_db = instance_service.get(instance._id)
        self.assertTrue(instance_db)
        self.assertTrue(instance_db.setup_failed())
        self.assertFalse(instance_db.is_deployed())

    def test_launch_fails(self):
        uid = database.ObjectId(None)
        instance_service = InstanceService(uid)
        instance = InstanceModel()
        instance_service.save(instance)

        with self.assertRaises(ValueError):
            instance = orm_worker.launch(instance._id)

        instance_db = instance_service.get(instance._id)
        self.assertTrue(instance_db)
        self.assertTrue(instance_db.setup_failed())
        self.assertEqual(instance_db.setup_stage, stage.LAUNCH)
        self.assertEqual(instance_db.setup_status, ss.FAILED)
        self.assertFalse(instance_db.is_deployed())

    def test_launch_succeeds(self):
        uid = database.ObjectId(None)
        instance_service = InstanceService(uid)
        instance = InstanceModel()
        instance_service.save(instance)

        self.AnsibleMock.launch = self.mock_launch

        instance_id = orm_worker.launch(instance._id)
        instance_db = instance_service.get(instance_id)
        self.assertTrue(instance_db)
        self.assertTrue(instance_db.private_ip)
        self.assertTrue(instance_db.instance_id)
        self.assertTrue(instance_db.host_name)
        self.assertEqual(instance_db.setup_stage, stage.LAUNCH)
        self.assertEqual(instance_db.setup_status, ss.COMPLETED)
        self.assertFalse(instance_db.setup_failed())

    def test_provision_succeeds(self):
        uid = database.ObjectId(None)
        instance_service = InstanceService(uid)
        instance = InstanceModel()
        instance_service.save(instance)

        instance_id = orm_worker.provision(instance._id)

        instance_db = instance_service.get(instance_id)
        self.assertTrue(instance_db)
        self.assertFalse(instance_db.setup_failed())
        self.assertEqual(instance_db.setup_stage, stage.PROVISION)
        self.assertEqual(instance_db.setup_status, ss.COMPLETED)

    def test_provision_raises_exception(self):
        uid = database.ObjectId(None)
        instance_service = InstanceService(uid)
        instance = InstanceModel()
        instance_service.save(instance)

        self.AnsibleMock.provision.side_effect = Exception('BOOM!')

        with self.assertRaises(Exception):
            orm_worker.provision(instance._id)

        instance_db = instance_service.get(instance._id)
        self.assertTrue(instance_db)
        self.assertTrue(instance_db.setup_failed())
        self.assertEqual(instance_db.setup_stage, stage.PROVISION)
        self.assertEqual(instance_db.setup_status, ss.FAILED)
        self.assertFalse(instance_db.is_deployed())

    def test_deploy_succeeds(self):
        uid = database.ObjectId(None)
        instance_service = InstanceService(uid)
        instance = InstanceModel()
        instance_service.save(instance)

        instance_id = orm_worker.deploy(instance._id)

        instance_db = instance_service.get(instance_id)
        self.assertTrue(instance_db)
        self.assertFalse(instance_db.setup_failed())
        self.assertEqual(instance_db.setup_stage, stage.DEPLOY)
        self.assertEqual(instance_db.setup_status, ss.COMPLETED)

    def test_deploy_raises_exception(self):
        uid = database.ObjectId(None)
        instance_service = InstanceService(uid)
        instance = InstanceModel()
        instance_service.save(instance)

        self.AnsibleMock.deploy.side_effect = Exception('BOOM!')

        with self.assertRaises(Exception):
            orm_worker.deploy(instance._id)

        instance_db = instance_service.get(instance._id)
        self.assertTrue(instance_db)
        self.assertTrue(instance_db.setup_failed())
        self.assertEqual(instance_db.setup_stage, stage.DEPLOY)
        self.assertEqual(instance_db.setup_status, ss.FAILED)
        self.assertFalse(instance_db.is_deployed())

    def test_restart_failed_setup_jobs(self):
        instances = [
            InstanceModel(setup_stage = stage.LAUNCH, setup_status = ss.INPROCESS),
            InstanceModel(setup_stage = stage.LAUNCH, setup_status = ss.FAILED),
            InstanceModel(setup_stage = stage.PROVISION, setup_status = ss.COMPLETED),
            InstanceModel(setup_stage = stage.DEPLOY, setup_status = ss.FAILED),
            InstanceModel(setup_stage = stage.STOP, setup_status = ss.FAILED),
        ]

        with patch.multiple(orm_worker, setup_instance = DEFAULT,
            control = DEFAULT) as mocks:

            orm_worker.restart_failed_jobs(instances)

            self.assertEqual(mocks['setup_instance'].delay.call_count, 2)

    def test_restart_failed_control_jobs(self):
        instances = [
            InstanceModel(setup_stage = stage.LAUNCH, setup_status = ss.INPROCESS),
            InstanceModel(setup_stage = stage.PROVISION, setup_status = ss.FAILED),
            InstanceModel(setup_stage = stage.START, setup_status = ss.COMPLETED),
            InstanceModel(setup_stage = stage.STOP, setup_status = ss.FAILED),
            InstanceModel(setup_stage = stage.TERMINATE, setup_status = ss.FAILED),
        ]

        with patch.multiple(orm_worker, setup_instance = DEFAULT,
            control = DEFAULT) as mocks:

            orm_worker.restart_failed_jobs(instances)

            self.assertEqual(mocks['control'].delay.call_count, 2)

    @patch('ORM.orm_worker.InstanceService', autospec=True)
    def test_kill_hanging_jobs(self, MockInstanceService):
        current_time = time.time()
        instances = [
            InstanceModel(
                type='m1.xlarge',
                setup_status_changed_on = current_time - 1,
                setup_stage =  stage.LAUNCH,
                setup_status = ss.REQUESTED,
            ),
            InstanceModel(
                type='m2.xlarge',
                setup_status_changed_on = current_time - 1500,
                setup_stage =  stage.LAUNCH,
                setup_status = ss.INPROCESS,
            ),
            InstanceModel(
                type='m3.xlarge',
                setup_status_changed_on = current_time - 4000,
                setup_stage =  stage.PROVISION,
                setup_status = ss.INPROCESS,
            ),
            InstanceModel(
                type='m4.xlarge',
                setup_status_changed_on = current_time - 5000,
                setup_stage =  stage.DEPLOY,
                setup_status = ss.REQUESTED,
            ),
        ]

        instance_service = MockInstanceService.return_value

        hanging_jobs = orm_worker.kill_hanging_jobs(instances)

        self.assertEquals(hanging_jobs, 2)

        self.assertEquals(instance_service.mark_as_failed.call_count, 2)
        instance_service.mark_as_failed.assert_has_calls([
            call(instances[2]),
            call(instances[3]),
        ])


    def test_move_jobs_to_next_phase(self):
        current_time = time.time()
        instances = [
            InstanceModel(
                setup_status_changed_on = current_time - 1,
                setup_status = ss.COMPLETED,
                setup_stage = stage.LAUNCH
            ),
            InstanceModel(
                setup_status_changed_on = current_time - 4500,
                setup_status = ss.COMPLETED,
                setup_stage = stage.LAUNCH
            ),
            InstanceModel(
                setup_status_changed_on = current_time - 4000,
                setup_status = ss.INPROCESS,
                setup_stage = stage.PROVISION
            ),
            InstanceModel(
                setup_status_changed_on = current_time - 100,
                setup_status = ss.COMPLETED,
                setup_stage = stage.PROVISION
            ),
            InstanceModel(
                setup_status_changed_on = current_time - 4000,
                setup_status = ss.COMPLETED,
                setup_stage = stage.PROVISION
            ),
            InstanceModel(
                setup_status_changed_on = current_time - 5000,
                setup_status = ss.REQUESTED,
                setup_stage = stage.DEPLOY
            ),
        ]

        with patch.multiple(orm_worker, setup_instance = DEFAULT, deploy = DEFAULT) as mocks:

            restarted_jobs_count = orm_worker.move_jobs_to_next_phase(instances)

            self.assertEquals(mocks['setup_instance'].delay.call_count, 1)
            self.assertEquals(mocks['deploy'].delay.call_count, 1)

            self.assertEquals(restarted_jobs_count, 2)

    def test_control_succeeds(self):
        uid = database.ObjectId(None)
        instance_service = InstanceService(uid)
        instance = InstanceModel()
        instance_service.save(instance)

        instance_id = orm_worker.control(instance._id, stage.START)

        instance_db = instance_service.get(instance_id)
        self.assertTrue(instance_db)
        self.assertFalse(instance_db.setup_failed())
        self.assertEqual(instance_db.setup_stage, stage.START)
        self.assertEqual(instance_db.setup_status, ss.COMPLETED)

    def test_control_raises_exception(self):
        uid = database.ObjectId(None)
        instance_service = InstanceService(uid)
        instance = InstanceModel()
        instance_service.save(instance)

        self.AnsibleMock.control.side_effect = Exception('BOOM!')

        with self.assertRaises(Exception):
            orm_worker.control(instance._id, stage.STOP)

        instance_db = instance_service.get(instance._id)
        self.assertTrue(instance_db)
        self.assertTrue(instance_db.setup_failed())
        self.assertEqual(instance_db.setup_stage, stage.STOP)
        self.assertEqual(instance_db.setup_status, ss.FAILED)
        self.assertFalse(instance_db.is_deployed())

    def test_control_exits_if_job_is_in_process_already(self):
        uid = database.ObjectId(None)
        instance_service = InstanceService(uid)
        instance = InstanceModel(setup_status = ss.INPROCESS)
        instance_service.save(instance)

        orm_worker.control(instance._id, stage.STOP)

        self.assertFalse(self.AnsibleMock.control.called)

    def test_get_api_credentials_returns_existing_api_token(self):
        username = 'username'
        api_token = database.ObjectId(None)
        uid = self.persistent.create(table=USER_TABLE_NAME, values={'username': username,
            'api_token': api_token})

        actual_username, actual_api_token = orm_worker.get_api_credentials(InstanceModel(uid = uid))
        self.assertEqual(api_token, actual_api_token)
        self.assertEqual(username, actual_username)

    def test_get_api_credentials_creates_a_new_api_token(self):
        username = 'username'
        uid = self.persistent.create(table=USER_TABLE_NAME, values={'username': username})

        actual_username, actual_api_token = orm_worker.get_api_credentials(InstanceModel(uid = uid))

        db_user = self.persistent.read(table=USER_TABLE_NAME, condition={'_id': uid}, result = {})
        self.assertEqual(actual_api_token, db_user.get('api_token'), db_user)
        self.assertEqual(username, actual_username)

    @patch('ORM.orm_worker.requests', autospec =True)
    def test_request_prediction(self, MockRequests):
        MockRequests.post.return_value.status_code = 200

        host_name = 'private-instance.datarobot.com'
        credentials = ('username', 'token')
        model = {'pid': database.ObjectId(None), '_id': database.ObjectId(None)}
        data = '{}'

        result = orm_worker.request_prediction(
            host_name, credentials, model, data
        )
        self.assertTrue(result)

    @patch('ORM.orm_worker.requests', autospec =True)
    def test_request_raises_an_exception_on_failure(self, MockRequests):
        MockRequests.post.return_value.status_code = 400

        host_name = 'private-instance.datarobot.com'
        credentials = ('username', 'token')
        model = {'pid': database.ObjectId(None), '_id': database.ObjectId(None)}
        data = '{}'

        with self.assertRaises(Exception):
            orm_worker.request_prediction(
                host_name, credentials, model, data
            )



