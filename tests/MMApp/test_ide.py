####################################################################
#
#       Test for MMApp ide service class
#
#       Author: TC
#
#       Copyright DataRobot, Inc 2013
#
####################################################################

import unittest
import pytest
import time

from mock import Mock, patch, call, DEFAULT, sentinel

import MMApp.entities.ide
from MMApp.entities.ide import IdeService, IdeSetupStatus
from redis import WatchError

class IdeTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.uid = '313233343536373839303930'
        self.pid = '839303930313233343536373'

    def setUp(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    @pytest.mark.unit
    def test_status_init(self):
        self.assertRaises(ValueError, IdeSetupStatus, "bad_status")

    @pytest.mark.unit
    def test_status_dict(self):
        ide_status = IdeSetupStatus(IdeSetupStatus.NOT_STARTED, key="1", worker_id="1", timestamp="0")
        self.assertEqual(ide_status.to_dict(), {'status': ide_status.status,
            'key': ide_status.key,
            'worker_id': ide_status.worker_id,
            'remove': IdeSetupStatus.NOT_STARTED,
            'timestamp': ide_status.timestamp})

    @pytest.mark.unit
    def test_status_encoded_dict(self):
        ide_status = IdeSetupStatus(IdeSetupStatus.NOT_STARTED,
            key="1",
            worker_id="1",
            rstudio_location="0.0.0.0:1",
            python_location="0.0.0.0:2")
        self.assertEqual(ide_status.to_encoded_dict(), {'status': ide_status.status,
            'key': ide_status.key,
            'worker_id': ide_status.worker_id,
            'remove': IdeSetupStatus.NOT_STARTED,
            'rstudio_location': "0_0_0_1",
            'python_location': "0_0_0_2"})

    @pytest.mark.unit
    @patch('MMApp.entities.ide.IdeBrokerClient', autospec = True)
    def test_client_cache(self, MockIdeBrokerClient):
        MockIdeBrokerClient.return_value = "client"
        ide_service = IdeService(self.uid, self.pid)
        self.assertEqual(ide_service.client, "client")
        ide_service._client = "cached client"
        self.assertEqual(ide_service.client, "cached client")

    @pytest.mark.unit
    @patch('MMApp.entities.ide.IdeBrokerClient', autospec = True)
    def test_setup(self, MockIdeBrokerClient):

        ide_service = IdeService(self.uid, self.pid)
        request = dict.fromkeys(['uid', 'pid', 'dataset_id', 'original_filename'], 'x')

        with patch.dict(MMApp.entities.ide.EngConfig, {'MAX_IDE_SESSIONS_PER_USER': 2 }, clear = False):
            with patch('MMApp.entities.ide.IdeService', autospec=True) as mock_ideservice:
                with patch.multiple(ide_service, get_status = DEFAULT,
                                    worker_exists=DEFAULT,
                                    clear_session=DEFAULT,
                                    create_key=DEFAULT,
                                    set_status = DEFAULT,
                                    create_setup_request = DEFAULT,
                                    assign_worker=DEFAULT,
                                    get_user_sessions=DEFAULT,
                                    remove_inactive_users_from_workers=DEFAULT) as mocks:

                    ide_status = IdeSetupStatus(status = IdeSetupStatus.NOT_STARTED)
                    mocks['get_status'].return_value = ide_status
                    mocks['assign_worker'].return_value = IdeSetupStatus(status = IdeSetupStatus.STARTED, worker_id="1")
                    mocks['create_setup_request'].return_value = request
                    mocks['get_user_sessions'].return_value = set(['pid1', 'pid2'])

                    ide_service.setup()

                    self.assertTrue(mocks['create_setup_request'].called)
                    self.assertTrue(mock_ideservice.return_value.remove.called)

                    client = MockIdeBrokerClient.return_value
                    client.ide_setup.assert_called_once_with(mocks['assign_worker'].return_value.worker_id, **request)

    @pytest.mark.unit
    @patch('MMApp.entities.ide.IdeBrokerClient', autospec = True)
    def test_setup_worker_not_exist(self, MockIdeBrokerClient):

        ide_service = IdeService(self.uid, self.pid)
        request = dict.fromkeys(['uid', 'pid', 'dataset_id', 'original_filename'], 'x')

        with patch.multiple(ide_service, get_status = DEFAULT,
                            worker_exists=DEFAULT,
                            clear_session=DEFAULT,
                            create_key=DEFAULT,
                            set_status = DEFAULT,
                            create_setup_request = DEFAULT,
                            assign_worker=DEFAULT,
                            get_user_sessions=DEFAULT,
                            remove_inactive_users_from_workers=DEFAULT) as mocks:

            ide_status = IdeSetupStatus(status = IdeSetupStatus.STARTED, worker_id="1")
            mocks['get_status'].return_value = ide_status
            mocks['worker_exists'].return_value = False
            mocks['assign_worker'].return_value = ide_status
            mocks['create_setup_request'].return_value = request
            mocks['get_user_sessions'].return_value = set()

            self.assertEqual(ide_service.setup().to_dict(), ide_status.to_dict())

            self.assertTrue(mocks['clear_session'].called)
            self.assertTrue(mocks['assign_worker'].called)
            self.assertTrue(mocks['create_setup_request'].called)
            self.assertTrue(mocks['get_status'].called)

            client = MockIdeBrokerClient.return_value
            client.ide_setup.assert_called_once_with(mocks['assign_worker'].return_value.worker_id, **request)

    @pytest.mark.unit
    @patch('MMApp.entities.ide.IdeBrokerClient', autospec = True)
    def test_setup_workers_full(self, MockIdeBrokerClient):

        ide_service = IdeService(self.uid, self.pid)
        request = dict.fromkeys(['uid', 'pid', 'dataset_id', 'original_filename'], 'x')

        with patch.multiple(ide_service, get_status = DEFAULT,
                            worker_exists=DEFAULT,
                            clear_session=DEFAULT,
                            create_key=DEFAULT,
                            set_status = DEFAULT,
                            create_setup_request = DEFAULT,
                            assign_worker=DEFAULT,
                            get_user_sessions=DEFAULT,
                            remove_inactive_users_from_workers=DEFAULT) as mocks:

            ide_status = IdeSetupStatus(status = IdeSetupStatus.NOT_STARTED)
            mocks['get_status'].return_value = ide_status
            mocks['worker_exists'].return_value = True
            mocks['assign_worker'].return_value = ide_status
            mocks['create_setup_request'].return_value = request
            mocks['get_user_sessions'].return_value = set()

            self.assertEqual(ide_service.setup().to_dict(), ide_status.to_dict())

            self.assertTrue(mocks['remove_inactive_users_from_workers'].called)

    @pytest.mark.unit
    @patch('MMApp.entities.ide.IdeBrokerClient', autospec = True)
    def test_remove(self, MockIdeBrokerClient):

        ide_service = IdeService(self.uid, self.pid)
        request = dict.fromkeys(['uid', 'pid', 'dataset_id', 'original_filename'], 'x')

        with patch.multiple(ide_service, get_status = DEFAULT,
                            worker_exists=DEFAULT,
                            clear_session=DEFAULT,
                            create_key=DEFAULT,
                            set_remove_status = DEFAULT) as mocks:

            ide_status = IdeSetupStatus(status = IdeSetupStatus.NOT_STARTED, worker_id="1")
            mocks['get_status'].return_value = ide_status
            self.assertIsNone(ide_service.remove())

            ide_status = IdeSetupStatus(status = IdeSetupStatus.STARTED, worker_id="1")
            mocks['get_status'].return_value = ide_status
            self.assertIsNone(ide_service.remove())
            self.assertTrue(mocks['clear_session'].called)

            ide_status = IdeSetupStatus(status = IdeSetupStatus.IN_PROCESS)
            mocks['get_status'].return_value = ide_status
            self.assertIsNone(ide_service.remove())

            ide_status = IdeSetupStatus(status = IdeSetupStatus.IN_PROCESS, worker_id="1")
            mocks['get_status'].return_value = ide_status
            self.assertIsNone(ide_service.remove())
            self.assertTrue(mocks['set_remove_status'].called)

            client = MockIdeBrokerClient.return_value
            client.ide_remove.assert_called_once_with("1", self.uid, self.pid, save=True)

    @pytest.mark.unit
    @patch('MMApp.entities.ide.IdeBrokerClient', autospec = True)
    def test_remove_worker_not_exist(self, MockIdeBrokerClient):

        ide_service = IdeService(self.uid, self.pid)
        request = dict.fromkeys(['uid', 'pid', 'dataset_id', 'original_filename'], 'x')

        with patch.multiple(ide_service, get_status = DEFAULT,
                            worker_exists=DEFAULT,
                            clear_session=DEFAULT,
                            create_key=DEFAULT,
                            set_remove_status = DEFAULT) as mocks:

            ide_status = IdeSetupStatus(status = IdeSetupStatus.IN_PROCESS, worker_id="1", remove=IdeSetupStatus.STARTED)
            mocks['get_status'].return_value = ide_status
            mocks['worker_exists'].return_value = False
            self.assertIsNone(ide_service.remove())
            self.assertTrue(mocks['clear_session'].called)

    def test_confirm_setup_request(self):
        ide_service = IdeService(self.uid, self.pid)
        request = dict.fromkeys(['uid', 'pid', 'dataset_id', 'original_filename'], 'x')

        with patch.multiple(ide_service, get_status = DEFAULT,
                            set_status=DEFAULT,
                            tempstore=DEFAULT,
                            remove_inactive_users_from_worker=DEFAULT) as mocks:
            ide_status = IdeSetupStatus(status = IdeSetupStatus.NOT_STARTED)
            mocks['get_status'].return_value = ide_status
            self.assertFalse(ide_service.confirm_setup_request("worker_id"))

            ide_status = IdeSetupStatus(status = IdeSetupStatus.STARTED)
            mocks['get_status'].return_value = ide_status
            self.assertTrue(ide_service.confirm_setup_request("worker_id"))
            self.assertTrue(mocks['set_status'].called)

    def test_confirm_remove_request(self):
        ide_service = IdeService(self.uid, self.pid)
        request = dict.fromkeys(['uid', 'pid', 'dataset_id', 'original_filename'], 'x')

        with patch.multiple(ide_service, get_status = DEFAULT,
                            set_remove_status=DEFAULT,
                            tempstore=DEFAULT) as mocks:
            ide_status = IdeSetupStatus(status = IdeSetupStatus.COMPLETED, remove=IdeSetupStatus.NOT_STARTED)
            mocks['get_status'].return_value = ide_status
            self.assertFalse(ide_service.confirm_remove_request())

            ide_status = IdeSetupStatus(status = IdeSetupStatus.COMPLETED, remove=IdeSetupStatus.STARTED)
            mocks['get_status'].return_value = ide_status
            self.assertTrue(ide_service.confirm_remove_request())
            self.assertTrue(mocks['set_remove_status'].called)

    @pytest.mark.unit
    def test_expire_status(self):
        ide_service = IdeService(self.uid, self.pid)
        with patch.multiple(ide_service, tempstore=DEFAULT) as mocks:
            ide_service.expire_status()
            self.assertTrue(mocks['tempstore'].destroy.called)

    @pytest.mark.unit
    def test_worker_exists(self):
        ide_service = IdeService(self.uid, self.pid)
        with patch.multiple(ide_service, tempstore=DEFAULT) as mocks:
            self.assertTrue(ide_service.worker_exists("worker_id"))
            self.assertTrue(mocks['tempstore'].conn.sismember.called)
            mocks['tempstore'].conn.sismember.return_value = False
            self.assertFalse(ide_service.worker_exists("worker_id"))

    @pytest.mark.unit
    def test_create_key(self):
        ide_service = IdeService(self.uid, self.pid)
        key = ide_service.create_key()
        self.assertTrue(isinstance(key, str))
        self.assertEqual(len(key), 16)

    @pytest.mark.unit
    @patch('MMApp.entities.ide.IdeBrokerClient', autospec = True)
    def test_in_process_setup(self, MockIdeBrokerClient):

        ide_service = IdeService(self.uid, self.pid)
        request = dict.fromkeys(['uid', 'pid', 'dataset_id', 'original_filename'], 'x')

        with patch.multiple(ide_service, get_status = DEFAULT,
                            create_setup_request = DEFAULT,
                            worker_exists = DEFAULT,
                            assign_worker=DEFAULT,
                            get_user_sessions=DEFAULT,
                            remove_inactive_users_from_workers=DEFAULT) as mocks:
            ide_status = IdeSetupStatus(status = IdeSetupStatus.STARTED)
            mocks['get_status'].return_value = ide_status
            mocks['create_setup_request'].return_value = request
            mocks['worker_exists'].return_value = True
            mocks['get_user_sessions'].return_value = set()

            ide_service.setup()

            self.assertTrue(mocks['get_status'].called)

            client = MockIdeBrokerClient.return_value
            self.assertFalse(client.ide_setup.called)

    @pytest.mark.unit
    @patch('MMApp.entities.ide.FileTransaction')
    @patch('MMApp.entities.ide.ProjectService', autospec = True)
    def test_create_setup_request(self, MockProjectService, MockFileTransaction):
        dataset_id = '343536373839303930313233'
        original_file_name = 'original_file_name.ext'
        mock_service = MockProjectService.return_value
        mock_service.get.return_value = {'default_dataset_id' : dataset_id}
        mock_service.get_metadata.return_value = {'originalName' : original_file_name, 'files': [dataset_id]}
        MockFileTransaction.exists.return_value = True

        ide_service = IdeService(self.uid, self.pid)
        key = 'key'
        request = ide_service.create_setup_request(key)

        self.assertEqual(request['uid'], self.uid)
        self.assertEqual(request['pid'], self.pid)
        self.assertEqual(request['key'], key)
        self.assertEqual(request['files'], [dataset_id + ext for ext in ['', '.pkl', '.Rdata']])

    @pytest.mark.unit
    @patch('MMApp.entities.ide.FileTransaction')
    @patch('MMApp.entities.ide.ProjectService', autospec = True)
    def test_create_setup_request_project_failure(self, MockProjectService, MockFileTransaction):
        dataset_id = '343536373839303930313233'
        original_file_name = 'original_file_name.ext'
        mock_service = MockProjectService.return_value
        mock_service.get.return_value = {'default_dataset_id' : dataset_id}
        mock_service.get_metadata.side_effect = Exception()
        MockFileTransaction.exists.return_value = True

        ide_service = IdeService(self.uid, self.pid)
        key = 'key'
        request = ide_service.create_setup_request(key)

        self.assertEqual(request['uid'], self.uid)
        self.assertEqual(request['pid'], self.pid)
        self.assertEqual(request['key'], key)
        self.assertFalse('files' in request)

    @pytest.mark.unit
    def test_set_status(self):
        ide_service = IdeService(self.uid, self.pid)
        ide_status = IdeSetupStatus(status = IdeSetupStatus.NOT_STARTED)

        with patch.multiple(ide_service, upsert_session = DEFAULT,
                            tempstore=DEFAULT,
                            clear_session=DEFAULT,
                            expire_status = DEFAULT) as mocks:
            ide_service.set_status(ide_status)

            expected_session = {'status' : IdeSetupStatus.NOT_STARTED, 'remove': IdeSetupStatus.NOT_STARTED}
            self.assertItemsEqual(ide_status.to_dict() , expected_session)

            mocks['upsert_session'].assert_called_once_with(ide_status.to_dict())

    @pytest.mark.unit
    def test_set_status_as_logout_started(self):
        ide_service = IdeService(self.uid, self.pid)

        with patch.multiple(ide_service, upsert_session = DEFAULT,
                            tempstore=DEFAULT,
                            expire_status = DEFAULT) as mocks:
            #LOGOUT STARTED - Arrange
            ide_status = IdeSetupStatus(status = IdeSetupStatus.STARTED)
            #Act
            ide_service.set_status(ide_status)
            #Assert
            expected_session = dict(status = IdeSetupStatus.STARTED, remove = IdeSetupStatus.NOT_STARTED)
            self.assertItemsEqual(ide_status.to_dict() , expected_session)

            mocks['upsert_session'].assert_called_once_with(ide_status.to_dict())

    @pytest.mark.unit
    @patch('MMApp.entities.ide.UserService', autospec = True)
    def test_remove_inactive_users_from_worker(self, mock_userservice):
        ide_service = IdeService(self.uid, self.pid)

        with patch('MMApp.entities.ide.IdeService', autospec=True) as mock_ideservice:
            with patch.multiple(ide_service, get_worker_sessions = DEFAULT) as mocks:
                mocks['get_worker_sessions'].return_value = ['ide:uid:pid']
                mock_userservice.return_value.is_online.return_value = False
                ide_service.remove_inactive_users_from_worker('worker_id')
                self.assertTrue(mock_ideservice.return_value.remove.called)

    @pytest.mark.unit
    def test_remove_inactive_users_from_workers(self):
        ide_service = IdeService(self.uid, self.pid)

        with patch.multiple(ide_service, tempstore = DEFAULT,
                            remove_inactive_users_from_worker=DEFAULT) as mocks:
            mocks['tempstore'].conn.smembers.return_value = set(['worker_id'])
            self.assertIsNone(ide_service.remove_inactive_users_from_workers())
            self.assertTrue(mocks['remove_inactive_users_from_worker'].called)

    @pytest.mark.unit
    def test_remove_all_users_from_worker(self):
        with patch.multiple(IdeService, get_worker_sessions=DEFAULT, remove=DEFAULT) as mocks:
            with patch('MMApp.entities.ide.IdeService', autospec=True) as mock_ideservice:
                mocks['get_worker_sessions'].return_value = ['ide:uid:pid']
                IdeService.remove_all_users_from_worker('worker_id')
                self.assertTrue(mock_ideservice.return_value.remove.called)

    @pytest.mark.unit
    def test_set_status_as_logout_in_process(self):
        ide_service = IdeService(self.uid, self.pid)
        with patch.multiple(ide_service, upsert_session = DEFAULT,
                            clear_session=DEFAULT,
                            get_status = DEFAULT,
                            tempstore=DEFAULT) as mocks:
            #LOGOUT IN-PROCESS - Arrange
            ide_status = IdeSetupStatus(status = IdeSetupStatus.COMPLETED)
            #Act
            ide_service.set_status(ide_status)
            ide_service.set_remove_status(IdeSetupStatus.IN_PROCESS)
            mocks['upsert_session'].assert_has_calls([
                call(ide_status.to_dict()),
                call({'remove': IdeSetupStatus.IN_PROCESS})
                ])

    @pytest.mark.unit
    def test_set_status_as_logout_completed(self):
        ide_service = IdeService(self.uid, self.pid)
        with patch.multiple(ide_service, upsert_session = DEFAULT,
                            clear_session = DEFAULT,
                            get_status=DEFAULT,
                            tempstore=DEFAULT) as mocks:
            ide_service.set_remove_status(IdeSetupStatus.COMPLETED)
            self.assertTrue(mocks['clear_session'].called)

    @pytest.mark.unit
    def test_set_status_as_failed(self):
        """
        Test FAILD is removed (in order to retry again in case )
        """
        ide_service = IdeService(self.uid, self.pid)
        ide_status = IdeSetupStatus(status = IdeSetupStatus.FAILED )
        ide_status.username = 'user'
        ide_status.password = 'pass'
        ide_status.location = '127.0.0.1:1365'
        with patch.multiple(ide_service, upsert_session = DEFAULT,
                            clear_session = DEFAULT,
                            get_status=DEFAULT,
                            tempstore=DEFAULT) as mocks:
            # Get encode_location out of the way
            with patch.object(ide_status, 'encode_location', return_value = ide_status.location):
                actual_session = ide_service.set_status(ide_status)

                self.assertTrue(mocks['clear_session'].called)

    @pytest.mark.unit
    def test_set_status_as_started(self):
        """
        Test STARTED is temporary (in order to retry again in case )
        """
        ide_service = IdeService(self.uid, self.pid)
        ide_status = IdeSetupStatus(status = IdeSetupStatus.STARTED )
        ide_status.username = 'user'
        ide_status.password = 'pass'
        ide_status.location = '127.0.0.1:1365'
        with patch.multiple(ide_service, upsert_session = DEFAULT,
                            expire_status = DEFAULT,
                            tempstore=DEFAULT) as mocks:
            # Get encode_location out of the way
            with patch.object(ide_status, 'encode_location', return_value = ide_status.location):
                actual_session = ide_service.set_status(ide_status)

                mocks['upsert_session'].assert_called_once_with(ide_status.to_dict())

    @pytest.mark.unit
    def test_set_status_as_complete(self):

        ide_service = IdeService(self.uid, self.pid)
        ide_status = IdeSetupStatus(status = IdeSetupStatus.COMPLETED )

        # Make sure a good call sets the right flags and returns the right session info
        ide_status.username = 'user'
        ide_status.password = 'pass'
        ide_status.rstudio_location = '127.0.0.1:1365'
        ide_status.python_location = '127.0.0.1:1366'
        with patch.multiple(ide_service, upsert_session = DEFAULT,
                            expire_status = DEFAULT,
                            tempstore=DEFAULT) as mocks:
            # Get encode_location out of the way
            with patch.object(ide_status, 'encode_location', return_value = ide_status.rstudio_location):
                ide_service.set_status(ide_status)

                expected_session = dict(status = IdeSetupStatus.COMPLETED,
                username = ide_status.username, password = ide_status.password,
                rstudio_location = ide_status.rstudio_location, python_location=ide_status.rstudio_location,
                remove = IdeSetupStatus.IN_PROCESS)

                self.assertItemsEqual(ide_status.to_dict(), expected_session)
                mocks['upsert_session'].assert_called_once_with(ide_status.to_dict())


    @pytest.mark.unit
    def test_get_status(self):
        #Arrange: No session/Not started
        ide_service = IdeService(self.uid, self.pid)

        with patch.object(ide_service, 'tempstore') as mock_db:
            mock_db.read.return_value = {}

            actual_session = ide_service.get_status()
            expected_session = {'status' : IdeSetupStatus.NOT_STARTED, 'remove': IdeSetupStatus.NOT_STARTED}
            self.assertItemsEqual(actual_session.to_dict(), expected_session)

    @pytest.mark.unit
    @patch('MMApp.entities.ide.database', autospec = True)
    def test_get_worker_sessions(self, mock_db):
        tempstore = mock_db.new.return_value
        tempstore.conn.smembers.return_value = set(['ide:uid:pid'])
        self.assertEqual(IdeService.get_worker_sessions('worker_id'), tempstore.conn.smembers.return_value)

    @pytest.mark.unit
    def test_get_status_when_started(self):
        ide_service = IdeService(self.uid, self.pid)

        with patch.object(ide_service, 'tempstore') as mock_db:
            #Arrange: Started
            mock_db.read.return_value = {'status' : IdeSetupStatus.STARTED}
            #Act
            actual_session = ide_service.get_status()

            #Assert
            expected_session = {'status' : IdeSetupStatus.STARTED, 'remove': IdeSetupStatus.NOT_STARTED}
            self.assertItemsEqual(actual_session.to_dict(), expected_session)

    @pytest.mark.unit
    def test_get_status_when_complete(self):
        ide_service = IdeService(self.uid, self.pid)

        with patch.object(ide_service, 'tempstore') as mock_db:
            # Arrange: Complete (get encode_location out of the way)
            location = '127.0.0.1:1365'
            python_location = '127.0.0.1:1366'

            mock_db.read.return_value = {'status' : IdeSetupStatus.COMPLETED,
            'rstudio_location': location, 'python_location': python_location,
            'username' : 'user', 'password': 'pass', 'remove': IdeSetupStatus.NOT_STARTED}

            actual_session = ide_service.get_status()

            #Assert
            expected_session = {'status' : IdeSetupStatus.COMPLETED,
                                'rstudio_location': location, 'python_location': python_location, 'username': 'user',
                                'password': 'pass', 'remove': IdeSetupStatus.NOT_STARTED}
            self.assertItemsEqual(actual_session.to_dict(), expected_session)

    @pytest.mark.unit
    def test_get_status_uses_right_keys(self):
        ide_service = IdeService(self.uid, self.pid)

        with patch.object(ide_service, 'tempstore') as mock_db:
            mock_db.read.return_value = {}
            session = ide_service.get_status()
            mock_db.read.assert_called_with(table='ide',
                                            keyname=self.uid,
                                            index=self.pid,
                                            result={})

    @pytest.mark.unit
    def test_encode_location(self):
        ide_status = IdeSetupStatus(status = IdeSetupStatus.COMPLETED )

        location = '127.0.0.1:1365'
        actual_location = ide_status.encode_location(location)
        expected_location = '0_0_1_1365'

        self.assertEqual(actual_location, expected_location)


        location = '192.168.1.115:18965'
        actual_location = ide_status.encode_location(location)
        expected_location = '168_1_115_18965'

        self.assertEqual(actual_location, expected_location)

    @pytest.mark.unit
    @patch('MMApp.entities.ide.IdeBrokerClient', autospec = True)
    def test_logout(self, MockIdeBrokerClient):
        ide_service = IdeService(self.uid, self.pid)

        with patch.multiple(ide_service, get_status = DEFAULT,
                            set_remove_status = DEFAULT,
                            worker_exists = DEFAULT,
                            clear_session = DEFAULT) as mocks:
            mocks['get_status'].return_value = IdeSetupStatus(status = IdeSetupStatus.COMPLETED, worker_id=1)
            ide_service.remove(save=True)
            self.assertTrue(mocks['set_remove_status'].called)
            client = MockIdeBrokerClient.return_value
            client.ide_remove.assert_called_once_with(1, self.uid, self.pid, save=True)

    @pytest.mark.unit
    def test_upsert_session(self):
        ide_service = IdeService(self.uid, self.pid)
        ide_status = IdeSetupStatus(status = IdeSetupStatus.STARTED)
        ide_status.username = 'user'
        ide_status.password = 'pass'
        ide_status.rstudio_location = '127.0.0.1:1365'

        with patch.object(ide_service, 'tempstore') as mock_db:
            mock_db.read.return_value = {}
            ide_service.upsert_session(ide_status.to_dict())
            self.assertTrue(mock_db.update.called)

    @pytest.mark.unit
    def test_clear_session(self):
        ide_service = IdeService(self.uid, self.pid)
        ide_status = IdeSetupStatus(status = IdeSetupStatus.FAILED)
        ide_status.username = 'user'
        ide_status.password = 'pass'
        ide_status.rstudio_location = '127.0.0.1:1365'
        ide_status.worker_id = "1"

        with patch.object(ide_service, 'tempstore') as mock_db:

            self.assertIsNone(ide_service.clear_session(ide_status))

            table, keyname, index = ide_service.user_session.split(':')
            mock_db.destroy.assert_has_calls([
                call(table=table, keyname=keyname, index=index),
                call(keyname=ide_service.USER_SESSION_KEY_PREFIX + ide_service.uid, fields={ide_service.pid}),
                call(keyname=ide_service.WORKER_SESSION_KEY_PREFIX + ide_status.worker_id, fields={ide_service.user_session}),
                call(keyname=ide_service.WORKERS_WITHOUT_RESOURCES, fields={ide_status.worker_id})
                ])

    @pytest.mark.unit
    def test_report_error_before_setup(self):
        ide_service = IdeService(self.uid, self.pid)
        with patch.multiple(ide_service, get_status = DEFAULT,
                            worker_exists = DEFAULT,
                            clear_session = DEFAULT,
                            ide_server_is_running = DEFAULT,
                            remove = DEFAULT) as mocks:

            mocks['get_status'].return_value = IdeSetupStatus(status = IdeSetupStatus.STARTED, timestamp=-1)
            ide_service.report_error()
            self.assertFalse(mocks['remove'].called)


    @pytest.mark.unit
    def test_report_error_before_timeout(self):
        ide_service = IdeService(self.uid, self.pid)
        with patch.multiple(ide_service, get_status = DEFAULT,
                            worker_exists = DEFAULT,
                            clear_session = DEFAULT,
                            ide_server_is_running = DEFAULT,
                            remove = DEFAULT) as mocks:

            mocks['get_status'].return_value = IdeSetupStatus(status = IdeSetupStatus.COMPLETED, timestamp=time.time() + 600)
            ide_service.report_error()
            self.assertFalse(mocks['remove'].called)

    @pytest.mark.unit
    def test_report_error_worker_does_not_exist(self):
        ide_service = IdeService(self.uid, self.pid)
        with patch.multiple(ide_service, get_status = DEFAULT,
                            worker_exists = DEFAULT,
                            clear_session = DEFAULT,
                            ide_server_is_running = DEFAULT,
                            remove = DEFAULT) as mocks:

            mocks['get_status'].return_value = IdeSetupStatus(status = IdeSetupStatus.COMPLETED, timestamp=-1)
            mocks['worker_exists'].return_value = False
            ide_service.report_error()
            self.assertTrue(mocks['clear_session'].called)
            self.assertFalse(mocks['remove'].called)

    @pytest.mark.unit
    def test_report_error_valid(self):
        ide_service = IdeService(self.uid, self.pid)
        with patch.multiple(ide_service, get_status = DEFAULT,
                            worker_exists = DEFAULT,
                            clear_session = DEFAULT,
                            ide_server_is_running = DEFAULT,
                            remove = DEFAULT) as mocks:

            mocks['get_status'].return_value = IdeSetupStatus(status = IdeSetupStatus.COMPLETED, timestamp=-1)
            mocks['worker_exists'].return_value = True
            ide_service.report_error()
            self.assertFalse(mocks['clear_session'].called)
            self.assertTrue(mocks['remove'].called)


    @patch('MMApp.entities.ide.requests', autospec = True)
    def test_ide_server_is_running(self, MockRequests):
        ide_service = IdeService(self.uid, self.pid)
        ide_status = IdeSetupStatus(status = IdeSetupStatus.COMPLETED)
        ide_status.timestamp = time.time() - 100
        ide_status.python_location = 'http://'
        ide_status.rstudio_location = 'http://'

        MockRequests.get.return_value.status_code = 200

        result = ide_service.ide_server_is_running(ide_status)

        self.assertTrue(result)

    def test_parse_user_session(self):
        ide_service = IdeService(self.uid, self.pid)
        #make sure the way user sessions are created hasn't changed
        self.assertEqual(ide_service.user_session, "ide:{0}:{1}".format(self.uid, self.pid))
        self.assertEqual(ide_service.parse_user_session(ide_service.user_session), (self.uid,self.pid))

    @patch('MMApp.entities.ide.database', autospec = True)
    def test_register_worker(self, mock_db):
        with patch.dict(MMApp.entities.ide.EngConfig,
                        {'REQUEST_ACCOUNTING': False},
                        clear=False):
            tempstore = mock_db.new.return_value
            tempstore.conn.incr.return_value = "worker_id"

            ide_service = IdeService(self.uid, self.pid)
            self.assertEqual(ide_service.register_worker({}), "worker_id")

    @patch('MMApp.entities.ide.database', autospec = True)
    def test_remove_worker_resources(self, mock_db):
        tempstore = mock_db.new.return_value
        tempstore.conn.delete.return_value = 1

        ide_service = IdeService(self.uid, self.pid)
        self.assertEqual(ide_service.remove_worker_resources("1"), 1)

    @pytest.mark.unit
    @patch('MMApp.entities.ide.time', autospec = True)
    def test_assign_worker(self, mock_time):
        mock_time.time = (i for i in xrange(100)).next
        ide_service = IdeService(self.uid, self.pid)
        ide_status = IdeSetupStatus(status = IdeSetupStatus.NOT_STARTED)

        with patch.multiple(ide_service, tempstore=DEFAULT, create_key=DEFAULT) as mocks:
            mocks['create_key'].return_value = "key"
            pipe = mocks['tempstore'].conn.pipeline.return_value

            #no workers available
            pipe.sdiff.return_value = set()
            self.assertEqual(ide_service.assign_worker().to_dict(), ide_status.to_dict())

            #resources were removed
            pipe.sdiff.return_value = set(["1"])
            pipe.hget.return_value = "0"
            pipe.scard.return_value = "0"
            self.assertEqual(ide_service.assign_worker().to_dict(), ide_status.to_dict())

            #success
            pipe.hget.return_value = "1"
            self.assertEqual(ide_service.assign_worker().to_dict(),
                IdeSetupStatus(status = IdeSetupStatus.STARTED, worker_id="1", key="key").to_dict())

            #redis watch error
            pipe.watch.side_effect = WatchError
            self.assertEqual(ide_service.assign_worker().to_dict(), ide_status.to_dict())

    @patch('MMApp.entities.ide.FileTransaction', autospec = True)
    def test_delete(self, mock_filetransaction):
        ide_service = IdeService(self.uid, self.pid)
        with patch.multiple(ide_service, remove=DEFAULT) as mocks:
            self.assertIsNone(ide_service.delete())
            #confirm the filename format hasn't changed (until it's refactored as an IDE service attribute)
            mock_filetransaction.assert_called_with("projects/%s/ide/%s" % (self.pid, self.uid))
            mocks['remove'].assert_called_once_with(save=False)

    @pytest.mark.unit
    def test_get_user_sessions(self):
        ide_service = IdeService(self.uid, self.pid)

        with patch.object(ide_service, 'tempstore') as mock_db:
            mock_db.conn.smembers.return_value = set(["pid"])
            self.assertEqual(ide_service.get_user_sessions(), mock_db.conn.smembers.return_value)


if __name__ == '__main__':
    unittest.main()

