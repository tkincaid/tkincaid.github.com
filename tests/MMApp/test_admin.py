import unittest
from mock import Mock, patch
import pytest

from bson import ObjectId
from MMApp.entities.user import UserModel
from MMApp.entities.user import USER_TABLE_NAME
import MMApp.entities.user as umod
from MMApp.entities.admin import UserFlagsAdmin
from MMApp.entities.admin import AdminAccessError
from MMApp.entities.permissions import Permissions

from config.test_config import db_config
from common.wrappers import database

def a_user_model(username='user@domain.com', password='hashpass',
                 first_name='Barack', last_name='Romney',
                 phone='808.555.1234', company='The Government',
                 industry='The Government', display_name='Data Champ',
                 max_workers=4, flags=None):
    if flags is None:
        flags = {}
    return UserModel(username=username, password=password,
                     first_name=first_name, last_name=last_name,
                     phone=phone, company=company, industry=industry,
                     display_name=display_name, max_workers=max_workers,
                     flags=flags)

class TestUserFlagsAdmin(unittest.TestCase):

    def setUp(self):
        self.mock_db = Mock()

    def test_find_existing_user_should_return_user_model(self):
        admin = UserFlagsAdmin(persistent=self.mock_db)
        uid = ObjectId()
        user = a_user_model()
        self.mock_db.read.return_value = user.to_dict()

        with patch.object(admin, 'validate_admin') as fake_validation:
            fake_validation.return_value = True
            returned_user = admin.find_user(uid)
            self.assertEqual(user.username, returned_user.username)

    def test_find_for_nonexistent_uid_returns_none(self):
        admin = UserFlagsAdmin(persistent=self.mock_db)
        uid = ObjectId()
        self.mock_db.read.return_value = {}

        with patch.object(admin, 'validate_admin') as fake_validation:
            fake_validation.return_value = True
            self.assertIsNone(admin.find_user(uid))

    def test_unauthenticated_writes_refused(self):
        admin = UserFlagsAdmin(persistent=self.mock_db)
        uid = ObjectId()
        self.mock_db.read.return_value = a_user_model().to_dict()

        with patch.object(admin, 'validate_admin') as fake_validation:
            fake_validation.side_effect = AdminAccessError()
            self.assertRaises(AdminAccessError, admin.set_user_flags, uid,
                              {'flag' : True})

    def test_unauthenticated_reads_refused(self):
        admin = UserFlagsAdmin(persistent=self.mock_db)
        uid = ObjectId()
        self.mock_db.read.return_value = a_user_model().to_dict()

        with patch.object(admin, 'validate_admin') as fake_validation:
            fake_validation.side_effect = AdminAccessError()
            self.assertRaises(AdminAccessError, admin.find_user, uid)

    @pytest.mark.unit
    def test_admin_cannot_deactivate_his_own_account(self):
        uid = ObjectId()
        admin = UserFlagsAdmin(uid, persistent=self.mock_db)
        with patch.object(admin, 'validate_admin') as fake_validation:
            self.assertRaises(AdminAccessError, admin.activate_account, uid, False)

    def test_get_new_permissions(self):
        permissions = {
            'PERMISSION_1' : True,
            'PERMISSION_2' : False,
            'PERMISSION_3' : True,
            'PERMISSION_5' : True

        }

        old_permissions = {
            'PERMISSION_1' : True,
            'PERMISSION_2' : True,
            'PERMISSION_4' : True,
            'PERMISSION_5' : True
        }

        new_permissions = UserFlagsAdmin().get_new_permissions(permissions, old_permissions)
        self.assertTrue(new_permissions)
        self.assertEqual(len(new_permissions.keys()), 2)

        self.assertFalse(new_permissions['PERMISSION_2'])
        self.assertTrue(new_permissions['PERMISSION_3'])

class TestUserFlagsDBInterface(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.persistent = database.new('persistent')

    def setUp(self):
        self.persistent.conn[USER_TABLE_NAME].drop()
        self.user = a_user_model()
        self.uid = self.persistent.create(table=USER_TABLE_NAME, values=self.user.to_dict())
        self.admin_user = a_user_model(username='admin@boss.com', first_name='Admin',
                                       last_name='Boss')
        self.admin_user.flags.CAN_MANAGE_APP_USERS = True
        self.admin_uid = self.persistent.create(table=USER_TABLE_NAME,
                                                values=self.admin_user.to_dict())

    @pytest.mark.db
    def test_set_user_flags(self):
        admin = UserFlagsAdmin(self.admin_uid, persistent=self.persistent)
        permissions = {
            Permissions.GRAYBOX_ENABLED : True,
            Permissions.IS_OFFLINE_QUEUE_ENABLED:True
        }
        result = admin.set_user_flags(self.uid, permissions)
        db_user = admin.find_user(self.uid)
        self.assertTrue(db_user.flags.GRAYBOX_ENABLED)
        self.assertTrue(db_user.flags.IS_OFFLINE_QUEUE_ENABLED)


        permissions = {
            Permissions.GRAYBOX_ENABLED : False,
        }
        result = admin.set_user_flags(self.uid, permissions)
        db_user = admin.find_user(self.uid)
        self.assertFalse(db_user.flags.GRAYBOX_ENABLED)
        self.assertTrue(db_user.flags.IS_OFFLINE_QUEUE_ENABLED, db_user.flags.to_dict())

    @pytest.mark.db
    def test_cant_remove_own_admin_privileges(self):
         admin = UserFlagsAdmin(self.admin_uid, persistent=self.persistent)
         with self.assertRaises(AdminAccessError):
             result = admin.set_user_flags(self.admin_uid,
                                          {Permissions.CAN_MANAGE_APP_USERS:False})

    @pytest.mark.db
    def test_unauthenticated_reads_refused(self):
        admin = UserFlagsAdmin(self.uid, persistent=self.persistent)
        with self.assertRaises(AdminAccessError):
            result = admin.set_user_flags(self.uid,
                                         {Permissions.GRAYBOX_ENABLED:False})

    @pytest.mark.db
    def test_set_user_flags_unknown_flag(self):
        admin = UserFlagsAdmin(self.admin_uid, persistent=self.persistent)
        with self.assertRaises(umod.UnknownFlagError):
            result = admin.set_user_flags(self.uid, {'not_a_known_key': True})

    @pytest.mark.db
    def test_set_user_flags_unknown_user(self):
        admin = UserFlagsAdmin(self.admin_uid, persistent=self.persistent)
        with self.assertRaises(umod.UserNotFoundError):
            result = admin.set_user_flags(ObjectId(), {Permissions.GRAYBOX_ENABLED: True})

    @pytest.mark.db
    def test_set_workers(self):
        admin = UserFlagsAdmin(self.admin_uid, persistent=self.persistent)
        db_user = admin.find_user(self.uid)
        new_worker_count = db_user.max_workers + 2
        db_user.max_workers = new_worker_count

        admin.save_profile(db_user)

        db_user = admin.find_user(self.uid)
        self.assertEqual(db_user.max_workers, new_worker_count)

    @pytest.mark.db
    def test_activate_account(self):
        admin = UserFlagsAdmin(self.admin_uid, persistent=self.persistent)

        admin.activate_account(self.uid, True)

        user = admin.find_user(self.uid)
        self.assertTrue(user.activated)

        admin.activate_account(self.uid, False)

        user = admin.find_user(self.uid)
        self.assertFalse(user.activated)

class TestSearchForUsers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.persistent = database.new('persistent')
        cls.persistent.conn[USER_TABLE_NAME].drop()
        cls.admin_user = a_user_model(username='admin@boss.com',
                                      first_name='Frank',
                                      last_name='Franklin')
        cls.admin_user.flags.CAN_MANAGE_APP_USERS = True
        cls.admin_uid = cls.persistent.create(table=USER_TABLE_NAME,
                                               values=cls.admin_user.to_dict())

        cls.first_names = ['Alex', 'Ahalya', 'Bill', 'Cameron', 'Delilah', 'Emily']
        cls.last_names = ['Zimmerman', 'Zeta', 'Young', 'Williams', 'Vasquez', 'Tailor']
        for first in cls.first_names:
            for last in cls.last_names:
                email= '{}.{}@work.com'.format(first, last)
                user = a_user_model(username=email, first_name=first,
                                         last_name=last)
                cls.persistent.create(table=USER_TABLE_NAME, values=user.to_dict())

    def setUp(self):
        self.admin = UserFlagsAdmin(self.admin_uid, self.persistent)


    @pytest.mark.db
    def test_search_for_user_by_domain_brings_back_regex_matches(self):
        search_doc = {'username' : '.*@work.com'}
        users = self.admin.search_users(search_doc)
        self.assertEqual(len(users), len(self.first_names)*len(self.last_names))

    @pytest.mark.db
    def test_search_for_user_by_username_brings_back_regex_matches(self):
        search_doc = {'username' : 'Alex'}
        users = self.admin.search_users(search_doc)
        self.assertEqual(len(users), len(self.last_names))

    @pytest.mark.db
    def test_search_for_user_by_firstname_brings_back_regex_matches(self):
        search_doc = {'first_name' : 'A'}
        users = self.admin.search_users(search_doc)
        self.assertGreater(len(users), 0)
        for user in users:
            self.assertIn(user.first_name, ['Alex', 'Ahalya'])

    @pytest.mark.db
    def test_search_for_user_by_lastname_brings_back_regex_matches(self):
        search_doc = {'last_name' : 'Z'}
        users = self.admin.search_users(search_doc)
        self.assertGreater(len(users), 0)
        for user in users:
            self.assertIn(user.last_name, ['Zimmerman', 'Zeta'])

    @pytest.mark.db
    def test_search_for_user_by_lastname_and_username(self):
        search_doc = {
            'last_name' : 'Z',
            'first_name' : 'Z',
            'username' : 'Z'
        }
        users = self.admin.search_users(search_doc)
        self.assertGreater(len(users), 10)

    @pytest.mark.db
    def test_search_users_by_lastname_and_username_with_helper_function(self):
        users = self.admin.search_users_by_name_and_username('z')
        self.assertGreater(len(users), 10)



